import os
import torch
import numpy as np
from typing import Optional, List

from camera_control.Module.rgbd_camera import RGBDCamera

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from robust_vggt import filter_valid_indices

class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
        device: str = 'cuda:0',
    ) -> None:
        self.device = device

        self.model = VGGT()

        if model_file_path is not None:
            self.loadModel(model_file_path, device)
        return

    def loadModel(
        self,
        model_file_path: str,
        device: str = 'cuda:0',
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        self.device = device

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        return True

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
        robust_mode: bool=True,
        cos_thresh: float=0.95,
    ) -> Optional[dict]:
        if images.shape[0] == 0:
            print('[ERROR][Detector::detect]')
            print("\t images are empty!")
            return None

        print(f"Preprocessed images shape: {images.shape}")

        # Run inference
        print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Robust mode: 使用 filter_valid_indices 筛选有效帧，然后二次推理
        rejected_indices = []
        rejected_extrinsics = None
        num_total_images = images.shape[0]

        if robust_mode and num_total_images > 1:
            print(f"Running robust mode filtering (cos_thresh={cos_thresh})...")
            valid_indices, filter_predictions = filter_valid_indices(images, self.model, cos_thresh=cos_thresh)
            all_indices = set(range(num_total_images))
            rejected_indices = sorted(list(all_indices - set(valid_indices)))
            print(f"Robust mode: {len(valid_indices)} valid frames, {len(rejected_indices)} rejected frames")
            print(f"Rejected frame indices: {rejected_indices}")

            # 如果没有被剔除的帧，直接使用 filter_valid_indices 返回的 predictions，不再重复推理
            if len(rejected_indices) == 0:
                print("No frames rejected, using predictions from filter pass (no second inference needed)...")
                predictions = filter_predictions

                # Convert pose encoding to extrinsic and intrinsic matrices
                print("Converting pose encoding to extrinsic and intrinsic matrices...")
                # filter_predictions 中的 tensor 已经在 CPU 上
                pose_enc = predictions["pose_enc"]
                if isinstance(pose_enc, torch.Tensor):
                    pose_enc = pose_enc.to(self.device)
                else:
                    pose_enc = torch.from_numpy(pose_enc).to(self.device)

                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                predictions["extrinsic"] = extrinsic
                predictions["intrinsic"] = intrinsic

                # Convert tensors to numpy
                for key in predictions.keys():
                    if isinstance(predictions[key], torch.Tensor):
                        # 先转换为 float32，因为 bfloat16 不支持直接转 numpy
                        tensor = predictions[key].cpu()
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.float()
                        predictions[key] = tensor.numpy().squeeze(0)
                predictions['pose_enc_list'] = None

                # Generate world points from depth map
                print("Computing world points from depth map...")
                depth_map = predictions["depth"]
                world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
                predictions["world_points_from_depth"] = world_points

                # 存储帧索引
                predictions["rejected_indices"] = np.array([], dtype=np.int64)
                predictions["valid_indices"] = np.array(list(range(num_total_images)), dtype=np.int64)

                torch.cuda.empty_cache()
                return predictions

            elif len(valid_indices) < num_total_images and len(valid_indices) > 0:
                # 有效相机数量少于总数量，需要进行第二次推理
                # 先用所有帧推理一次，获取被剔除帧的位姿
                print("First pass: getting poses for all frames...")
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=dtype):
                        all_predictions = self.model(images)

                # 保存被剔除帧的位姿
                all_extrinsic, _ = pose_encoding_to_extri_intri(all_predictions["pose_enc"], images.shape[-2:])
                all_extrinsic_np = all_extrinsic.cpu().numpy().squeeze(0)  # (S, 3, 4)
                rejected_extrinsics = {idx: all_extrinsic_np[idx] for idx in rejected_indices}

                del all_predictions
                torch.cuda.empty_cache()

                # 用有效帧重新推理
                print(f"Second pass: re-inferencing with {len(valid_indices)} valid frames...")
                valid_images = images[valid_indices]

                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=dtype):
                        predictions = self.model(valid_images)

                # 转换位姿
                valid_extrinsic, valid_intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], valid_images.shape[-2:])

                # 重建完整的 extrinsic 数组（有效帧用新位姿，被剔除帧用旧位姿）
                full_extrinsic = np.zeros((num_total_images, 3, 4), dtype=np.float32)
                full_intrinsic = np.zeros((num_total_images, 3, 3), dtype=np.float32)

                valid_extrinsic_np = valid_extrinsic.cpu().numpy().squeeze(0)
                valid_intrinsic_np = valid_intrinsic.cpu().numpy().squeeze(0)

                for new_idx, orig_idx in enumerate(valid_indices):
                    full_extrinsic[orig_idx] = valid_extrinsic_np[new_idx]
                    full_intrinsic[orig_idx] = valid_intrinsic_np[new_idx]

                for orig_idx in rejected_indices:
                    full_extrinsic[orig_idx] = rejected_extrinsics[orig_idx]
                    # 使用第一个有效帧的 intrinsic 作为被剔除帧的 intrinsic
                    full_intrinsic[orig_idx] = valid_intrinsic_np[0]

                # 转换其他预测结果
                for key in predictions.keys():
                    if isinstance(predictions[key], torch.Tensor):
                        predictions[key] = predictions[key].cpu().numpy().squeeze(0)

                # 重建完整的预测数组（被剔除帧的数据设为零或无效）
                valid_depth = predictions["depth"]  # (S_valid, H, W, 1)
                valid_depth_conf = predictions["depth_conf"]
                valid_world_points = predictions.get("world_points")
                valid_world_points_conf = predictions.get("world_points_conf")
                valid_pred_images = predictions["images"]

                H, W = valid_depth.shape[1:3]

                # 创建完整数组
                full_depth = np.zeros((num_total_images, H, W, 1), dtype=valid_depth.dtype)
                full_depth_conf = np.zeros((num_total_images, H, W), dtype=valid_depth_conf.dtype)
                full_images = np.zeros((num_total_images,) + valid_pred_images.shape[1:], dtype=valid_pred_images.dtype)

                for new_idx, orig_idx in enumerate(valid_indices):
                    full_depth[orig_idx] = valid_depth[new_idx]
                    full_depth_conf[orig_idx] = valid_depth_conf[new_idx]
                    full_images[orig_idx] = valid_pred_images[new_idx]

                predictions["depth"] = full_depth
                predictions["depth_conf"] = full_depth_conf
                predictions["images"] = full_images
                predictions["extrinsic"] = full_extrinsic
                predictions["intrinsic"] = full_intrinsic

                if valid_world_points is not None:
                    full_world_points = np.zeros((num_total_images, H, W, 3), dtype=valid_world_points.dtype)
                    full_world_points_conf = np.zeros((num_total_images, H, W), dtype=valid_world_points_conf.dtype)
                    for new_idx, orig_idx in enumerate(valid_indices):
                        full_world_points[orig_idx] = valid_world_points[new_idx]
                        full_world_points_conf[orig_idx] = valid_world_points_conf[new_idx]
                    predictions["world_points"] = full_world_points
                    predictions["world_points_conf"] = full_world_points_conf

                predictions['pose_enc_list'] = None

                # 生成 world_points_from_depth（只对有效帧）
                print("Computing world points from depth map...")
                world_points = unproject_depth_map_to_point_map(full_depth, full_extrinsic, full_intrinsic)
                predictions["world_points_from_depth"] = world_points

                # 存储被剔除帧索引和有效帧索引
                predictions["rejected_indices"] = np.array(rejected_indices, dtype=np.int64)
                predictions["valid_indices"] = np.array(valid_indices, dtype=np.int64)

                # Clean up
                torch.cuda.empty_cache()
                return predictions

        # 正常推理（非 robust 模式或只有一帧）
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)

        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        predictions['pose_enc_list'] = None # remove pose_enc_list

        # Generate world points from depth map
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        # 存储被剔除的帧索引（用于可视化）
        predictions["rejected_indices"] = np.array(rejected_indices) if rejected_indices else np.array([], dtype=np.int64)
        predictions["valid_indices"] = np.array(list(range(num_total_images)), dtype=np.int64)

        # Clean up
        torch.cuda.empty_cache()
        return predictions

    @torch.no_grad()
    def detectImages(
        self,
        images: torch.Tensor,
        image_bounds: torch.Tensor,
        robust_mode: bool=True,
        cos_thresh: float=0.95,
        crop_bound_pixel_num: int=1,
    ) -> Optional[List[RGBDCamera]]:
        '''
        image_bounds: torch.Tensor of shape (N, 4) containing [x1, y1, x2, y2] for each image,
                      indicating the pixel bounds of the original image content in the input tensor.
        '''
        if images.shape[0] == 0:
            print('[WARN][Detector::detectImages]')
            print("\t images are empty!")
            return None

        predictions = self.detect(
            images,
            robust_mode,
            cos_thresh,
        )

        if predictions is None:
            return predictions

        pred_images = predictions['images'] # N, 3, H, W
        depths = predictions['depth'] # N, H, W, 1
        depth_conf = predictions['depth_conf'] # N, H, W
        extrinsics = predictions['extrinsic'] # N, 3, 4
        intrinsics = predictions['intrinsic'] # N, 3, 3

        pred_images = (np.transpose(pred_images, (0, 2, 3, 1)) * 255.0).astype(np.uint8)[..., ::-1]
        depths = depths.reshape(*pred_images.shape[:3])

        # Crop predictions based on image_bounds to remove padding
        print('Cropping predictions based on image_bounds...')
        if isinstance(image_bounds, torch.Tensor):
            image_bounds = image_bounds.numpy()

        cropped_images_list = []
        cropped_depths_list = []
        cropped_depth_conf_list = []
        adjusted_intrinsics_list = []

        for i in range(pred_images.shape[0]):
            x1, y1, x2, y2 = image_bounds[i]
            x1 = int(x1) + crop_bound_pixel_num
            y1 = int(y1) + crop_bound_pixel_num
            x2 = int(x2) - crop_bound_pixel_num
            y2 = int(y2) - crop_bound_pixel_num

            # Crop image, depth and depth_conf
            cropped_images_list.append(pred_images[i, y1:y2, x1:x2, :])
            cropped_depths_list.append(depths[i, y1:y2, x1:x2])
            cropped_depth_conf_list.append(depth_conf[i, y1:y2, x1:x2])

            # Adjust intrinsic matrix (shift principal point)
            adjusted_intrinsic = intrinsics[i].copy()
            adjusted_intrinsic[0, 2] -= x1  # cx
            adjusted_intrinsic[1, 2] -= y1  # cy
            adjusted_intrinsics_list.append(adjusted_intrinsic)

        pred_images = cropped_images_list
        depths = cropped_depths_list
        depth_conf = cropped_depth_conf_list
        intrinsics = adjusted_intrinsics_list

        extrinsic_44_list = []
        for i in range(len(pred_images)):
            extrinsic_44 = np.zeros((4, 4), dtype=extrinsics.dtype)
            extrinsic_44[:3, :4] = extrinsics[i]
            extrinsic_44[3, :] = np.array([0, 0, 0, 1], dtype=extrinsics.dtype)
            extrinsic_44_list.append(extrinsic_44)
        extrinsics = extrinsic_44_list

        print('start create cameras...')
        camera_list = []
        for i in range(len(pred_images)):
            camera = RGBDCamera.fromVGGTPose(extrinsics[i], intrinsics[i])

            camera.loadImage(pred_images[i])

            camera.loadDepth(depths[i], depth_conf[i])

            camera_list.append(camera)

        return camera_list

    @torch.no_grad()
    def detectImageFiles(
        self,
        image_file_path_list: list,
        robust_mode: bool=True,
        cos_thresh: float=0.95,
    ) -> Optional[List[RGBDCamera]]:
        '''
        Args:
            image_file_path_list: List of image file paths
            robust_mode: Whether to use robust mode filtering
            cos_thresh: Cosine threshold for robust mode
            crop_padding: If True, crop out the padding area from predictions to get original image regions
        '''
        if len(image_file_path_list) == 0:
            print('[WARN][Detector::detectImageFiles]')
            print("\t images are empty!")
            return None

        print(f"Found {len(image_file_path_list)} images")

        images, image_bounds = load_and_preprocess_images(image_file_path_list)
        images = images.to(self.device)

        if images.shape[0] == 0:
            print('[WARN][Detector::detectImageFiles]')
            print("\t images not found!")
            return None

        camera_list = self.detectImages(
            images,
            image_bounds,
            robust_mode,
            cos_thresh,
        )

        return camera_list

    @torch.no_grad()
    def detectImageFolder(
        self,
        image_folder_path: str,
        robust_mode: bool=True,
        cos_thresh: float=0.95,
    ) -> Optional[List[RGBDCamera]]:
        '''
        Args:
            image_folder_path: Path to folder containing images
            robust_mode: Whether to use robust mode filtering
            cos_thresh: Cosine threshold for robust mode
            crop_padding: If True, crop out the padding area from predictions to get original image regions
        '''
        if not os.path.exists(image_folder_path):
            print('[ERROR][Detector::detectImageFolder]')
            print('\t image folder not exist!')
            print('\t image_folder_path:', image_folder_path)
            return None

        image_file_name_list = os.listdir(image_folder_path)
        image_file_name_list.sort()

        image_file_path_list = []
        for image_file_name in image_file_name_list:
            if image_file_name.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
                continue

            image_file_path_list.append(image_folder_path + image_file_name)

        camera_list = self.detectImageFiles(
            image_file_path_list,
            robust_mode,
            cos_thresh,
        )
        return camera_list

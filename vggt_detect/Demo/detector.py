import sys
sys.path.append('../../MATCH/camera-control')

import os

from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.colmap_renderer import COLMAPRenderer

from vggt_detect.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/VGGT/VGGT-1B/model.pt'
    vggsfm_model_file_path = home + '/chLi/Model/VGGT/vggsfm_v2_tracker.pt'
    device = 'cuda:1'
    video_file_path = home + '/chLi/Dataset/GS/haizei_1_v3.MOV'
    image_folder_path = home + '/chLi/Dataset/GS/haizei_1_v3/images/'
    save_folder_path = home + '/chLi/Dataset/GS/haizei_1_v3/vggt/'
    robust_mode = True
    cos_thresh = 0.95

    detector = Detector(
        model_file_path,
        vggsfm_model_file_path,
        device,
    )

    camera_list = detector.detectVideoFile(
        video_file_path,
        image_folder_path,
        robust_mode,
        cos_thresh,
        target_image_num=60,
    )

    assert camera_list is not None

    print('camera num:', len(camera_list))

    pcd = CameraConvertor.createDepthPcd(
        camera_list,
        conf_thresh=0.8,
    )

    save_colmap_folder_path = save_folder_path + 'colmap/'
    if not os.path.exists(save_colmap_folder_path):
        CameraConvertor.createColmapDataFolder(
            cameras=camera_list,
            pcd=pcd,
            save_data_folder_path=save_folder_path + 'colmap/',
            point_num_max=200000,
        )

    COLMAPRenderer.renderColmap(save_colmap_folder_path)
    return True

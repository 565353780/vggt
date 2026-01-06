"""
Robust VGGT: 基于注意力和特征相似度筛选有效帧的工具模块。
"""

import gc
import math
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional, Tuple

from vggt.models.vggt import VGGT


def _safe_empty_cache() -> None:
    """清理 GPU 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_num_images(images: Tensor) -> int:
    """获取输入图像的数量。"""
    if images.ndim == 5:   # (B, N, C, H, W)
        return int(images.shape[1])
    if images.ndim == 4:   # (N, C, H, W)
        return int(images.shape[0])
    if images.ndim == 3:   # (C, H, W)
        return 1
    raise ValueError(f"不支持的图像形状: {images.shape}")


def filter_valid_indices(
    images: Tensor,
    model: VGGT,
    cos_thresh: float = 0.5,
    target_layer: int = 23,
) -> Tuple[List[int], Dict]:
    """
    基于帧间余弦相似度的图连通性筛选有效帧索引。
    
    核心逻辑：
    1. 计算任意两帧之间的特征余弦相似度矩阵 (N x N)
    2. 找出相似度最高的两帧作为初始有效视角集合
    3. 依次检查其他帧，如果该帧与有效视角集合中任意一帧的相似度 >= 阈值，
       则将其加入有效视角集合
    4. 重复步骤3直到没有新的帧可以加入
    
    Args:
        images: 输入图像张量，形状为 (N, C, H, W) 或 (B, N, C, H, W)
        model: VGGT 模型实例
        cos_thresh: 余弦相似度阈值，用于判断帧是否有效
        target_layer: 用于计算特征的 transformer 层索引
        
    Returns:
        valid_indices: 有效帧的索引列表
        predictions: 模型预测结果字典
    """
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    num_images = _get_num_images(images)
    if num_images <= 1:
        return list(range(num_images)), {}
    
    if num_images == 2:
        # 只有两帧时，直接返回两帧
        return [0, 1], {}
    
    # 准备输入
    images_device = images.to(device=device, dtype=dtype)
    
    # 获取模型参数
    aggregator = model.aggregator
    patch_start_idx = aggregator.patch_start_idx
    
    # 设置 hooks 获取 aggregated_tokens
    aggregated_tokens_out: Dict[int, Tensor] = {}
    handles = []
    
    def _make_block_hook(store_dict: dict, layer_idx: int):
        """Hook for capturing block output (aggregated tokens)."""
        def _hook(_module, _inp, out):
            store_dict[layer_idx] = out.detach()
        return _hook
    
    # 添加 hook 来获取 global_blocks 的输出（aggregated tokens）
    handles.append(model.aggregator.global_blocks[target_layer].register_forward_hook(
        _make_block_hook(aggregated_tokens_out, target_layer)
    ))
    
    # 模型推理
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_device)
        else:
            predictions = model(images_device)
    
    # 移除 hooks
    for h in handles:
        h.remove()
    
    # ========== 计算帧间余弦相似度矩阵 ==========
    # 从 hook 获取 global block 的输出（即 global tokens）
    if target_layer not in aggregated_tokens_out:
        print(f"Warning: aggregated_tokens not captured for layer {target_layer}")
        return list(range(num_images)), predictions
    
    # global_blocks 的输出形状是 (B, S*P, C)
    global_block_output = aggregated_tokens_out[target_layer]
    
    # 需要重新 reshape 为 (B, S, P, C) 格式
    B = 1  # batch size
    S = num_images  # sequence length
    C = global_block_output.shape[-1]
    
    if global_block_output.ndim == 2:
        # shape: (S*P, C)
        total_tokens = global_block_output.shape[0]
        P = total_tokens // S
        global_tokens = global_block_output.view(B, S, P, C)
    elif global_block_output.ndim == 3:
        # shape: (B, S*P, C) -> (B, S, P, C)
        total_tokens = global_block_output.shape[1]
        P = total_tokens // S
        global_tokens = global_block_output.view(B, S, P, C)
    else:
        # 已经是 (B, S, P, C) 格式
        global_tokens = global_block_output
    
    # 计算帧间余弦相似度矩阵
    cos_sim_matrix: Optional[Tensor] = None
    if global_tokens.ndim == 4:
        B, N, T, C = global_tokens.shape
        if T > 0 and C > 0:
            # 提取 patch tokens 的特征
            feature = global_tokens[:, :, patch_start_idx:, :].detach().float()  # (B, N, T', C)
            B, N, T_prime, C = feature.shape
            
            # 计算每帧的平均特征向量
            frame_features = feature.mean(dim=2)  # (B, N, C)
            frame_features = frame_features.squeeze(0)  # (N, C)
            
            # 归一化
            frame_features_norm = F.normalize(frame_features, p=2, dim=-1)  # (N, C)
            
            # 计算帧间余弦相似度矩阵 (N x N)
            cos_sim_matrix = torch.mm(frame_features_norm, frame_features_norm.t())  # (N, N)
    
    # ========== 基于图连通性筛选有效帧 ==========
    valid_indices = list(range(num_images))
    
    if cos_sim_matrix is not None:
        N = cos_sim_matrix.shape[0]
        
        # 找出相似度最高的两帧（排除对角线）
        # 将对角线设为 -1，避免选择同一帧
        sim_matrix_no_diag = cos_sim_matrix.clone()
        sim_matrix_no_diag.fill_diagonal_(-1)
        
        # 找到最大相似度的位置
        max_sim_flat_idx = sim_matrix_no_diag.argmax().item()
        max_i = max_sim_flat_idx // N
        max_j = max_sim_flat_idx % N
        
        max_sim_value = cos_sim_matrix[max_i, max_j].item()
        print(f"Maximum pairwise similarity: {max_sim_value:.4f} between frames {max_i} and {max_j}")
        
        # 初始化有效视角集合：相似度最高的两帧
        valid_set = {max_i, max_j}
        remaining = set(range(N)) - valid_set
        
        # 迭代扩展有效视角集合
        changed = True
        while changed and remaining:
            changed = False
            to_add = set()
            
            for idx in remaining:
                # 检查该帧与有效集合中任意一帧的相似度是否 >= 阈值
                max_sim_to_valid = max(cos_sim_matrix[idx, v].item() for v in valid_set)
                if max_sim_to_valid >= cos_thresh:
                    to_add.add(idx)
                    changed = True
            
            valid_set.update(to_add)
            remaining -= to_add
        
        valid_indices = sorted(list(valid_set))
        
        print(f"Similarity threshold: {cos_thresh}")
        print(f"Valid frames: {valid_indices} ({len(valid_indices)}/{N})")
        print(f"Rejected frames: {sorted(list(remaining))}")
        
        # 打印相似度矩阵（调试用）
        if N <= 10:
            print("Cosine similarity matrix:")
            for i in range(N):
                row_str = " ".join([f"{cos_sim_matrix[i, j].item():.3f}" for j in range(N)])
                marker = " *" if i in valid_set else ""
                print(f"  Frame {i}: [{row_str}]{marker}")
    
    # 清理
    del aggregated_tokens_out
    _safe_empty_cache()
    
    # 转换 predictions 到 CPU
    predictions_cpu = {}
    for k, v in predictions.items():
        if torch.is_tensor(v):
            predictions_cpu[k] = v.detach().cpu()
        else:
            predictions_cpu[k] = v
    
    return valid_indices, predictions_cpu

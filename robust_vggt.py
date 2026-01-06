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
    attn_weight: float = 0.5,
    cos_weight: float = 0.5,
    reject_thresh: float = 0.4,
    target_layer: int = 23,
) -> Tuple[List[int], Dict]:
    """
    基于注意力和余弦相似度筛选有效帧索引。
    
    核心逻辑：
    1. 计算每帧与参考帧（第一帧）的特征余弦相似度
    2. 计算每帧在参考帧注意力图中的平均权重
    3. 综合两个指标，筛选出有效帧
    
    Args:
        images: 输入图像张量，形状为 (N, C, H, W) 或 (B, N, C, H, W)
        model: VGGT 模型实例
        attn_weight: 注意力权重系数
        cos_weight: 余弦相似度权重系数
        reject_thresh: 拒绝阈值，低于此值的帧将被过滤
        target_layer: 用于计算的 transformer 层索引
        
    Returns:
        valid_indices: 有效帧的索引列表
        predictions: 模型预测结果字典
    """
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    num_images = _get_num_images(images)
    if num_images <= 1:
        return list(range(num_images)), {}
    
    # 准备输入
    images_device = images.to(device=device, dtype=dtype)
    
    # 获取模型参数
    aggregator = model.aggregator
    patch_size = aggregator.patch_size
    patch_start_idx = aggregator.patch_start_idx
    H, W = images.shape[-2:]
    h_patches = H // patch_size
    w_patches = W // patch_size
    num_patch_tokens = h_patches * w_patches
    tokens_per_image = patch_start_idx + num_patch_tokens
    
    # 设置 hooks 获取注意力权重和 aggregated_tokens
    q_out: Dict[int, Tensor] = {}
    k_out: Dict[int, Tensor] = {}
    aggregated_tokens_out: Dict[int, Tensor] = {}
    handles = []
    
    def _make_hook(store_dict: dict, layer_idx: int):
        def _hook(_module, _inp, out):
            store_dict[layer_idx] = out.detach()
        return _hook
    
    def _make_block_hook(store_dict: dict, layer_idx: int):
        """Hook for capturing block output (aggregated tokens)."""
        def _hook(_module, _inp, out):
            store_dict[layer_idx] = out.detach()
        return _hook
    
    blk = model.aggregator.global_blocks[target_layer].attn
    handles.append(blk.q_norm.register_forward_hook(_make_hook(q_out, target_layer)))
    handles.append(blk.k_norm.register_forward_hook(_make_hook(k_out, target_layer)))
    
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
    
    # ========== 计算余弦相似度 ==========
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
    
    cos_sim_mean: Optional[Tensor] = None
    if global_tokens.ndim == 4:
        B, N, T, C = global_tokens.shape
        if T > 0 and C > 0:
            feature = global_tokens[:, :, patch_start_idx:, :].detach().float()
            B, N, T, C = feature.shape
            layer_feat = feature.reshape(B * N, T, C)
            
            ref_feat = layer_feat[0:1]  # 参考帧特征
            ref_feat_norm = F.normalize(ref_feat, p=2, dim=-1)
            layer_feat_norm = F.normalize(layer_feat, p=2, dim=-1)
            
            cos_sim = torch.einsum("bic,bjc->bij", layer_feat_norm, ref_feat_norm)
            cos_sim_mean = cos_sim.mean(-1).mean(-1)  # (N,)
    
    # ========== 计算注意力均值 ==========
    attn_mean_vals: List[float] = []
    
    if target_layer in q_out and target_layer in k_out:
        Q = q_out[target_layer]
        K = k_out[target_layer]
        
        T_k = int(K.shape[-2])
        num_images_in_seq = T_k // tokens_per_image
        
        if num_images_in_seq > 0:
            # 第一帧的 query
            q_first = Q[:, :, patch_start_idx:patch_start_idx + num_patch_tokens, :]
            # 所有帧的 key
            T_k_slice = min(num_images, num_images_in_seq) * tokens_per_image
            K_slice = K[:, :, :T_k_slice, :]
            
            scale = 1.0 / math.sqrt(float(q_first.shape[-1]))
            logits = torch.einsum("bhqd,bhtd->bhqt", q_first, K_slice) * scale
            probs = torch.softmax(logits, dim=-1)
            attn_first = probs.mean(dim=1).mean(dim=1)[0]  # (T_k_slice,)
            
            # 计算每帧的注意力均值
            for img_idx in range(num_images):
                start = img_idx * tokens_per_image + patch_start_idx
                end = start + num_patch_tokens
                if start >= attn_first.shape[-1]:
                    break
                end = min(end, attn_first.shape[-1])
                
                patch_attn = attn_first[start:end]
                if patch_attn.numel() == num_patch_tokens:
                    attn_mean_vals.append(float(patch_attn.mean().item()))
    
    # ========== 综合评分并筛选 ==========
    valid_indices = list(range(num_images))
    
    if attn_mean_vals and cos_sim_mean is not None and len(attn_mean_vals) == num_images:
        attn_tensor = torch.tensor(attn_mean_vals, device=cos_sim_mean.device, dtype=cos_sim_mean.dtype)
        
        # 归一化
        cos_norm = (cos_sim_mean - cos_sim_mean.min()) / (cos_sim_mean.max() - cos_sim_mean.min() + 1e-6)
        attn_norm = (attn_tensor - attn_tensor.min()) / (attn_tensor.max() - attn_tensor.min() + 1e-6)
        
        # 综合评分
        combined_score = attn_weight * attn_norm + cos_weight * cos_norm
        
        # 筛选（第一帧永不拒绝）
        valid_indices = [0]
        for idx in range(1, num_images):
            if combined_score[idx] >= reject_thresh:
                valid_indices.append(idx)
    
    # 清理
    del q_out, k_out, aggregated_tokens_out
    _safe_empty_cache()
    
    # 转换 predictions 到 CPU
    predictions_cpu = {}
    for k, v in predictions.items():
        if torch.is_tensor(v):
            predictions_cpu[k] = v.detach().cpu()
        else:
            predictions_cpu[k] = v
    
    return valid_indices, predictions_cpu

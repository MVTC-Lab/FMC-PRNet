import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from typing import Tuple, Optional  # 军师修复：补充缺失的类型导入


# =========================================
# 0. 军师修复：补充原作者缺失的辅助函数
# =========================================
def _grid2seq(x: Tensor, region_size: Tuple[int, int], num_heads: int):
    """将特征图划分为区域序列"""
    bs, c, h, w = x.size()
    head_dim = c // num_heads
    region_h, region_w = region_size
    q_region_h, q_region_w = h // region_h, w // region_w
    x = x.view(bs, num_heads, head_dim, q_region_h, region_h, q_region_w, region_w)
    x = x.permute(0, 1, 3, 5, 4, 6, 2).contiguous()
    x = x.view(bs, num_heads, q_region_h * q_region_w, region_h * region_w, head_dim)
    return x, q_region_h, q_region_w


def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int, int]):
    """将区域序列还原为特征图"""
    bs, nhead, nregion, reg_size_sq, head_dim = x.size()
    rh, rw = region_size
    x = x.view(bs, nhead, region_h, region_w, rh, rw, head_dim)
    x = x.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
    x = x.view(bs, nhead * head_dim, region_h * rh, region_w * rw)
    return x


# =========================================
# 1. 核心运算函数
# =========================================
def regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: LongTensor, region_size: Tuple[int, int],
        kv_region_size: Optional[Tuple[int, int]] = None,
        auto_pad=True) -> Tensor:
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))

            # to sequence format
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)

    # gather key and values
    bs, nhead, kv_nregion, kv_region_size_num, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size_num, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size_num, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size_num, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)

    # token-to-token attention
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ value_g.flatten(-3, -2)

    # to BCHW format
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn


# =========================================
# 2. 原作者封装的 NCHW 类
# =========================================
class nchwBRA(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=4, side_dwconv=3, auto_pad=False,
                 attn_backend='torch'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        self.topk = topk
        self.n_win = n_win

        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        N, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)

        qkv = self.qkv_linear.forward(x)
        q, k, v = qkv.chunk(3, dim=1)

        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)
        k_r: Tensor = k_r.flatten(2, 3)
        a_r = q_r @ k_r
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)

        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size)

        output = output + self.lepe(v)
        output = self.output_linear(output)

        if ret_attn_mask:
            return output, attn_mat

        return output


# =========================================
# 3. ======= 军师特制：YOLO 无缝接入外壳 =======
# =========================================
class BRA_YOLO(nn.Module):
    """
    专门为 Ultralytics YOLO 架构设计的 Bi-Level Routing Attention 适配器
    """

    def __init__(self, c1, c2, num_heads=4, n_win=7):
        super().__init__()
        # 强制通道数对齐。如果您的 YOLO 深层通道是 512，num_heads 设为 4 或 8 都是安全的 (512能被整除)
        self.bra = nchwBRA(dim=c1, num_heads=num_heads, n_win=n_win)

    def forward(self, x):
        return self.bra(x)
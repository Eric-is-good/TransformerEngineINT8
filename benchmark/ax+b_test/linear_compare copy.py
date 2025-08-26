#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
INT8 数据 + INT8 位移缩放 的 QATLinear 原型，与 FP16 Linear 的对照实验。
- 分块 block-wise 动态 2^k 缩放（k 为 int8），量化为 int8，再反量化；
- 使用 STE（straight-through estimator）让量化-反量化的梯度近似为恒等；
- 对同一批数据做一次前向+反向+参数更新；
- 对比：输出差异、梯度差异、参数更新差异、耗时；

说明：为保证可运行性，这里用“反量化后”的浮点 matmul 来近似 INT8 前向，
这样便于在任意环境（CPU/GPU）重现实验，不依赖特定 INT8 MatMul 内核。
如果你在老显卡上有 INT8 TensorCore/DP4A，可把 matmul 替换为 int8 算子即可。
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# 量化与反量化：INT8 + 2^k（k 为 int8）
# =============================

def _compute_pow2_exponent(max_abs: torch.Tensor) -> torch.Tensor:
    """给定每个 block 的 max_abs，返回整型指数 k（int8 范围内），使得 |x|/2^k ≤ 127。
    k = ceil(log2(max_abs / 127)), 允许 k 为负以提升小值精度。
    """
    eps = 1e-8
    ratio = (max_abs + eps) / 127.0
    k = torch.ceil(torch.log2(ratio))
    return k.to(torch.int32)  # 后续再 clamp 到 int8 范围


def quantize_pow2_int8(x: torch.Tensor, block_size: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """按维度 dim 以 block_size 分块量化到 int8，并返回 (q_int8, k_int8)。

    x: 任意形状，按 dim 切 blocks。
    量化公式：q = clamp(round(x / 2^k), -128, 127)
    k 按每个 block 的最大绝对值自适应。
    """
    # 置换 dim 为最后一维便于分块
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(*perm).contiguous()
    orig_shape = x_perm.shape
    L = orig_shape[-1]
    pad = (block_size - (L % block_size)) % block_size
    if pad:
        x_perm = F.pad(x_perm, (0, pad))
    new_L = x_perm.shape[-1]
    num_blocks = new_L // block_size

    x_blocks = x_perm.view(-1, num_blocks, block_size)
    max_abs = x_blocks.abs().amax(dim=-1)  # [*, num_blocks]
    k = _compute_pow2_exponent(max_abs)

    # clamp 到 int8 范围
    k = torch.clamp(k, min=-128, max=127)

    # 反算 scale = 2^k
    s = torch.pow(2.0, k.to(torch.float32))  # [*, num_blocks]

    # 量化：x / s -> round -> clamp
    # 对齐维度
    s_expanded = s.unsqueeze(-1)  # [*, num_blocks, 1]
    q = torch.round(x_blocks / s_expanded)
    q = torch.clamp(q, -128, 127).to(torch.int8)

    # 反量化
    x_hat = (q.to(torch.float32) * s_expanded).view(*x_perm.shape)
    if pad:
        x_hat = x_hat[..., :L]
        q = q.view(*x_perm.shape)[..., :L]
        k = k.view(*x_perm.shape[:-1], num_blocks)
    else:
        q = q.view(*x_perm.shape)
        k = k.view(*x_perm.shape[:-1], num_blocks)

    # 还原维度顺序
    inv_perm = list(range(x.ndim))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    x_hat = x_hat.permute(*inv_perm).contiguous()
    q = q.permute(*inv_perm).contiguous()

    # k 的维度：把 dim 的长度换成 num_blocks（向上取整）
    k_shape = list(x.shape)
    k_shape[dim] = math.ceil(L / block_size)
    k = k.view(*k_shape).to(torch.int8)

    return x_hat, k


def ste_quant(x: torch.Tensor, block_size: int, dim: int):
    """量化-反量化并用 STE 让梯度近似恒等。返回 x_qdq 和 k。"""
    with torch.no_grad():
        x_hat, k = quantize_pow2_int8(x, block_size, dim)
    # STE: 前向用 x_hat，反向用 x
    x_qdq = x + (x_hat - x).detach()
    return x_qdq, k


# =============================
# QATLinear 实现
# =============================

@dataclass
class QATConfig:
    in_features: int
    out_features: int
    bias: bool = True
    block_size: int = 32


class QATLinear(nn.Module):
    def __init__(self, cfg: QATConfig):
        super().__init__()
        self.in_features = cfg.in_features
        self.out_features = cfg.out_features
        self.block_size = cfg.block_size
        self.weight = nn.Parameter(torch.empty(cfg.out_features, cfg.in_features))
        if cfg.bias:
            self.bias = nn.Parameter(torch.zeros(cfg.out_features))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # for inspection
        self.last_k_x = None
        self.last_k_w = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 激活在最后一维（features）按 block 量化
        x_qdq, k_x = ste_quant(x, self.block_size, dim=-1)
        # 权重在 in_features 维按 block 量化
        w_qdq, k_w = ste_quant(self.weight, self.block_size, dim=-1)
        self.last_k_x = k_x
        self.last_k_w = k_w
        out = x_qdq.matmul(w_qdq.t())
        if self.bias is not None:
            out = out + self.bias
        return out


# =============================
# 实验：QATLinear vs FP16 Linear
# =============================

def run_experiment(device: torch.device = None):
    torch.manual_seed(42)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    in_f, out_f = 1024, 4096
    bs = 64
    block = 32

    # 数据与“真值”标签（用一个隐藏的 FP32 线性层生成 target，便于两边一致对齐）
    true_linear = nn.Linear(in_f, out_f).to(device)
    x = torch.randn(bs, in_f, device=device)
    with torch.no_grad():
        y = true_linear(x)

    # QATLinear（参数用 FP32 保持稳定）
    qat = QATLinear(QATConfig(in_f, out_f, True, block)).to(device)
    # FP16 baseline，与 QAT 权重初始化一致
    fp16 = nn.Linear(in_f, out_f, bias=True).to(device)
    with torch.no_grad():
        fp16.weight.copy_(qat.weight)
        fp16.bias.copy_(qat.bias)
    fp16 = fp16.half()

    # 优化器
    lr = 1e-3
    opt_qat = torch.optim.SGD(qat.parameters(), lr=lr)
    opt_fp16 = torch.optim.SGD(fp16.parameters(), lr=lr)

    # 损失
    loss_fn = nn.MSELoss()

    # --------------- QAT step ---------------
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()

    opt_qat.zero_grad(set_to_none=True)
    out_qat = qat(x)
    loss_qat = loss_fn(out_qat, y)
    loss_qat.backward()
    grad_w_qat = qat.weight.grad.detach().clone()
    grad_b_qat = qat.bias.grad.detach().clone() if qat.bias is not None else None
    opt_qat.step()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    qat_time = (time.time() - t0) * 1000

    # --------------- FP16 step ---------------
    x16 = x.half()
    y16 = y.half()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()

    opt_fp16.zero_grad(set_to_none=True)
    out_fp16 = fp16(x16)
    loss_fp16 = loss_fn(out_fp16, y16)
    loss_fp16.backward()
    grad_w_fp16 = fp16.weight.grad.detach().float().clone()
    grad_b_fp16 = fp16.bias.grad.detach().float().clone() if fp16.bias is not None else None
    opt_fp16.step()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    fp16_time = (time.time() - t1) * 1000

    # --------------- 差异度量 ---------------
    with torch.no_grad():
        # 输出差异（用 FP32 计算 L2 和相对误差）
        out_fp16_f = out_fp16.float()
        out_diff_l2 = (out_qat - out_fp16_f).pow(2).mean().sqrt().item()
        rel = (out_fp16_f.abs().mean() + 1e-6).item()
        out_diff_rel = out_diff_l2 / rel

        # 梯度差异
        grad_w_diff = (grad_w_qat - grad_w_fp16).pow(2).mean().sqrt().item()
        grad_b_diff = (
            (grad_b_qat - grad_b_fp16).pow(2).mean().sqrt().item()
            if grad_b_qat is not None else float("nan")
        )

        # 参数更新差异
        # 取 step 后的权重/偏置差异（两边初值相同）
        w_upd_diff = (qat.weight.detach() - fp16.weight.detach().float()).pow(2).mean().sqrt().item()
        b_upd_diff = (qat.bias.detach() - fp16.bias.detach().float()).pow(2).mean().sqrt().item()

    # 打印结果
    print("==== Results ====")
    print(f"QAT loss:   {loss_qat.item():.6f}  | time: {qat_time:.2f} ms")
    print(f"FP16 loss:  {loss_fp16.float().item():.6f}  | time: {fp16_time:.2f} ms")
    print(f"Output L2 diff: {out_diff_l2:.6e}  | rel: {out_diff_rel:.6e}")
    print(f"Grad W L2 diff: {grad_w_diff:.6e}")
    print(f"Grad b L2 diff: {grad_b_diff:.6e}")
    print(f"Update W L2 diff after 1 step: {w_upd_diff:.6e}")
    print(f"Update b L2 diff after 1 step: {b_upd_diff:.6e}")

    return {
        "qat_time_ms": qat_time,
        "fp16_time_ms": fp16_time,
        "out_diff_l2": out_diff_l2,
        "out_diff_rel": out_diff_rel,
        "grad_w_diff": grad_w_diff,
        "grad_b_diff": grad_b_diff,
        "w_upd_diff": w_upd_diff,
        "b_upd_diff": b_upd_diff,
    }


if __name__ == "__main__":
    run_experiment()

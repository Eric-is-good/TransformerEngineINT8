# 按 n 行划分

import torch
import time
import math
from typing import Tuple

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
device = torch.device("cuda")

# -------------------------------------------------
# 超参
# -------------------------------------------------
M, K, N = 4096, 4096, 4096
TILE_M = 1          # 行块大小
TILE_N = 1          # 列块大小
TILE_K = K            # K 维不切块，可扩展

assert M % TILE_M == 0 and N % TILE_N == 0

# -------------------------------------------------
# 构造 float16 原始张量
# -------------------------------------------------
x = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
w = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
b = torch.randn(N, device=device, dtype=torch.float16)

# -------------------------------------------------
# 分块量化
# -------------------------------------------------
def block_int8_exp(tensor: torch.Tensor,
                   dim: int,
                   tile: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tensor: 要量化的矩阵
    dim   : 0 表示按行切块（x 用），1 表示按列切块（w 用）
    tile  : 每块大小
    return: (int8_data, int8_exp)  exp shape = ceil(size/tile)
    """
    if dim == 1:
        tensor = tensor.t()   # 按列切块先转置，逻辑变成按行
    n_blocks = (tensor.size(0) + tile - 1) // tile
    int8_data = torch.empty_like(tensor, dtype=torch.int8)
    exp = torch.empty(n_blocks, dtype=torch.int8, device=tensor.device)

    for i in range(n_blocks):
        sl = slice(i*tile, (i+1)*tile)
        blk = tensor[sl]
        # 选最大绝对值
        max_abs = blk.abs().max()
        # 指数：让 127 * 2^e ≈ max_abs
        e = int(torch.round(torch.log2(max_abs / 127.0)).item())
        e = max(-128, min(127, e))          # 保证 int8
        scale = 2 ** e
        int8_blk = (blk / scale).round().clamp(-128, 127).to(torch.int8)
        int8_data[sl] = int8_blk
        exp[i] = e

    if dim == 1:
        int8_data = int8_data.t()
    return int8_data, exp

x_int8, x_exp = block_int8_exp(x, dim=0, tile=TILE_M)   # shape (M,K) 和 (M//TILE_M,)
w_int8, w_exp = block_int8_exp(w.t(), dim=0, tile=TILE_N)  # w 转置后按行切块
w_int8 = w_int8.t()                                    # 再转回来，得到 (K,N)

# -------------------------------------------------
# 分块 INT8 GEMM + 位移还原
# -------------------------------------------------
@torch.inference_mode()
def int8_gemm_block_exp(x_int8, x_exp, w_int8, w_exp, b):
    """
    x_int8: (M, K)  int8
    x_exp : (M//TILE_M,)  int8
    w_int8: (K, N)  int8
    w_exp : (N//TILE_N,)  int8
    b     : (N,)   fp16
    return: (M, N) fp16
    """
    y_int32 = torch._int_mm(x_int8, w_int8)   # (M, N)  int32

    # 构造每个输出 tile 的指数
    x_exp_t = x_exp.view(-1, 1).expand(-1, N // TILE_N)   # (M//TILE_M, N//TILE_N)
    w_exp_t = w_exp.view(1, -1).expand(M // TILE_M, -1)   # (M//TILE_M, N//TILE_N)
    total_exp = x_exp_t + w_exp_t                       # 指数相加

    # 现在把 total_exp 扩展到 (M, N) 精度
    y_expanded = total_exp.repeat_interleave(TILE_M, dim=0).repeat_interleave(TILE_N, dim=1)

    # 位移还原（int32 -> float32 -> 左移）
    y_fp32 = y_int32.to(torch.float32)
    # 用 ldexp 做 2^exp
    y_scaled = torch.ldexp(y_fp32, y_expanded.to(torch.float32))
    return (y_scaled + b.to(torch.float32)).to(torch.float16)

# -------------------------------------------------
# 误差 & 性能对比
# -------------------------------------------------
y_fp16 = torch.matmul(x, w) + b
y_block_int8 = int8_gemm_block_exp(x_int8, x_exp, w_int8, w_exp, b)

abs_err = (y_fp16 - y_block_int8).abs().mean().item()
rel_err = abs_err / y_fp16.abs().mean().item()

def bench(fn, *args, loops=100):
    torch.cuda.synchronize()
    for _ in range(10): _ = fn(*args)
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(loops): _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / loops

t_fp16 = bench(lambda: torch.matmul(x, w) + b)
t_int8 = bench(int8_gemm_block_exp, x_int8, x_exp, w_int8, w_exp, b)

print("------------------------------------------------")
print(f"Shape: {M}×{K} @ {K}×{N}")
print(f"FP16: {t_fp16*1e3:.3f} ms   INT8-block: {t_int8*1e3:.3f} ms")
print(f"INT8/FP16 速度比: {t_fp16/t_int8:.2f}x")
print(f"绝对误差: {abs_err:.5f}   相对误差: {rel_err:.3%}")
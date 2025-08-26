# 按单行划分，再行内划分 micro_block

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
# 新增：沿着 K 维进行分块的块大小，粒度更细
BLOCK_SIZE_K = 4096

assert K % BLOCK_SIZE_K == 0

# -------------------------------------------------
# 构造 float16 原始张量
# -------------------------------------------------
x = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
w = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
b = torch.randn(N, device=device, dtype=torch.float16)

# -------------------------------------------------
# 新的细粒度分块量化函数
# -------------------------------------------------
def quantize_fine_grained(tensor: torch.Tensor,
                          block_size: int,
                          dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    沿着 K 维度对张量进行细粒度分块量化

    tensor: 要量化的矩阵
    block_size: K 维度上的块大小
    dim: K 维度所在的维度 (对于 x(M,K) 是 1, 对于 w(K,N) 是 0)
    return: (int8_data, int8_exp)
            x_exp shape: (M, K // block_size)
            w_exp shape: (K // block_size, N)
    """
    assert tensor.dim() == 2
    assert tensor.shape[dim] % block_size == 0

    if dim == 1:  # 处理 x (M, K)
        M, K = tensor.shape
        # 重塑为 (M, K/block_size, block_size) 以便按块计算
        reshaped_tensor = tensor.view(M, K // block_size, block_size)
        # 沿着最后一个维度（块内）找最大绝对值
        max_abs = reshaped_tensor.abs().max(dim=2, keepdim=True).values
    elif dim == 0:  # 处理 w (K, N)
        K, N = tensor.shape
        # 重塑为 (K/block_size, block_size, N)
        reshaped_tensor = tensor.view(K // block_size, block_size, N)
        # 沿着中间维度（块内）找最大绝对值
        max_abs = reshaped_tensor.abs().max(dim=1, keepdim=True).values
    else:
        raise ValueError("dim 必须是 0 或 1")

    # 避免 log(0)
    max_abs[max_abs == 0] = 1.0
    
    # 指数：让 127 * 2^e ≈ max_abs
    exponents = torch.round(torch.log2(max_abs / 127.0))
    exponents = torch.clamp(exponents, -128, 127).to(torch.int8)
    
    # 根据指数计算缩放因子
    scales = torch.pow(2.0, exponents.float())

    # 量化
    quantized_tensor = (tensor.view_as(reshaped_tensor) / scales).round().clamp(-128, 127).to(torch.int8)

    # 恢复原始形状
    int8_data = quantized_tensor.view_as(tensor)
    
    # 移除多余的维度，得到最终的 exp 张量
    if dim == 1:
        exp_shape = (M, K // block_size)
    else:
        exp_shape = (K // block_size, N)
    
    return int8_data, exponents.view(exp_shape)

# 使用新的函数进行量化
x_int8, x_exp = quantize_fine_grained(x, block_size=BLOCK_SIZE_K, dim=1)
w_int8, w_exp = quantize_fine_grained(w, block_size=BLOCK_SIZE_K, dim=0)

# -------------------------------------------------
# 新的细粒度 INT8 GEMM + 反量化
# -------------------------------------------------
@torch.inference_mode()
def int8_gemm_fine_grained(x_int8, x_exp, w_int8, w_exp, b, block_size_k):
    """
    x_int8: (M, K) int8
    x_exp : (M, K//block_size_k) int8
    w_int8: (K, N) int8
    w_exp : (K//block_size_k, N) int8
    b     : (N,) fp16
    return: (M, N) fp16
    """
    M, K = x_int8.shape
    _, N = w_int8.shape
    
    assert K % block_size_k == 0
    n_k_blocks = K // block_size_k

    # 初始化fp32输出张量用于累加
    y_fp32 = torch.zeros(M, N, device=x_int8.device, dtype=torch.float32)

    # 沿着K维度分块计算
    for i in range(n_k_blocks):
        k_start = i * block_size_k
        k_end = (i + 1) * block_size_k
        
        # 切分当前块的 int8 数据
        x_sub_int8 = x_int8[:, k_start:k_end]
        w_sub_int8 = w_int8[k_start:k_end, :]
        
        # 对当前块进行 int8 矩阵乘法
        y_sub_int32 = torch._int_mm(x_sub_int8, w_sub_int8)
        
        # 提取当前块对应的指数
        x_sub_exp = x_exp[:, i]      # Shape: (M,)
        w_sub_exp = w_exp[i, :]      # Shape: (N,)
        
        # 组合指数，通过广播得到 (M,N) 的总指数
        total_sub_exp = x_sub_exp.view(-1, 1) + w_sub_exp.view(1, -1)
        
        # 反量化当前块的结果
        y_sub_fp32 = torch.ldexp(y_sub_int32.to(torch.float32), total_sub_exp.to(torch.float32))
        
        # 累加到最终结果
        y_fp32 += y_sub_fp32
        
    # 添加偏置并转为fp16
    return (y_fp32 + b.to(torch.float32)).to(torch.float16)

# -------------------------------------------------
# 误差 & 性能对比
# -------------------------------------------------
y_fp16 = torch.matmul(x, w) + b
y_block_int8 = int8_gemm_fine_grained(x_int8, x_exp, w_int8, w_exp, b, BLOCK_SIZE_K)

abs_err = (y_fp16 - y_block_int8).abs().mean().item()
rel_err = abs_err / y_fp16.abs().mean().item()

def bench(fn, *args, loops=100):
    torch.cuda.synchronize()
    # Warmup
    for _ in range(10): _ = fn(*args)
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(loops): _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / loops

t_fp16 = bench(lambda: torch.matmul(x, w) + b)
t_int8 = bench(int8_gemm_fine_grained, x_int8, x_exp, w_int8, w_exp, b, BLOCK_SIZE_K)

print("------------------------------------------------")
print(f"Shape: {M}×{K} @ {K}×{N}")
print(f"Block Size K: {BLOCK_SIZE_K}")
print("------------------------------------------------")
print(f"FP16: {t_fp16*1e3:.3f} ms")
print(f"INT8-fine-grained: {t_int8*1e3:.3f} ms")
print(f"INT8/FP16 速度比: {t_fp16/t_int8:.2f}x")
print(f"绝对误差: {abs_err:.5f}    相对误差: {rel_err:.3%}")
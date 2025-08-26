# 按单行划分，再行内划分 micro_block，多次运行测试


import torch
import time
import math
from typing import Tuple

torch.backends.cuda.matmul.allow_tf32 = False
# torch.manual_seed(0) # 移除全局随机种子，确保每次循环数据都不同
device = torch.device("cuda")

# -------------------------------------------------
# 超参
# -------------------------------------------------
M, K, N = 4096, 4096, 4096
BLOCK_SIZE_K = 4096
ERROR_AVG_RUNS = 20  # 新增：计算误差的平均运行次数

assert K % BLOCK_SIZE_K == 0

# -------------------------------------------------
# (函数定义部分与之前相同，这里为保持完整性而保留)
# -------------------------------------------------
def quantize_fine_grained(tensor: torch.Tensor,
                          block_size: int,
                          dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert tensor.dim() == 2
    assert tensor.shape[dim] % block_size == 0
    if dim == 1:
        M, K = tensor.shape
        reshaped_tensor = tensor.view(M, K // block_size, block_size)
        max_abs = reshaped_tensor.abs().max(dim=2, keepdim=True).values
    elif dim == 0:
        K, N = tensor.shape
        reshaped_tensor = tensor.view(K // block_size, block_size, N)
        max_abs = reshaped_tensor.abs().max(dim=1, keepdim=True).values
    else:
        raise ValueError("dim 必须是 0 或 1")
    max_abs[max_abs == 0] = 1.0
    exponents = torch.round(torch.log2(max_abs / 127.0))
    exponents = torch.clamp(exponents, -128, 127).to(torch.int8)
    scales = torch.pow(2.0, exponents.float())
    quantized_tensor = (tensor.view_as(reshaped_tensor) / scales).round().clamp(-128, 127).to(torch.int8)
    int8_data = quantized_tensor.view_as(tensor)
    if dim == 1:
        exp_shape = (M, K // block_size)
    else:
        exp_shape = (K // block_size, N)
    return int8_data, exponents.view(exp_shape)

@torch.inference_mode()
def int8_gemm_fine_grained(x_int8, x_exp, w_int8, w_exp, b, block_size_k):
    M, K = x_int8.shape
    _, N = w_int8.shape
    assert K % block_size_k == 0
    n_k_blocks = K // block_size_k
    y_fp32 = torch.zeros(M, N, device=x_int8.device, dtype=torch.float32)
    for i in range(n_k_blocks):
        k_start = i * block_size_k
        k_end = (i + 1) * block_size_k
        x_sub_int8 = x_int8[:, k_start:k_end]
        w_sub_int8 = w_int8[k_start:k_end, :]
        y_sub_int32 = torch._int_mm(x_sub_int8, w_sub_int8)
        x_sub_exp = x_exp[:, i]
        w_sub_exp = w_exp[i, :]
        total_sub_exp = x_sub_exp.view(-1, 1) + w_sub_exp.view(1, -1)
        y_sub_fp32 = torch.ldexp(y_sub_int32.to(torch.float32), total_sub_exp.to(torch.float32))
        y_fp32 += y_sub_fp32
    return (y_fp32 + b.to(torch.float32)).to(torch.float16)

# -------------------------------------------------
# 误差 & 性能对比
# -------------------------------------------------
total_abs_err = 0.0
total_rel_err = 0.0

print(f"开始计算 {ERROR_AVG_RUNS} 次运行的平均误差...")

for i in range(ERROR_AVG_RUNS):
    # 在循环内生成全新的随机张量
    x = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
    w = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    b = torch.randn(N, device=device, dtype=torch.float16)

    # 量化
    x_int8, x_exp = quantize_fine_grained(x, block_size=BLOCK_SIZE_K, dim=1)
    w_int8, w_exp = quantize_fine_grained(w, block_size=BLOCK_SIZE_K, dim=0)

    # 计算
    y_fp16 = torch.matmul(x, w) + b
    y_block_int8 = int8_gemm_fine_grained(x_int8, x_exp, w_int8, w_exp, b, BLOCK_SIZE_K)

    # 计算当次误差并累加
    run_abs_err = (y_fp16 - y_block_int8).abs().mean().item()
    run_rel_err = run_abs_err / y_fp16.abs().mean().item()
    total_abs_err += run_abs_err
    total_rel_err += run_rel_err
    print(f"  Run {i+1}/{ERROR_AVG_RUNS}: 相对误差 = {run_rel_err:.3%}")

# 计算平均误差
avg_abs_err = total_abs_err / ERROR_AVG_RUNS
avg_rel_err = total_rel_err / ERROR_AVG_RUNS

# 性能测试 (使用最后一次循环的数据)
def bench(fn, *args, loops=100):
    torch.cuda.synchronize()
    for _ in range(10): _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(loops): _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / loops

t_fp16 = bench(lambda: torch.matmul(x, w) + b)
t_int8 = bench(int8_gemm_fine_grained, x_int8, x_exp, w_int8, w_exp, b, BLOCK_SIZE_K)


print("\n------------------------------------------------")
print(f"Shape: {M}×{K} @ {K}×{N}")
print(f"Block Size K: {BLOCK_SIZE_K}")
print(f"误差平均次数: {ERROR_AVG_RUNS}")
print("------------------------------------------------")
print(f"FP16 性能: {t_fp16*1e3:.3f} ms")
print(f"INT8-fine-grained 性能: {t_int8*1e3:.3f} ms")
print(f"INT8/FP16 速度比: {t_fp16/t_int8:.2f}x")
print(f"平均绝对误差: {avg_abs_err:.5f}    平均相对误差: {avg_rel_err:.3%}")
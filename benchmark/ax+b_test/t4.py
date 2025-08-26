# 按单行划分，再行内划分 micro_block，多次运行测试
# 测试更加复杂的网络 Transformer中FFN output = LayerNorm(x + Linear2(ReLU(Linear1(x))))

import torch
import torch.nn as nn
import time
import math
from typing import Tuple

torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device("cuda")

# -------------------------------------------------
# 超参
# -------------------------------------------------
M, D_MODEL = 4096, 4096   # 批大小 和 模型维度
D_FFN = D_MODEL * 4      # FFN中间层维度
BLOCK_SIZE_K = 128
ERROR_AVG_RUNS = 10      # 误差平均次数 (适当减少以加快运行)

# -------------------------------------------------
# (工具函数部分与之前相同，直接复用)
# -------------------------------------------------
def quantize_fine_grained(tensor: torch.Tensor,
                          block_size: int,
                          dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert tensor.dim() == 2
    assert tensor.shape[dim] % block_size == 0
    if dim == 1:
        M_dim, K_dim = tensor.shape
        reshaped_tensor = tensor.view(M_dim, K_dim // block_size, block_size)
        max_abs = reshaped_tensor.abs().max(dim=2, keepdim=True).values
    elif dim == 0:
        K_dim, N_dim = tensor.shape
        reshaped_tensor = tensor.view(K_dim // block_size, block_size, N_dim)
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
        exp_shape = (tensor.shape[0], tensor.shape[1] // block_size)
    else:
        exp_shape = (tensor.shape[0] // block_size, tensor.shape[1])
    return int8_data, exponents.view(exp_shape)

@torch.inference_mode()
def int8_gemm_fine_grained(x_int8, x_exp, w_int8, w_exp, b, block_size_k):
    M, K = x_int8.shape
    _, N = w_int8.shape
    assert K % block_size_k == 0
    n_k_blocks = K // block_size_k
    y_fp32 = torch.zeros(M, N, device=x_int8.device, dtype=torch.float32)
    for i in range(n_k_blocks):
        k_start, k_end = i * block_size_k, (i + 1) * block_size_k
        y_sub_int32 = torch._int_mm(x_int8[:, k_start:k_end], w_int8[k_start:k_end, :])
        total_sub_exp = x_exp[:, i].view(-1, 1) + w_exp[i, :].view(1, -1)
        y_fp32 += torch.ldexp(y_sub_int32.to(torch.float32), total_sub_exp.to(torch.float32))
    return (y_fp32 + b.to(torch.float32)).to(torch.float16)

def bench(fn, *args, loops=50):
    torch.cuda.synchronize()
    for _ in range(5): _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(loops): _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / loops

# -------------------------------------------------
# 1. 定义FP16基准网络模型
# -------------------------------------------------
class ComplexNetFP16(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x + residual
        x = self.layernorm(x)
        return x

# -------------------------------------------------
# 2. 定义INT8量化网络模型
# -------------------------------------------------
class ComplexNetINT8:
    def __init__(self, fp16_model: ComplexNetFP16, block_size: int):
        self.block_size = block_size
        
        # 量化并存储权重
        w1 = fp16_model.linear1.weight.t().contiguous() # (D_MODEL, D_FFN)
        self.w1_int8, self.w1_exp = quantize_fine_grained(w1, block_size, dim=0)
        self.b1 = fp16_model.linear1.bias
        
        w2 = fp16_model.linear2.weight.t().contiguous() # (D_FFN, D_MODEL)
        self.w2_int8, self.w2_exp = quantize_fine_grained(w2, block_size, dim=0)
        self.b2 = fp16_model.linear2.bias
        
        # LayerNorm 和 ReLU 直接复用
        self.relu = fp16_model.relu
        self.layernorm = fp16_model.layernorm

    def forward(self, x):
        residual = x
        
        # ---- 第一层 Linear + ReLU ----
        # 动态量化输入 x
        assert x.shape[1] % self.block_size == 0
        x_int8, x_exp = quantize_fine_grained(x, self.block_size, dim=1)
        # 计算
        x = int8_gemm_fine_grained(x_int8, x_exp, self.w1_int8, self.w1_exp, self.b1, self.block_size)
        x = self.relu(x)
        
        # ---- 第二层 Linear ----
        # 动态量化中间激活值 x
        assert x.shape[1] % self.block_size == 0
        x_int8, x_exp = quantize_fine_grained(x, self.block_size, dim=1)
        # 计算
        x = int8_gemm_fine_grained(x_int8, x_exp, self.w2_int8, self.w2_exp, self.b2, self.block_size)
        
        # ---- 残差连接和LayerNorm ----
        x = x + residual
        x = self.layernorm(x)
        return x

# -------------------------------------------------
# 误差 & 性能对比
# -------------------------------------------------
total_abs_err, total_rel_err = 0.0, 0.0
print(f"开始计算 {ERROR_AVG_RUNS} 次运行的平均误差...")

for i in range(ERROR_AVG_RUNS):
    # 初始化FP16模型并移至GPU
    fp16_model = ComplexNetFP16(D_MODEL, D_FFN).to(device).to(torch.float16)
    fp16_model.eval()

    # 从FP16模型创建INT8量化模型
    int8_model = ComplexNetINT8(fp16_model, BLOCK_SIZE_K)

    # 创建随机输入
    x = torch.randn(M, D_MODEL, device=device, dtype=torch.float16)

    # 前向传播
    with torch.no_grad():
        y_fp16 = fp16_model(x)
        y_int8 = int8_model.forward(x)

    # 计算并累加误差
    run_abs_err = (y_fp16 - y_int8).abs().mean().item()
    run_rel_err = run_abs_err / y_fp16.abs().mean().item()
    total_abs_err += run_abs_err
    total_rel_err += run_rel_err
    print(f"  Run {i+1}/{ERROR_AVG_RUNS}: 相对误差 = {run_rel_err:.3%}")

avg_abs_err = total_abs_err / ERROR_AVG_RUNS
avg_rel_err = total_rel_err / ERROR_AVG_RUNS

# 性能测试 (使用最后一次循环的模型和数据)
print("\n开始性能基准测试...")
with torch.no_grad():
    t_fp16 = bench(fp16_model.forward, x)
    t_int8 = bench(int8_model.forward, x)

print("\n------------------------------------------------")
print(f"网络: FFN(Linear({D_MODEL},{D_FFN}) -> ReLU -> Linear({D_FFN},{D_MODEL})) + Residual + LayerNorm")
print(f"输入: {M}×{D_MODEL}, Block Size K: {BLOCK_SIZE_K}, 误差平均次数: {ERROR_AVG_RUNS}")
print("------------------------------------------------")
print(f"FP16 端到端性能: {t_fp16*1e3:.3f} ms")
print(f"INT8 端到端性能: {t_int8*1e3:.3f} ms")
print(f"INT8/FP16 速度比: {t_fp16/t_int8:.2f}x")
print(f"平均绝对误差: {avg_abs_err:.5f}    平均相对误差: {avg_rel_err:.3%}")
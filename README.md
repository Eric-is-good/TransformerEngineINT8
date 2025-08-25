# TransformerEngineINT8（画饼版 readme）

[](https://www.google.com/search?q=https://github.com/your-username/TransformerEngineINT8)
[](https://www.google.com/search?q=https://pypi.org/project/transformer-engine-int8/)
[](https://opensource.org/licenses/Apache-2.0)
[](https://www.google.com/search?q=https://your-username.github.io/TransformerEngineINT8/)

**TransformerEngineINT8** 是一个为PyTorch打造的高性能INT8量化加速框架，其设计灵感源自NVIDIA的[Transformer Engine](https://github.com/NVIDIA/TransformerEngine)。它旨在通过一个极其简洁的API，提供无缝的量化感知训练（Quantization-Aware Training, QAT）。

## 概述 (Overview)

在不修改任何模型定义代码的前提下，`TransformerEngineINT8` 允许您使用一个简单的 `autocast` 上下文管理器，将模型中的计算密集型部分（如`Linear`层）自动切换到INT8模式。这使得模型能够在几乎不损失精度的情况下（maybe），享受到INT8带来的显著的内存节省和潜在的性能提升。


## 路线图 (Roadmap)

我们致力于将`TransformerEngineINT8`打造成一个生产级的量化加速库。

  * [ ] 🟡 完成 linear 的 QAT 实验与可行性验证
  * [ ] ⚪️ 核心API (`te_int8_autocast`) 与框架设计
  * [ ] ⚪️ 基于纯PyTorch的量化感知训练（QAT）功能实现
  * [ ] ⚪️ 基于纯PyTorch的INT8推理模拟功能实现
  * [ ] ⚪️ **高性能CUDA Kernels**: 为`Linear`层等实现基于CUTLASS或cuBLAS的INT8 GEMM Kernel，提供真实的性能加速。
  * [ ] ⚪️ **`torch.compile` 后端集成**: 开发一个自定义后端，将框架的优化能力与PyTorch 2.x的编译器无缝集成。
  * [ ] ⚪️ **扩展模块支持**: 增加对`nn.Conv2d`、`nn.LayerNorm`等更多模块的量化支持。
  * [ ] ⚪️ **全面的文档与教程**: 提供详细的API文档、用户指南和性能调优技巧。



## 突出特性 (Key Features)

  * **⚡ 极简的API**: 只需将您的代码包裹在 `te_int8_autocast` 上下文中，即可启用INT8量化，无需手动修改模型。
  * **🚀 量化感知训练 (QAT)**: 内置对QAT的完整支持，通过直通估计器（Straight-Through Estimator）技术，在微调过程中模拟量化效应，最大程度地保持模型精度。
  * ** seamlessly 集成PyTorch**: 作为一个纯粹的PyTorch扩展，与现有的生态系统、模型和训练循环无缝集成。
  * **🔧 模块化与可扩展**: 清晰的架构设计，当前使用纯PyTorch后端进行功能验证，并为未来的高性能CUDA Kernel集成和`torch.compile`后端开发预留了接口。



## 架构与原理 (Architecture & Principles)

`TransformerEngineINT8` 的核心在于其**动态模块替换**机制。当进入`te_int8_autocast`上下文时，框架会：

1.  遍历模型计算图，查找所有目标模块（如 `nn.Linear`）。
2.  将其动态替换为一个内置的、可感知量化的模块 (`QuantizedLinear`)，执行QAT或推理逻辑。
3.  退出上下文时，所有模块将自动恢复原状，确保对原始模型零侵入。

在训练模式下，我们采用**直通估计器 (Straight-Through Estimator, STE)** 来解决量化操作（如`round()`）不可导的问题，从而保证梯度能够顺畅地回传至全精度的“影子权重”，实现真正的端到端微调。



## 安装 (Installation)（即将推出）

您可以通过pip从PyPI安装：

```bash
pip install transformer-engine-int8
```

或者，从源代码安装以获取最新版本：

```bash
git clone https://github.com/your-username/TransformerEngineINT8.git
cd TransformerEngineINT8
pip install -e .
```

## 快速上手 (Quick Start)（即将推出）

体验`TransformerEngineINT8`的强大功能只需两步：首先进行量化感知训练（微调），然后进行量化推理。

```python
import torch
import torch.nn as nn
from transformer_engine_int8 import te_int8_autocast

# 1. 定义或加载一个标准的PyTorch Transformer模型
# (这里我们用一个简单的模型作为示例)
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, 10) # 输出层

    def forward(self, src):
        memory = self.encoder(src)
        return self.output_layer(memory[:, 0, :]) # 取[CLS] token输出

# --- 步骤一：量化感知训练 (QAT) ---

# 加载预训练好的FP32模型
model = SimpleTransformer().cuda()
# model.load_state_dict(torch.load('pretrained_fp32.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
dummy_data = torch.randn(16, 128, 512).cuda()
dummy_target = torch.randint(0, 10, (16,)).cuda()

print("--- Starting Quantization-Aware Training (QAT) ---")
model.train()
optimizer.zero_grad()
# 使用 te_int8_autocast 开启QAT模式
with te_int8_autocast(training=True):
    # 在此上下文中, 所有内部的nn.Linear都将以QAT模式运行
    output = model(dummy_data)
    loss = criterion(output, dummy_target)
    
# 梯度计算和参数更新在上下文外部进行
loss.backward()
optimizer.step()
print("QAT step completed. Loss:", loss.item())


# --- 步骤二：量化推理 ---

# 加载经过QAT微调的模型
# model.load_state_dict(torch.load('qat_tuned.pt'))
model.eval()

print("\n--- Starting Quantized Inference ---")
with torch.no_grad():
    # 使用同一个API，但将training设置为False以进入推理模式
    with te_int8_autocast(training=False):
        # 在此上下文中, nn.Linear将以高性能INT8推理模式运行
        quantized_output = model(dummy_data)

print("Inference completed. Output shape:", quantized_output.shape)

```

-----

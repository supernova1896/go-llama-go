# 作业完成过程与验证记录

## 1. 作业目标

本仓库是 InfiniTensor 大模型与人工智能系统训练营 Triton & 九齿方向的 Llama 推理优化作业。目标是在不明显降低生成文本逻辑质量的前提下，提高 `infer.py` 输出的 `num_tokens_per_second`。

README 对优化范围有明确限制：性能优化必须来源于 Triton 或九齿；不能通过加入 KV cache 等系统级改动获得性能，也不能直接以 `torch.nn.functional.rms_norm` 替换 RMSNorm 实现。

## 2. 基线分析

原始实现集中在 `llama.py`，主要流程如下：

1. `RMSNorm.forward` 使用 `pow`、`mean`、`rsqrt` 和逐元素乘法等多个 PyTorch 操作。
2. `apply_rotary_position_embedding` 将输入切成前后两半，分别执行旋转所需的乘法和加法，最后 `cat`。
3. `apply_scaled_dot_product_attention` 先用 `repeat_interleave` 将 KV head 扩展到 Query head 数，再依次构造 QKᵀ、causal mask、softmax 和 PV 结果。
4. `generate` 每生成一个 token 都重新计算完整的输入和历史序列。由于 KV cache 明确属于限制范围之外，本次没有修改这一行为。

其中 attention 会产生较大的 score 和 softmax 中间张量，并且 GQA 的 K/V 扩展会产生数据复制，是最主要的优化目标。RMSNorm 和 RoPE 则适合通过 Triton 进行算子融合。

## 3. 实现方案

### 3.1 Triton 可选加载

`llama.py` 使用可选导入：

- 安装 Triton 且 CUDA 可用时，允许启用 Triton kernel。
- Triton 未安装、CUDA 不可用、输入不在 CUDA 上或输入不是连续布局时，继续使用原 PyTorch 路径。
- 这样可以在 CPU 环境执行基础导入和回退逻辑，也不会改变模型权重格式。

### 3.2 Triton RMSNorm

新增 `_rms_norm_kernel`，每个 Triton program 处理一个 token 对应的 hidden dimension 行：

```text
x² → 行内归约 → rsqrt(mean(x²) + eps) → x × scale × weight
```

平方和使用 FP32 归约，以降低 FP16/BF16 输入下的误差。`BLOCK_SIZE` 取不小于 hidden size 的 2 次幂，并对尾部元素使用 mask。

### 3.3 Triton RoPE

新增 `_rope_kernel`，保持原仓库的 half-split RoPE 语义：

```text
x0 = x[..., :D/2]
x1 = x[..., D/2:]
y0 = x0 × cos - x1 × sin
y1 = x0 × sin + x1 × cos
```

kernel 直接融合输入读取、sin/cos 读取、旋转计算和输出写回，避免产生多个临时张量。调用方仍按原逻辑先在 `[B, S, H, D]` 布局上应用 RoPE，再转换到 attention 使用的 `[B, H, S, D]` 布局。

### 3.4 Triton causal GQA attention

新增 `_attention_kernel`，每个 program 负责一个 batch、Query head 和 Query position：

1. 根据 Query head 映射到对应 KV head，不物化 `repeat_interleave` 的 K/V。
2. 按 `BLOCK_N=64` 分块读取 K/V。
3. 对 `key_position <= query_position` 的位置保留 causal attention，其余位置屏蔽。
4. 使用 running maximum、running sum 和 value accumulator 实现 online softmax。
5. 直接写出 `[B, Hq, S, D]` attention 输出。

wrapper 检查 Query/KV head 数可整除关系，并限制 head dimension 不超过 256；不满足条件时回退原 PyTorch 实现。实现没有引入 KV cache，也没有调用高层 fused RMSNorm。

### 3.5 推理入口改进

`infer.py` 做了三项与推理测量直接相关的改进：

- 使用 `torch.inference_mode()`，避免推理阶段创建 autograd graph。
- 使用 `torch.device(device).type` 判断 CUDA，兼容 `cuda:0` 等设备写法。
- 校验生成 token 数、warmup 次数和 profiling 次数，避免无效参数导致除零或负长度行为。

这些改动不改变生成算法，仍然是原来的 greedy `argmax` 解码。

## 4. 依赖与环境

`requirements.txt` 现在显式声明：

- `torch`
- `transformers`
- `safetensors`
- `triton`

本地验证使用仓库内 `.venv`，避免覆盖系统 Python。环境检查命令：

```bash
.venv/bin/python - <<'PY'
import torch
import triton
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
print(triton.__version__)
PY
```

## 5. 验证方法

### 5.1 静态检查

```bash
.venv/bin/python -m py_compile llama.py infer.py
.venv/bin/python -m compileall -q llama.py infer.py
 git diff --check
```

### 5.2 算子数值对齐

在 CUDA/Triton 环境中，应该分别使用随机输入比较 Triton 路径和原 PyTorch reference：

- RMSNorm：不同 batch、sequence length、hidden size 和 FP16/BF16。
- RoPE：不同 batch、sequence length、head 数和 head dimension。
- Attention：MHA、GQA、不同 sequence length、causal 边界和非整块长度。

建议记录：

```text
最大绝对误差 = max(abs(triton - reference))
最大相对误差 = max(abs(triton - reference) / (abs(reference) + 1e-6))
```

### 5.3 端到端验证

使用 README 中相同的模型、prompt、设备、`max-new-tokens`、warmup 和 profiling 参数，对比优化前后的：

- 生成文本。
- `average_time`。
- `num_tokens_per_second`。
- 多次运行的波动。

示例：

```bash
srun --cpus-per-task=16 --mem=64G --gres=gpu:1 \
.venv/bin/python infer.py \
  --model /data/shared/models/Llama-3.2-1B/ \
  --prompts "your prompt" \
  --max-new-tokens 64 \
  --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3
```

## 6. 实际验证结果

### 6.1 环境

本机使用仓库内 `.venv` 完成安装和验证：

```text
Python 3.12.3
PyTorch 2.13.0+cu130
Triton 3.7.1
GPU: NVIDIA GB10
compute capability: 12.1
CUDA driver: 13.0
```

`torch.cuda.is_available()` 返回 `True`，三个 Triton kernel 均完成了实际 CUDA 编译和执行。

### 6.2 数值对齐

使用 Triton 路径与强制关闭 Triton 后的原 PyTorch reference 进行比较。测试包含 FP16、BF16、GQA、MHA、序列长度 1、9、17、65，以及 head dimension 32、64、128。

代表性 FP16 结果：

| 算子 | 测试形状 | 最大绝对误差 | 有限值检查 |
|---|---|---:|---|
| RMSNorm | `[2, 5, 128]` | `0.001953125` | 通过 |
| RoPE | `[2, 7, 4, 64]` | `0.0078125` | 通过 |
| GQA attention | Q `[2,8,17,64]`，KV `[2,2,17,64]` | `0.001953125` | 通过 |
| RMSNorm | hidden size `96/128`，FP16/BF16 | `0.015625` 以内 | 通过 |
| RoPE | sequence `1/9/65`，head `1/3/5` | `0.00390625` 以内 | 通过 |
| Attention | sequence `1/9/65`，MHA/GQA | `0.00292969` 以内 | 通过 |

测试过程中首次发现 RoPE kernel 的行跨度错误：wrapper 传入了整段序列的跨度，导致同一 program 的后续行读取错误数据。修正为单个 head 的 `head_dim` 行跨度后，NaN 消失且上述测试通过。

随机小模型端到端前向测试也通过：Triton 开启和关闭时，最终 normalized hidden states 最大绝对误差为 `0.00390625`，输出全部为有限值。

### 6.3 性能观察

在随机输入 Q `[2,8,129,64]`、K/V `[2,2,129,64]` 上进行了 3 次预热和 10 次测量：

```text
PyTorch reference attention: 0.063 ms
Triton attention:            0.090 ms
```

这个小形状下自定义 kernel 尚未快于 PyTorch reference，说明当前 Triton attention 仍需要进一步针对目标 Llama 形状调优，不能据此宣称已经达到作业要求的 80% 性能提升。当前实现的主要收益是避免 GQA K/V 物化和完整 attention 中间张量；最终评分应以训练营指定的 Llama-3.2-1B、固定 prompt 和 benchmark 参数为准。

本机没有找到 `/data/shared/models/Llama-3.2-1B/` 或其他本地模型权重，因此无法执行真实模型的 `infer.py` 端到端生成和 token/s 对比，也无法验证最终生成文本质量。获得模型目录后，应按 README 示例补充真实 benchmark 结果。

## 7. 模型获取与最终 benchmark

### 7.1 推荐模型来源

优先使用训练营服务器已经提供的模型目录，例如 README 中的：

```text
/data/shared/models/Llama-3.2-1B/
```

如果服务器没有该目录，可以从 Hugging Face 的 Meta 官方仓库获取 Llama 3.2 1B Instruct：

<https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>

该模型需要 Hugging Face 账号、访问 token，并且需要先接受 Meta 的模型许可协议。登录和下载命令如下：

```bash
.venv/bin/hf auth login

mkdir -p "$HOME/models/Llama-3.2-1B-Instruct"
.venv/bin/hf download \
  meta-llama/Llama-3.2-1B-Instruct \
  --local-dir "$HOME/models/Llama-3.2-1B-Instruct"
```

也可以使用 Python API：

```bash
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    local_dir=os.path.expanduser("~/models/Llama-3.2-1B-Instruct"),
)
PY
```

下载完成后，确认模型目录包含 `config.json`、tokenizer 文件和权重文件。当前仓库的 `from_pretrained` 读取固定的 `model.safetensors`，如果 Hugging Face 下载结果是多个 `model-0000x-of-0000y.safetensors` 分片，则需要先适配分片权重加载，不能直接假设当前代码可以读取。

### 7.2 benchmark 命令

模型准备好后，用相同参数分别运行 baseline 和优化版本。当前仓库的 baseline 可以通过临时关闭 Triton 快路径进行对照；正式评分应固定相同 GPU、prompt、batch、dtype、输入长度、warmup 和 profiling 次数：

```bash
.venv/bin/python infer.py \
  --model "$HOME/models/Llama-3.2-1B-Instruct" \
  --prompts "The emergence of deep learning domain-specific languages has substantially reduced the obstacles in developing high-performance, cross-platform compute kernels, but current DSLs" \
  --max-new-tokens 64 \
  --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3
```

需要记录两次运行输出中的：

- `texts`：确认生成文本逻辑质量没有明显退化。
- `average_time`：比较平均推理时间。
- `num_tokens_per_second`：计算性能提升。

基础要求的目标是：

```text
optimized_num_tokens_per_second / baseline_num_tokens_per_second >= 1.8
```

### 7.3 当前已完成工作

- 完成原始 `llama.py` 的性能热点分析。
- 完成合规的 Triton RMSNorm kernel。
- 完成合规的 Triton half-split RoPE kernel。
- 完成合规的 Triton online-softmax causal GQA attention kernel。
- 避免 GQA attention 中显式物化 K/V head 扩展。
- 为 Triton 不可用、CPU、非连续布局和不支持 shape 保留 PyTorch fallback。
- 在 `infer.py` 中启用 `torch.inference_mode()`。
- 修正 `cuda:0` 等设备名称下的同步判断。
- 增加 profiling 参数的边界检查。
- 补充 `safetensors` 和 `triton` 直接依赖。
- 完成 Python 语法、CLI、差异格式检查。
- 在 NVIDIA GB10 上实际编译并运行三个 Triton kernel。
- 完成 FP16/BF16、多序列长度、MHA/GQA、多 head dimension 的数值对齐测试。
- 修复并验证 RoPE kernel 的行跨度 bug。
- 完成随机小模型端到端前向一致性验证。
- 将实现过程、合规性说明、测试结果和限制写入本文档。

### 7.4 仍需完成工作

- 获取训练营指定的 Llama-3.2-1B 权重，优先使用服务器已经提供的 `/data/shared/models/Llama-3.2-1B/`。
- 确认权重是否为单个 `model.safetensors`；若为分片，需要补充分片加载支持或使用作业指定的单文件权重。
- 在真实模型和评分 prompt 上测量 PyTorch baseline。
- 在完全相同的环境和参数下测量 Triton 版本的 `num_tokens_per_second`。
- 对真实模型生成文本做 baseline/优化版本对照。
- 根据真实 Llama 形状调优 attention 的 `BLOCK_N`、`BLOCK_D`、warp 数和输入布局。
- 对端到端生成进行多次 warmup/profile，记录平均值、方差和性能提升百分比。
- 只有真实 benchmark 达到至少 80% 性能提升且文本质量合格后，才能确认满足基础评分要求；进阶筛选还需要结合评审样本的平均值和标准差判断。

## 8. 合规性说明

本次实现只增加 Triton 自定义算子及其 fallback，没有：

- 添加 KV cache。
- 改变模型的生成策略。
- 使用 `torch.nn.functional.rms_norm` 规避 Triton 实现。
- 修改模型权重格式或通过外部推理框架替换模型。

因此优化的主要性能来源是 Triton kernel 对 attention、RMSNorm 和 RoPE 的融合，以及 attention 内部避免 GQA K/V 复制。

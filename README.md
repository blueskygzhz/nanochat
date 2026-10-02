# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is the simplest experimental harness for training LLMs. It is designed to run on a single GPU node, the code is minimal/hackable, and it covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, and inference. For example, you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $48 (~2 hours of 8XH100 GPU node) and then talk to it over a simple CLI. On a spot instance, the total cost can be closer to ~$15. More generally, nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability happens to be approximately depth 26). All other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) are calculated automatically in an optimal way.

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Time-to-GPT-2 Leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train GPT-2 grade model and talk to it. The current leaderboard looks as follows:

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100 (e.g. at the current ~$3/GPU/hr, an 8XH100 node is ~$24/hr, so 2 hours is ~$48).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

## Getting started

### Setup

nanochat uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```

### 本分支：实验性 7B MoE 训练

`runs/moe7b.sh` 配置为 **7.26B 总参数 MoE**，不是 7B 稠密模型：24 层、宽度 2048、16 个 Q heads / 4 个 KV heads、48 个路由专家（top-6）+ 2 个共享专家。激活 Transformer 参数约 1.35B，含 LM head 的激活矩阵约 1.42B。默认约 49.6B tokens 仅为首轮实验预算，不保证收敛或达到其他 7B 模型的能力。

先完成上面的环境安装，并将 `NANOCHAT_BASE_DIR` 设在容量充足的持久化磁盘。参考硬件为单机 8×80GB CUDA GPU；所有参数和梯度仍在每卡复制，只有优化器状态分片，实际显存与速度必须先测。

| 操作 | 命令 | 说明 |
|---|---|---|
| 模型预检查 | `bash runs/moe7b.sh check` | 默认模式；不下载语料、不分配 7B 权重、不训练 |
| 准备语料与 tokenizer | `bash runs/moe7b.sh prepare` | 默认下载 1400 个训练分片，可用 `SHARDS` 调整；预留百 GB 级空间 |
| 7B 冒烟验证 | `bash runs/moe7b.sh smoke` | 真实 7B 模型跑 5 步，每步一个 microbatch；保存至独立的 `moe7b-smoke` 标签 |
| 正式预训练 | `bash runs/moe7b.sh train` | 不自动执行 SFT；默认 BF16、每卡 batch=1 |
| 恢复预训练 | `RESUME_STEP=2000 bash runs/moe7b.sh train` | 架构、world size、Muon 桶大小和总 batch 必须与原训练一致 |
| 评测 / SFT | `bash runs/moe7b.sh eval` / `bash runs/moe7b.sh sft` | SFT 继承分块 loss、激活重计算和优化器桶配置 |

显存余量足够后，可将 `DEVICE_BATCH` 逐步调至 2 或 4；脚本自动用梯度累积维持 `TOTAL_BATCH`。`LOSS_CHUNK_SIZE=512` 限制单个 logits 块，`MUON_BUCKET_MB=128` 将大专家组拆桶并逐桶通信/更新。这两个设置以显存为优先，不保证更高吞吐。最小桶还受完整矩阵和 rank 对齐约束，128 MiB 不是整个优化器的硬显存上限。

默认关闭 FP8；先取得 BF16 基线，再在支持的 GPU 上以 `FP8=1` 单独验证数值与速度。路由专家仍使用 BF16，未接入 grouped GEMM 或 expert parallelism。`NO_COMPILE=1` 可用于定位编译问题；无 FA3 时可先用 `WINDOW_PATTERN=L` 测试，注意这会改变模型窗口配置。

模型 checkpoint 保持兼容；旧优化器 checkpoint 恢复须使用原 `MUON_BUCKET_MB`（旧版为 0），不能直接套用新桶布局。每个保存点的权重与全部优化器分片合计约 60GB，需为多次保存预留空间。数据恢复仍为 row-group 级近似恢复，并非逐 token 精确重放。正式训练前建议至少做数百步稳定性实验，依据 `val/bpb`、step time、显存峰值决定预算；不能仅按激活参数套用稠密模型缩放定律。

### 可选 DeepSeek 风格 MLA

默认仍为 GQA。设置 `ATTENTION_TYPE=mla` 可启用 Multi-head Latent Attention：KV 联合低秩压缩、解耦 RoPE、可选 Q 低秩投影；latent RMSNorm 使用可训练权重并归入 AdamW。实现位于 `nanochat/mla.py`，参考 [DeepSeek 官方 MLA](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。

| 操作 | 命令 |
|---|---|
| MLA 参数预检查 | `ATTENTION_TYPE=mla bash runs/moe7b.sh check` |
| MLA 冒烟训练 | `ATTENTION_TYPE=mla bash runs/moe7b.sh smoke` |
| MLA 正式训练 | `ATTENTION_TYPE=mla bash runs/moe7b.sh train` |
| 从 MLA checkpoint 做 SFT | `ATTENTION_TYPE=mla bash runs/moe7b.sh sft` |
| MLA 评测 | `ATTENTION_TYPE=mla bash runs/moe7b.sh eval` |

默认模型标签自动改为 `moe7b-mla`，避免与 GQA 混用。默认 `Q_LORA_RANK=0`（直接投影 Q）、`KV_LORA_RANK=512`、`QK_NOPE_HEAD_DIM=128`、`QK_ROPE_HEAD_DIM=64`、`V_HEAD_DIM=128`。如需压缩 Q，可设置 `Q_LORA_RANK=512`。也可直接给 `scripts.base_train` 传 `--attention-type=mla`、`--q-lora-rank`、`--kv-lora-rank`、`--qk-nope-head-dim`、`--qk-rope-head-dim`、`--v-head-dim`；这些架构字段随 checkpoint 保存，SFT/推理自动恢复。

训练和首段 prefill 将 latent 展开为各头 K/V，使用 PyTorch SDPA；后续单 token 或多 token 续写使用吸收形式：Q 非位置部分乘 K 上投影权重，与缓存 latent 做 attention；先对 latent 加权汇总，再做 V 上投影。持久缓存只保存 `[layers, batch, length, kv_lora_rank]` 和 `[layers, batch, length, rope_dim]`，不保存或重建历史各头 K/V。支持因果/滑窗、前缀复制到多个采样行、reset、容量校验；仅支持批内统一位置的推理缓存，不支持 ragged continuous batching。

默认 24 层配置下，BF16 的每 token、每行缓存（不含少量位置/smear 状态）：GQA 为 `24 × 2 × 4 × 128 × 2 = 49152` 字节，MLA 为 `24 × (512+64) × 2 = 27648` 字节，减少 **43.75%**。这仅是持久 KV 容量；吸收式参考实现会两次使用 latent，并有 attention scores 等临时张量，不能把容量降低直接等同于带宽或延迟提升。`check` 在 CPU 默认 FP32，显示的字节数会翻倍，并打印实际 compute dtype。

兼容性与限制：
- MLA 禁用原 GQA 的 Value Embedding 和完整 Q/K head norm，使用 latent norm 与 `(nope_dim+rope_dim)^(-1/2)` scale，以保持投影吸收成立；保留 nanochat 的 RoPE 方向/基频，不包含 DeepSeek 的 YaRN 扩展。其他 MoE、smear/backout、loss 分块和激活重计算不变。
- 默认 MLA 总参数约 **7.133B**（32768 词表），含 LM head 的激活矩阵约 **1.494B**；切换后参数量和默认 token 预算会变化，不应继续套用 GQA 的精确统计。
- **GQA checkpoint 不能直接当 MLA checkpoint 恢复**，也不支持直接导入官方 DeepSeek 权重。旧 GQA checkpoint 缺失这些字段时继续使用 GQA；切换 MLA 需要新训练或另行做转换/蒸馏。
- 首版 MLA 注意力不参与本地 FP8 Linear 转换；`--fp8` 仍可作用于其他符合条件的层。尚未接入 FlashMLA、专用 Triton kernel、分页 KV 或张量并行。SDPA 能否选择高效 CUDA kernel 取决于硬件、维度和 mask；带滑窗的训练可能退回较慢路径。先测小模型数值和真实 GPU 显存/吞吐，再长跑。

### 可选 MTP 训练与投机解码

首版采用**单步、token 条件化的轻量 MTP 头**：`norm(h_t)` 与下一 token 的 embedding 拼接，经 `2d→d` 融合投影和残差 ReLU² MLP，使用共享 LM head 预测 `x_(t+2)`。这是本项目的辅助草稿头，不是 DeepSeek/HY3 的完整 Transformer MTP 层；没有额外的 MTP attention cache，也没有多层草稿树。

默认关闭，使用 `MTP=1` 或 `scripts.base_train --mtp` 开启。默认辅助权重 0.1，可通过 `MTP_LOSS_WEIGHT` / `--mtp-loss-weight` 调整。训练目标为 `CE + MoE_aux + weight*MTP_CE`，MTP 梯度同时进入共享主干、embedding、LM head 和草稿头。行内移位后才展平，沿用 SFT 的 target mask；两步中任意一步为忽略位置或跨 BOS 边界时不计 MTP loss。分块损失会对整个草稿分支重计算；无有效辅助标签时返回图连接的零。评估以及 `loss_reduction='none'/'sum'` 不加入 MTP loss，BPB 和 RL 逐 token loss 不变。

| 操作 | 命令 |
|---|---|
| GQA+MTP 预检查 | `MTP=1 bash runs/moe7b.sh check` |
| MLA+MTP 预检查 | `ATTENTION_TYPE=mla MTP=1 bash runs/moe7b.sh check` |
| 冒烟 / 正式预训练 | `MTP=1 bash runs/moe7b.sh smoke` / `MTP=1 bash runs/moe7b.sh train` |
| SFT，继续训练已有 MTP 头 | `MTP=1 bash runs/moe7b.sh sft` |
| 使用 SFT checkpoint 投机聊天 | `python -m scripts.chat_cli --model-tag=moe7b-mtp --temperature=0 --speculative` |
| 对比普通与投机解码 | `python -m scripts.infer_bench -i sft -g moe7b-mtp --speculative --prompt-tokens=512 --decode-tokens=128` |

启用时默认标签为 `moe7b-mtp` 或 `moe7b-mla-mtp`；对 MLA 的后续操作同样设置 `ATTENTION_TYPE=mla`。SFT 自动加载并训练草稿头，可用 `--mtp-loss-weight=0` 禁用辅助目标（这不是冻结共享主干）。旧 checkpoint 没有 MTP 参数时仍可普通推理，但不能直接加 `--speculative`；也不能将旧 optimizer 原样恢复为新增草稿头的模型。当前不提供旧模型的自动头迁移，需要从 MTP 配置开始训练。SFT 保存完整 MTP 配置和权重；RL 尚不继续训练 MTP 辅助目标。

解码过程：主头从当前前缀确定 token `a`，MTP 根据该前缀隐藏态和 `a` 提议 `b`；主干一次处理 `[a,b]`，用 `a` 位置的主头 argmax 检验 `b`。接受则保留两个位置的缓存；拒绝则只保留 `a`，恢复 GQA/MLA 长度与 pre-smear embedding，下一步使用主头纠正 token。草稿从不绕过主模型验证。工具表达式、强制工具结果及工具边界走普通逐 token 路径；只对已提交的 token 执行工具状态变更。

限制与性能：
- 仅支持 `temperature=0`、`num_samples=1`、`model.eval()`；其他组合明确报错，普通采样不受影响。`max_tokens=None` 的投机路径使用训练上下文剩余长度作为预算。没有概率接受/拒绝采样、continuous batching、CUDA Graph 或多 GPU 专用投机调度。
- 在相同主模型概率下保持 greedy 语义；不同长度 GEMM/attention 的浮点舍入可能影响近似并列 argmax，实际设备上应逐 token 对比。基准程序检查输出与 mask 相同，若不一致就不报告加速比。
- `stats` 可记录草稿调用数、实际验证草稿数、接受数、验证调用数、主干处理 token 数、提交 token 数及强制 token 数。接受率是接受草稿数/验证草稿数，不含因工具边界跳过的提案；提交数包含终止标记和工具注入 token。基准按完整生成流计时，不把一次 yield 当成一次 GPU decode，也不套用普通解码 MFU/MBU 公式。
- 首版 MTP 训练与 `--fp8` 组合会提前报错：移位后的 token 数不满足当前 FP8 backward 对齐约束。先使用 BF16/FP32。草稿头是否带来吞吐收益取决于训练后的接受率和硬件，不能因主干调用减少就保证加速。
- 宽度 2048 时新增约 41.94M 参数；普通目标模型推理 FLOPs 不包含未执行的草稿头，训练 FLOPs 会额外计算草稿头和第二次 LM-head 投影。token 数据预算仍按主干 active 参数决定，不把辅助标签计作额外训练 token。

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~1.5 hours to run. Once it's done, you can talk to your model over the CLI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and chat:

```bash
python -m scripts.chat_cli
```

Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :). An example conversation with a speedrun model:

> **User:** Hello whats up?
>
> **Assistant:** Hello! How can I help you today?
>
> **User:** Why is the sky blue?
>
> **Assistant:** The sky is blue due to an optical illusion called the Rayleigh Scattering of Light, which occurs when sunlight passes through tiny particles in the air. These particles are made up of tiny blue and violet particles that cause the light to bend in a particular way.
>
> **User:** That's cool. Write a poem about it.
>
> **Assistant:** In the twilight's glow, the sky is blue,<br>
> A canvas of shimmering blue, where skies unfold,<br>
> A world of blue, where blue and more blue,<br>
> Are woven together in a tapestry so fine,<br>
> Where every hue seems to sing a story.<br>
> ...

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE score)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model (which is of most interest at the moment) happens to be somewhere around d24-d26 range with the current code. But any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT supports this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   └── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── infer_bench.py              # Inference: latency/throughput/VRAM bench
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   └── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
├── tests
│   ├── test_attention_fallback.py  # FA3/SDPA attention fallback
│   ├── test_engine.py              # Inference engine, KV cache
│   ├── test_execution.py           # Sandboxed code execution
│   ├── test_optim.py               # MuonAdamW optimizer (needs GPU)
│   ├── test_tasks.py               # Task slicing, mixtures, HubDataset
│   └── test_tokenizer.py           # BPE round-trips, chat rendering
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~1.5 hours (down from 3h), but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT

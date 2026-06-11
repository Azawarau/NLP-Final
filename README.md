# 从大语言模型中抽取长文本表示 — 实验代码

南京大学 NLP 大作业基础实验阶段实现：在 **QMSum**、**2WikiMultihop（LongEmbed）**、**ArguAna** 上对比 **PromptEOL** 与 **mean-pooling**，并支持不同 Transformer **层** 的消融。

## 环境

- Python 3.10+
- GPU 建议 ≥ 24GB（`Mistral-7B-Instruct-v0.3`）；显存不足可用 `--fallback` 加载 `Qwen2-1.5B-Instruct`（作业说明会扣分）

```bash
cd D:\NLP-Final
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

首次运行前可预下载资源：

```bash
python scripts/download_assets.py          # 数据集 + Mistral-7B
python scripts/download_assets.py --skip-model   # 仅数据集
```

模型默认保存到 `models/Mistral-7B-Instruct-v0.3`，`configs/basic.yaml` 已指向该路径。

## 目录结构

```
configs/basic.yaml      # 默认实验配置
src/
  prompts.py            # PromptEOL 模板
  pooling.py            # mean / last-token 池化
  llm_encoder.py        # 模型加载与编码（MTEB 接口）
  evaluate.py           # MTEB 评估与结果汇总
scripts/
  run_basic.py          # 基础实验主入口
  run_layer_ablation.py # 不同层表示质量消融
  smoke_test.py         # 单元自检（可不加载大模型）
reports/
  basic_experiment.md   # 实验报告撰写模板
results/                # 运行输出（gitignore）
```

## 快速自检

```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --with-model --fallback
```

## 基础实验（3.1）

对比两种方法在**最后一层**（`layer=-1`）上的检索表现：

```bash
python scripts/run_basic.py
```

指定模型与输出目录：

```bash
python scripts/run_basic.py --model mistralai/Mistral-7B-Instruct-v0.3 --output-dir results/basic_mistral
```

仅跑某一种方法：

```bash
python scripts/run_basic.py --methods prompteol
python scripts/run_basic.py --methods mean
```

结果写入 `results/basic/basic_results.json` 及各子目录 `prompteol_layer-1.json` 等。

### MTEB 任务对应

| 作业数据集 | MTEB 任务名 |
|-----------|-------------|
| QMSum | `LEMBQMSumRetrieval` |
| 2WikiMultihop | `LEMBWikimQARetrieval` |
| ArguAna | `ArguAna` |

指标由 MTEB 自动计算（如 **nDCG@1/10**、**MAP**、**MRR**、**Recall** 等，以任务配置为准）。

## 不同层消融（proposal 基础阶段第 2 点）

```bash
python scripts/run_layer_ablation.py
```

或手动指定层（Mistral-7B 共 32 层，可用 `1..32` 或 `-1`）：

```bash
python scripts/run_layer_ablation.py --layers 8 16 24 32
```

## 方法说明

### PromptEOL

模板（`src/prompts.py`）：

```text
This sentence : "{text}" means in one word:"
```

取指定层 **最后一个非 padding token** 的 hidden state 作为句向量（与 Jiang et al., EMNLP 2024 Findings 一致）。

### mean-pooling

直接以原文本输入模型，对指定层所有 token hidden states 做 **attention mask 加权平均**。

### 层索引约定

- `layer=-1`：最后一层 Transformer 输出（默认）
- `layer=k`（`1 ≤ k ≤ num_hidden_layers`）：第 `k` 层 block 的输出 hidden state

## 实验报告

见 `reports/basic_experiment.md`，将 `results/` 中的 JSON 填入表格后导出 PDF 提交。

## 后续阶段（未在本目录实现）

- 分析阶段：PromptEOL vs mean 对比、RoPE 频率实验 → 见作业 3.2
- 进阶阶段：关键词加权、分块聚合、语义压缩等 → 见 `proposal.md` 第三节

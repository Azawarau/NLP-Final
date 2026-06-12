# 从大语言模型中抽取长文本表示

> 南京大学 NLP 期末实验 | 实验日期：2026-05-17/06-12
> 模型：Mistral-7B-Instruct-v0.3 (4-bit 量化)
> 数据集：QMSum / 2WikiMultihop / ArguAna

---

## 主要结论

### 基础实验（§1）

**Mean-pooling 在所有三个数据集上均大幅优于 PromptEOL**（提升 5.9× ~ 6.7×）。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| PromptEOL | 0.0191 | 0.0182 | 0.0427 |
| **mean-pooling** | **0.1133** | **0.1134** | **0.2871** |

层消融显示 mean-pooling 在所有层上均远超 PromptEOL，最差层仍优于 PromptEOL 的最佳层，且差距随层数加深而扩大（layer 8: 2.6× → layer 32: 5.9×）。

### 分析任务（§2）

**位置贡献实验**揭示 mean-pooling 在所有位置片段上几乎完全均匀（uniformity > 0.999），PromptEOL 严重偏向尾部（60%~100%）。ArguAna PromptEOL 在文本开头出现负余弦相似度（−0.059），表明 last token 对开头的"记忆"已被后续内容完全覆盖并翻转。

**RoPE 频率分析**揭示了完整的因果链：
1. Mistral 的 base=1,000,000 使 2048 tokens 下仅 48.4% 维度对能有效区分位置
2. FFT 频谱分析证明低频化是 Transformer 结构自身的涌现属性（mean 与 PromptEOL 频谱几乎相同）
3. 频段滤波实验发现 mean-pooling 语义完全在低频通道（低通 cos_sim=1.000），而 PromptEOL 语义在高频通道（高通 cos_sim=0.643）
4. 高频维度波长最短（≤900 tokens），在长文本上最先饱和——**PromptEOL 的语义恰好落在位置编码最先失效的频段**

### 进阶实验（§3）

| 研究点 | QMSum | 2WikiMultihop | ArguAna | 适用场景 |
|------|:---:|:---:|:---:|------|
| RP1 加权池化 | −8.5% | +4.8% | +1.4% | 信息密集型短中文本 |
| RP2 分块聚合 | **+90.0%** | **+27.9%** | −3.7% | 超长文本 (>1000 tokens) |
| RP3 语义压缩 | +50.5% | +4.9% | −5.6% | 高冗余长文本 |

ArguAna 三种方法均衰退，验证了这些方法的前提是**文本足够长**（≥1000 tokens）。RP3 基于 TextRank/LexRank 的句子中心度算法（Mihalcea & Tarau, 2004）。

---

## 环境

- Python 3.10+
- GPU ≥ 16GB VRAM（Mistral-7B 4-bit 量化约需 6GB）

```bash
cd D:\NLP-Final
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

首次运行前预下载资源：

```bash
python scripts/download_assets.py              # 数据集 + Mistral-7B
python scripts/download_assets.py --skip-model # 仅数据集
```

---

## 目录结构

```
├── README.md
├── reports/final_report.md        # 完整实验报告
│
├── scripts/
│   ├── standalone_eval.py               # 基础实验（全量评估，最终层）
│   ├── run_layer_ablation.py            # 层消融（QMSum + 2WikiMultihop）
│   ├── run_arguana_ablation_fast.py     # 层消融（ArguAna）
│   ├── rope_frequency_analysis.py       # RoPE 频率分析
│   ├── position_contribution_analysis.py # 位置贡献分析
│   ├── run_advanced_sampled.py          # RP1 加权池化
│   ├── run_rp23_final.py                # RP2+RP3 正式评估（QMSum/2Wiki）
│   ├── run_rp23_fast.py                 # RP2+RP3 快速评估（ArguAna）
│   ├── download_assets.py               # 预下载数据与模型
│   └── smoke_test.py                    # 环境自检
│
├── src/
│   ├── llm_encoder.py                   # 模型加载与编码
│   ├── pooling.py                       # mean / last-token 池化
│   ├── prompts.py                       # PromptEOL 模板
│   ├── evaluate.py                      # 检索指标计算
│   ├── advanced_pooling.py              # 加权池化
│   ├── chunk_encoder.py                 # 分块编码
│   └── semantic_compression.py          # 语义压缩（TextRank 原理）
│
├── results/
│   ├── basic/                           # 基础实验 JSON
│   ├── layer_ablation/                  # 层消融 JSON
│   ├── advanced/                        # 进阶实验 JSON
│   ├── rope_analysis/                   # RoPE 分析 JSON
│   └── position_analysis/               # 位置贡献 JSON
│
└── configs/basic.yaml                   # 默认实验配置
```

---

## 快速自检

```bash
python scripts/smoke_test.py
```

---

## 复现实验

所有实验数据即为 `results/` 目录下的 JSON 文件。重新运行将覆盖已有结果。

### 一、基础实验（§1.2—§1.3）

```bash
# PromptEOL + mean-pooling 全量评估（最终层）
python scripts/standalone_eval.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --methods prompteol mean --layers -1 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --max-length 512 --batch-size 8 --output-dir results/basic

# 层消融（QMSum + 2WikiMultihop，layer 8/16/24/32）
python scripts/run_layer_ablation.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --layers 8 16 24 32 \
  --output-dir results/layer_ablation

# 层消融（ArguAna，layer 8/16/24/32）
python scripts/run_arguana_ablation_fast.py
```

### 二、分析实验（§2.1—§2.2）

```bash
# RoPE 频率分析（理论 + 频谱 + 滤波）
python scripts/rope_frequency_analysis.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --output-dir results/rope_analysis --max-samples 20 --batch-size 1

# 位置贡献分析
python scripts/position_contribution_analysis.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --output-dir results/position_analysis --max-samples 20 --batch-size 1
```

### 三、进阶实验（§3.1—§3.5）

```bash
# RP1 加权池化（max_corpus=200, max_queries=50）
python scripts/run_advanced_sampled.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets ArguAna QMSum 2WikiMultihop \
  --max-length 512 --batch-size 4 --max-corpus 200 --max-queries 50 \
  --output-dir results/advanced

# RP2+RP3 正式评估（QMSum + 2WikiMultihop, max_corpus=100）
python scripts/run_rp23_final.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets QMSum 2WikiMultihop \
  --max-length 512 --max-corpus 100 --max-queries 50 \
  --chunk-size 1024 --chunk-overlap 128 --compression-ratio 0.3 \
  --output-dir results/advanced

# RP2+RP3 快速评估（ArguAna, max_corpus=200）
python scripts/run_rp23_fast.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets ArguAna \
  --max-length 512 --batch-size 4 --max-corpus 200 --max-queries 50 \
  --output-dir results/advanced
```

---

## 方法说明

### PromptEOL

```
模板: 'This sentence : "{text}" means in one word:"'
```

指定层**最后一个非 padding token** 的 hidden state 作为文本表示。来源于 LLM2Vec（BehnamGhader et al., 2024）。

### Mean-pooling

直接以原文本输入模型，对指定层所有 token hidden states 做 **attention mask 加权平均**，然后 L2 归一化。

### 进阶方法

| 研究点 | 原理 | 来源 |
|------|------|------|
| RP1 加权池化 | token 按 L2 范数加权后平均 | `src/advanced_pooling.py` |
| RP2 分块聚合 | tokenizer 精确分块 → 展平编码 → 按原文聚合 | `src/chunk_encoder.py` |
| RP3 语义压缩 | 句子向量化 → 中心度排序 → Top-30% → 原文顺序还原 | `src/semantic_compression.py`（TextRank 原理） |

---

## 关键设计说明

### 基础实验 vs 进阶实验的 Baseline 差异

- **基础实验（§1）**：全量数据集，无采样（QMSum: 1,527 queries / 2Wiki: 300 / ArguAna: 1,406）
- **进阶实验（§3）**：采样数据集（max_corpus=100~200, max_queries=50），以节省 GPU 时间（进阶方法涉及数十种变体配置，全量运行不可行）
- 采样后 corpus 缩小使 nDCG 绝对值偏高，但所有对比在同一子集上完成，**相对百分比有效**

### 分块参数为何不同

QMSum/2Wiki（3000~8000 tokens）用 chunk_size=1024，ArguAna（100~300 tokens）用 chunk_size=512——分块窗口必须匹配目标文本的长度分布。

### RoPE 分析无需 GPU 训练

频段分类、有效分辨率、Base 敏感性均为纯数学推导（θ_i = base^(−2i/d)，波长 = 2π/θ_i）。频谱分析和频段滤波需要模型前向传播提取 hidden states（仅推理，无梯度），后续 FFT/IFFT 在 CPU 上用 NumPy 完成。

---

## 参考文献

| # | 论文 | 本项目使用 |
|---|------|------|
| 1 | Su et al. "RoFormer." arXiv:2104.09864, 2021 | RoPE 频率分析理论基础 |
| 2 | Jiang et al. "Mistral 7B." arXiv:2310.06825, 2023 | 模型配置来源（base=1M, head_dim=128） |
| 3 | Zhu et al. "LongEmbed." arXiv:2404.12096, 2024 | QMSum/2Wiki 数据集来源 |
| 4 | BehnamGhader et al. "LLM2Vec." arXiv:2404.05961, 2024 | **核心参考**：PromptEOL 方法与对比框架 |
| 5 | Mihalcea & Tarau. "TextRank." EMNLP, 2004 | RP3 语义压缩算法原理 |
| 6 | Peng et al. "YaRN." arXiv:2309.00071, 2023 | RoPE base 敏感性分析理论背景 |
| 7 | Vaswani et al. "Attention Is All You Need." NeurIPS, 2017 | 因果注意力 + 位置编码理论基础 |

---

完整实验报告见 [reports/final_report.md](reports/final_report.md)。

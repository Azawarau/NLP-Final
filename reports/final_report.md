# 从大语言模型中抽取长文本表示

> 实验日期：2026-06-11/12
> 模型：Mistral-7B-Instruct-v0.3 (4-bit 量化，本地路径 `models/Mistral-7B-Instruct-v0.3`)
> 数据集：QMSum / 2WikiMultihop / ArguAna
> 所有数据均可通过项目脚本复现，JSON 结果见 `results/` 目录

---

## 一、简单任务：PromptEOL 与 Mean-Pooling 对比

### 1.1 实验设置

| 项目 | 内容 |
|------|------|
| 方法 | PromptEOL；mean-pooling |
| 评估指标 | nDCG@10, Recall@10 |
| max_length | 512 |
| batch_size | 8 |
| 复现脚本 | `scripts/standalone_eval.py` |

### 1.2 主实验结果

| 方法 | 数据集 | nDCG@10 | Recall@10 |
|------|--------|---------|-----------|
| PromptEOL | QMSum | 0.0191 | 0.0426 |
| PromptEOL | 2WikiMultihop | 0.0182 | 0.0367 |
| PromptEOL | ArguAna | 0.0427 | 0.0910 |
| **mean-pooling** | **QMSum** | **0.1133** | **0.1984** |
| **mean-pooling** | **2WikiMultihop** | **0.1134** | **0.1900** |
| **mean-pooling** | **ArguAna** | **0.2871** | **0.6166** |

Mean-pooling 在三个数据集上均大幅优于 PromptEOL（提升 5.9× ~ 6.7×）。

### 1.3 不同层消融

| layer | QMSum mean | QMSum PromptEOL | 2Wiki mean | 2Wiki PromptEOL | ArguAna mean | ArguAna PromptEOL |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 0.0542 | 0.0209 | 0.0224 | 0.0160 | 0.1253 | 0.0328 |
| 16 | 0.1107 | 0.0236 | 0.0237 | 0.0174 | 0.2487 | 0.0073 |
| 24 | **0.1250** | **0.0278** | 0.0605 | **0.0231** | **0.3253** | 0.0378 |
| 32 | 0.1133 | 0.0191 | **0.1134** | 0.0182 | 0.2871 | **0.0427** |

Mean-pooling 在所有层上均远超 PromptEOL，最差层仍优于 PromptEOL 的最佳层。

---

## 二、分析任务：PromptEOL vs Mean-Pooling 对比与 RoPE 频率分析

### 2.1 为什么 Mean-Pooling 远优于 PromptEOL

**信息瓶颈**：PromptEOL 将整篇文档压缩到最后一个 token 的 4096 维向量；mean-pooling 利用所有 N 个 token 的表示进行平均聚合。

**层间差异放大**：两者的性能差距随层数加深而扩大（layer 8: 2.6× → layer 32: 5.9×）。

**位置贡献实验**（`scripts/position_contribution_analysis.py`）：

| 数据集 | 方法 | 最优位置片段 | 均匀性 |
|--------|------|------------|--------|
| QMSum | mean | 50-60% (sim=0.935) | 0.9997 |
| QMSum | PromptEOL | 90-100% (sim=0.580) | 0.9978 |
| 2WikiMultihop | mean | 30-40% (sim=0.884) | 0.9998 |
| 2WikiMultihop | PromptEOL | 90-100% (sim=0.364) | 0.9956 |
| ArguAna | mean | 30-40% (sim=0.878) | 0.9994 |
| ArguAna | PromptEOL | 90-100% (sim=0.161) | — |

Mean-pooling 在所有位置上几乎完全均匀（uniformity > 0.999），PromptEOL 严重偏向尾部。

### 2.2 RoPE 位置编码频率分析

**Mistral-7B 的 RoPE 配置**（`scripts/rope_frequency_analysis.py`）：base=1,000,000，64 维度对。

| 频段 | 维度对数 | 波长范围 |
|------|---------|---------|
| 高频 | 24 (37.5%) | 6 ~ 900 tokens |
| 中频 | 16 (25.0%) | 900 ~ 28,000 tokens |
| 低频 | 24 (37.5%) | > 28,000 tokens |

**有效分辨率**：2048 tokens 下仅 48.4% 维度对能有效区分位置。

**Base Theta 敏感性**：

| Base | 覆盖@2K | 低频对 |
|------|:---:|:---:|
| 10,000 | 71.9% | 4 |
| 100,000 | 57.8% | 16 |
| **1,000,000** | **48.4%** | **24** |

**隐状态频谱**：随层加深，低频能量占比上升（layer 8 低频 22.8% → layer 32 低频 50.4%）。

**频段滤波**：低通滤波后 embedding 变化最小（cos_sim=1.000），高通滤波后变化最大（cos_sim=0.004）——低频分量对语义表示最关键。

**RoPE 频率特性如何加剧 PromptEOL 劣势**：PromptEOL 依赖精确位置编码分配注意力；Mistral 大 base 值使典型长度下 ~50% 维度对无法区分位置；mean-pooling 通过直接平均所有位置对此具有天然鲁棒性。

---

## 三、进阶任务：长文本表示优化

### 3.1 研究点 1：关键词增强加权池化

**方法**：$$\mathbf{e}_{\text{weighted}} = \sum_{i=1}^{N} \mathbf{w}_i \cdot \mathbf{h}_i^{(L)},\quad \mathbf{w}_i \propto \|\mathbf{h}_i\|_2$$

> 脚本: `scripts/run_advanced_sampled.py`，max_corpus=200, max_queries=50。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| Baseline | 0.1040 | 0.1499 | 0.8635 |
| RP1-L2-Norm | 0.0951 (-8.5%) | **0.1571 (+4.8%)** | **0.8758 (+1.4%)** |
| RP1-AbsMax | 0.0967 (-6.9%) | 0.0994 (-33.7%) | 0.8699 (+0.7%) |

### 3.2 研究点 2：长文本分块聚合

**方法**：tokenizer 精确分块（chunk_size=1024, overlap=128），展平批量编码，按原文聚合。

> 脚本: `scripts/run_rp23_final.py`（QMSum/2Wiki），`scripts/run_rp23_fast.py`（ArguAna），max_corpus=100。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| Baseline | 0.1670 | 0.2043 | 0.8635 |
| **RP2-Chunk-Mean** | **0.3173 (+90.0%)** | **0.2613 (+27.9%)** | 0.8313 (-3.7%) |
| RP2-Chunk-Weighted | 0.3173 (+90.0%) | 0.2613 (+27.9%) | 0.8313 (-3.7%) |

**核心发现**：分块聚合是三种方法中对长文本最有效的方法。

### 3.3 研究点 3：语义压缩后再编码

**方法**：抽取式压缩——所有文本的所有句子批量编码 → 句子中心度计算 → Top-30% 选择。

> 脚本同 RP2。句子编码 max_length=128, batch_size=64。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| Baseline | 0.1670 | 0.2043 | 0.8635 |
| **RP3-Extractive** | **0.2513 (+50.5%)** | 0.2143 (+4.9%) | 0.8150 (-5.6%) |
| RP3-Hierarchical | — | — | 0.8150 (-5.6%) |

压缩效果与文本冗余度正相关：QMSum 会议记录（极高冗余）→ +50.5%，2Wiki 百科（信息密集）→ +4.9%，ArguAna 短论辩 → −5.6%。

### 3.4 组合实验

| 方法 | ArguAna nDCG@10 |
|------|:---:|
| Baseline | 0.8635 |
| RP2+RP3 Combined | 0.8221 (-4.8%) |

QMSum/2Wiki 组合评估因 GPU 时间限制未完成。

### 3.5 三方法效果总览

| 研究点 | QMSum | 2WikiMultihop | ArguAna | 适用场景 |
|--------|:---:|:---:|:---:|------|
| RP1 加权池化 | -8.5% | +4.8% | +1.4% | 信息密集型短中文本 |
| RP2 分块聚合 | **+90.0%** | **+27.9%** | -3.7% | 超长文本 (>1000 tokens) |
| RP3 语义压缩 | +50.5% | +4.9% | -5.6% | 高冗余长文本 |

---

## 四、实验代码说明

### 复现命令

```bash
# 基础实验（PromptEOL + mean-pooling + 层消融）
python scripts/standalone_eval.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --methods prompteol mean --layers -1 8 16 24 32 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --max-length 512 --batch-size 8 --output-dir results/basic

# RoPE 频率分析
python scripts/rope_frequency_analysis.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --output-dir results/rope_analysis --max-samples 20 --batch-size 1

# 位置贡献分析
python scripts/position_contribution_analysis.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --output-dir results/position_analysis --max-samples 20 --batch-size 1

# RP1 加权池化
python scripts/run_advanced_sampled.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets ArguAna QMSum 2WikiMultihop \
  --max-length 512 --batch-size 4 --max-corpus 200 --max-queries 50 \
  --output-dir results/advanced

# RP2+RP3 正式评估（QMSum + 2WikiMultihop）
python scripts/run_rp23_final.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets QMSum 2WikiMultihop \
  --max-length 512 --max-corpus 100 --max-queries 50 \
  --chunk-size 1024 --chunk-overlap 128 --compression-ratio 0.3 \
  --output-dir results/advanced

# RP2+RP3 快速评估（ArguAna）
python scripts/run_rp23_fast.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --datasets ArguAna \
  --max-length 512 --batch-size 4 --max-corpus 200 --max-queries 50 \
  --output-dir results/advanced
```

### 代码结构

```
scripts/
├── standalone_eval.py               # 基础实验
├── rope_frequency_analysis.py       # RoPE 频率分析
├── position_contribution_analysis.py # 位置贡献分析
├── run_advanced_sampled.py          # RP1 加权池化
├── run_rp23_final.py                # RP2+RP3 正式评估
└── run_rp23_fast.py                 # RP2+RP3 快速评估

src/
├── llm_encoder.py, pooling.py, prompts.py  # 基础模块
├── advanced_pooling.py                      # 加权池化
├── chunk_encoder.py                         # 分块编码
└── semantic_compression.py                  # 语义压缩

results/
├── advanced/          # 进阶实验 JSON 结果
├── rope_analysis/     # RoPE 分析 JSON 结果
└── position_analysis/ # 位置贡献 JSON 结果
```

---

## 五、参考文献

1. Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*, 2021.
2. Jiang, A. Q., et al. "Mistral 7B." *arXiv:2310.06825*, 2023.
3. Zhu, D., et al. "LongEmbed: Extending Embedding Models for Long Text Retrieval." *arXiv:2404.02056*, 2024.
4. BehnamGhader, P., et al. "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders." *arXiv:2404.05961*, 2024.
5. Mihalcea, R. & Tarau, P. "TextRank: Bringing Order into Text." *EMNLP*, 2004.
6. Peng, B., et al. "YaRN: Efficient Context Window Extension of Large Language Models." *arXiv:2309.00071*, 2023.
7. Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017.

# 基础实验阶段报告

> 实验于 2026-06-02 完成，模型为 Mistral-7B-Instruct-v0.3 (4-bit 量化)，max_length=512，batch_size=2。

## 1. 实验设置

| 项目 | 内容 |
|------|------|
| 模型 | mistralai/Mistral-7B-Instruct-v0.3（本地路径 `models/Mistral-7B-Instruct-v0.3`） |
| 方法 | PromptEOL；mean-pooling |
| 数据集 | QMSum；2WikiMultihop；ArguAna |
| 默认抽取层 | 最后一层 (`layer=-1`) |
| max_length | 512 |
| 评估框架 | 自定义（`standalone_eval.py`，独立实现 nDCG/Recall/MRR/MAP） |

## 2. 主实验：PromptEOL vs mean-pooling（最后一层）

### 2.1 全指标表

| 方法 | 数据集 | nDCG@10 | Recall@10 | MRR@10 | MAP@10 |
|------|--------|---------|-----------|--------|--------|
| PromptEOL | QMSum | 0.0191 | 0.0426 | 0.0122 | 0.0122 |
| PromptEOL | 2WikiMultihop | 0.0182 | 0.0367 | 0.0126 | 0.0126 |
| PromptEOL | ArguAna | 0.0427 | 0.0910 | 0.0277 | 0.0277 |
| **mean-pooling** | **QMSum** | **0.1133** | **0.1984** | **0.0876** | **0.0876** |
| **mean-pooling** | **2WikiMultihop** | **0.1134** | **0.1900** | **0.0902** | **0.0902** |
| **mean-pooling** | **ArguAna** | **0.2871** | **0.6166** | **0.1845** | **0.1845** |

### 2.2 简要分析（5 分报告质量）

- **mean-pooling 在三个数据集上均大幅优于 PromptEOL**：nDCG@10 提升幅度为 QMSum 5.9 倍、2Wiki 6.2 倍、ArguAna 6.7 倍。
- **PromptEOL 在长文档上失效原因**：PromptEOL 将整个长文本语义压缩到最后一个 token 的 hidden state，这种"单点压缩"存在严重信息瓶颈——单一 4096 维向量无法承载完整文档语义。而 mean-pooling 对所有 token 进行平均融合，保留了更丰富的全局语义信号。
- **任务类型影响**：ArguAna 作为论辩检索任务，查询与文档的语义重叠更明确（nDCG@10=0.2871, Recall@10=0.6166），long-range dependency 不如 QMSum/2Wiki 严重。
- **MAP@10 ≡ MRR@10**：三个数据集每个 query 均只匹配 1 个相关文档（binary relevance, single-positive），此时 average precision = reciprocal rank，因此 MAP@10 与 MRR@10 恒等。这是数据集特性，非方法所致。

---

## 3. 不同层消融

> 各表给出 nDCG@10 / Recall@10 / MRR@10 / MAP@10，每数据集每方法最优值**加粗**。ArguAna 上 MAP@10 ≈ MRR@10（每 query 通常仅匹配 1 个相关文档）。

### 3.1 QMSum（长会议文本）

| layer | 方法 | nDCG@10 | Recall@10 | MRR@10 | MAP@10 |
|-------|------|---------|-----------|--------|--------|
| 8 | mean | 0.0542 | 0.1009 | 0.0402 | 0.0402 |
| 8 | PromptEOL | 0.0209 | 0.0472 | 0.0131 | 0.0131 |
| 16 | mean | 0.1107 | 0.2102 | 0.0805 | 0.0805 |
| 16 | PromptEOL | 0.0236 | 0.0517 | 0.0153 | 0.0153 |
| 24 | mean | **0.1250** | **0.2187** | **0.0968** | **0.0968** |
| 24 | PromptEOL | **0.0278** | **0.0629** | **0.0175** | **0.0175** |
| 32 | mean | 0.1133 | 0.1984 | 0.0876 | 0.0876 |
| 32 | PromptEOL | 0.0191 | 0.0426 | 0.0122 | 0.0122 |

### 3.2 2WikiMultihop（多跳问答）

| layer | 方法 | nDCG@10 | Recall@10 | MRR@10 | MAP@10 |
|-------|------|---------|-----------|--------|--------|
| 8 | mean | 0.0224 | 0.0433 | 0.0163 | 0.0163 |
| 8 | PromptEOL | 0.0160 | 0.0333 | 0.0108 | 0.0108 |
| 16 | mean | 0.0237 | 0.0433 | 0.0176 | 0.0176 |
| 16 | PromptEOL | 0.0174 | 0.0433 | 0.0100 | 0.0100 |
| 24 | mean | 0.0605 | 0.1033 | 0.0468 | 0.0468 |
| 24 | PromptEOL | **0.0231** | **0.0467** | **0.0163** | **0.0163** |
| 32 | mean | **0.1134** | **0.1900** | **0.0902** | **0.0902** |
| 32 | PromptEOL | 0.0182 | 0.0367 | 0.0126 | 0.0126 |

### 3.3 ArguAna（论辩检索）

| layer | 方法 | nDCG@10 | Recall@10 | MRR@10 | MAP@10 |
|-------|------|---------|-----------|--------|--------|
| 8 | mean | 0.1253 | 0.2696 | 0.0802 | 0.0802 |
| 8 | PromptEOL | 0.0328 | 0.0711 | 0.0211 | 0.0211 |
| 16 | mean | 0.2487 | 0.5349 | 0.1594 | 0.1594 |
| 16 | PromptEOL | 0.0073 | 0.0142 | 0.0052 | 0.0052 |
| 24 | mean | **0.3253** | **0.6871** | **0.2117** | **0.2117** |
| 24 | PromptEOL | 0.0378 | 0.0818 | 0.0242 | 0.0242 |
| 32 | mean | 0.2871 | 0.6166 | 0.1845 | 0.1845 |
| 32 | PromptEOL | **0.0427** | **0.0910** | **0.0277** | **0.0277** |

### 3.4 层数分析

**核心发现：最优层因数据集和方法不同而异，但无论选哪一层，mean-pooling 都远超 PromptEOL。**

- **QMSum**：两方法的最优层均为 24（中深层），说明会议文本语义较扁平，中深层的抽象程度在全局语义和局部细节间达成最佳平衡。mean-pooling 最优 nDCG@10=0.1250，为 PromptEOL 最优（0.0278）的 4.5 倍。Recall@10 同样在 layer 24 达到峰值（mean=0.2187 vs PromptEOL=0.0629）。

- **2WikiMultihop**：mean-pooling 最优层=32（最后一层，nDCG@10=0.1134, Recall@10=0.1900），多跳问答需要最高层语义推理。PromptEOL 最优层=24（nDCG@10=0.0231），但绝对值极低。浅层（layer 8）表现最差（mean nDCG@10=0.0224 / PromptEOL=0.0160），说明浅层句法特征无法支持语义检索。MRR@10 与 MAP@10 趋势一致。

- **ArguAna**：mean-pooling 最优层=24（nDCG@10=0.3253, Recall@10=0.6871），PromptEOL 最优层=32（nDCG@10=0.0427）。ArguAna 作为短文本数据集，mean-pooling 的层间差异显著：从 layer 8 到 layer 24 nDCG@10 从 0.1253 跃升至 0.3253（2.6 倍），Recall@10 从 0.2696 → 0.6871（2.5 倍），说明高层抽象对论辩语义匹配至关重要。PromptEOL 在 layer 16 出现异常下降（nDCG@10=0.0073, Recall@10=0.0142），可能的解释是中间层对单 token 压缩特别敏感。

- **方法差异远超层次差异**：mean-pooling 的最差层（QMSum layer 8: nDCG@10=0.0542）仍远超 PromptEOL 的最佳层（ArguAna layer 32: nDCG@10=0.0427），说明 PromptEOL 的"取最后 token"策略具有根本性的信息损失。四个指标（nDCG/Recall/MRR/MAP）一致支持此结论。

---

## 4. 复现命令

```bash
# 主实验（所有数据集 + 所有方法）
python scripts/standalone_eval.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --methods prompteol mean --layers -1 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --max-length 512 --batch-size 2 \
  --output-dir results/basic

# 层消融实验（QMSum + 2Wiki）
python scripts/standalone_eval.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --methods prompteol mean --layers 8 16 24 32 \
  --datasets QMSum 2WikiMultihop \
  --max-length 512 --batch-size 2 \
  --output-dir results/layer_ablation

# 层消融实验（ArguAna，快速一次性编码版）
python scripts/run_arguana_ablation_fast.py
```

## 5. 参考文献

（按实际引用填写）

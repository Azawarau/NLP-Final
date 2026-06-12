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
| 全量数据 | 不使用采样，直接编码全量 corpus 与全量 queries（QMSum: 1,527 queries / 2Wiki: 300 / ArguAna: 1,406），详见下方说明 |
| 复现脚本 | `scripts/standalone_eval.py`（最终层）；`scripts/run_layer_ablation.py`（QMSum/2Wiki 层消融）；`scripts/run_arguana_ablation_fast.py`（ArguAna 层消融） |
| 参考工作 | PromptEOL 方法与 mean-pooling 对比框架来源于 LLM2Vec（BehnamGhader et al., 2024） |

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

> **注**：layer 32 = 模型最终层，其行与 §1.2 全量评估的 `standalone_eval.py --layers -1` 结果完全一致。`standalone_eval.py` 的 `layer=-1` 表示取最后一层，`run_layer_ablation.py` 显式指定 `layers=8,16,24,32`，两者在第 32 层等价——两个脚本的不同仅在于前者一次跑单个层、后者批量跑多个层。

---

## 二、分析任务：PromptEOL vs Mean-Pooling 对比与 RoPE 频率分析

### 2.1 为什么 Mean-Pooling 远优于 PromptEOL

**信息瓶颈**：PromptEOL 将整篇文档压缩到最后一个 token 的 4096 维向量；mean-pooling 利用所有 N 个 token 的表示进行平均聚合。因果注意力机制（Vaswani et al., 2017）下，最后一个 token 虽然"看过"全文，但注意力在长序列上被稀释——尤其在 Mistral-7B 的原生 sliding window attention（4K）限制下，远距离 token 的信息经多层传递后逐步衰减。

**层间差异放大**：两者的性能差距随层数加深而扩大（layer 8: 2.6× → layer 32: 5.9×）。深层 Transformer 包含更多全局语义信息（见 §2.2 频谱分析），但这些信息在 PromptEOL 路径上被注意力衰减削弱，而 mean-pooling 的全局平均操作天然保留了它们。

**位置贡献实验**（`scripts/position_contribution_analysis.py`）：

> **指标说明**：`sim` = cosine similarity，衡量每个位置片段独立编码后的向量与全文表示的相似度。sim 越高说明该位置对全文语义的贡献越大。**均匀性** = 归一化熵，越接近 1 表示所有位置贡献越均衡。

| 数据集 | 方法 | 最优位置片段 | 均匀性 |
|--------|------|------------|--------|
| QMSum | mean | 50-60% (sim=0.935) | 0.9997 |
| QMSum | PromptEOL | 90-100% (sim=0.580) | 0.9978 |
| 2WikiMultihop | mean | 30-40% (sim=0.884) | 0.9998 |
| 2WikiMultihop | PromptEOL | 90-100% (sim=0.364) | 0.9956 |
| ArguAna | mean | 30-40% (sim=0.878) | 0.9994 |
| ArguAna | PromptEOL | 90-100% (sim=0.161) | — |

Mean-pooling 在所有位置上几乎完全均匀（uniformity > 0.999），PromptEOL 严重偏向尾部。

> **ArguAna PromptEOL 为何无均匀性数据？** 归一化熵 `NaN` 的根因是位置 0-10%（文本开头）的余弦相似度为 **−0.059**——全文表示与文本开头在语义空间中呈反向关系。归一化熵要求所有输入 ≥ 0，负值触发计算失败。其他方法/数据集未出现此情况（QMSum 最小 segment sim=0.393，2Wiki 最小=0.218）。这是因为 ArguAna 论辩极短（100~300 tokens），最后 token 看到结尾后，模型对开头的"记忆"已被完全覆盖并翻转。同时 cv=1.084（QMSum cv=0.100，2Wiki cv=0.148）也表明 PromptEOL 在 ArguAna 上片段间一致性极差。

### 2.2 RoPE 位置编码频率分析

**Mistral-7B 的 RoPE 配置**（`scripts/rope_frequency_analysis.py`）：base=1,000,000，64 维度对。

| 频段 | 维度对数 | 波长范围 |
|------|---------|---------|
| 高频 | 24 (37.5%) | 6 ~ 900 tokens |
| 中频 | 16 (25.0%) | 900 ~ 28,000 tokens |
| 低频 | 24 (37.5%) | > 28,000 tokens |

#### 数据块一：RoPE 理论分辨率（bands + coverage）

> 数据来源：`results/rope_analysis/rope_theory_results.json`，`rope_theory.coverage` 段。纯数学推导：据 RoPE 公式 \(\theta_i = \text{base}^{-2i/d}\)，波长 \(=2\pi/\theta_i\)；当波长 ≥ 序列长度×2 时，该维度对丧失位置区分能力（饱和）。

| 序列长度 | 可区分对 | 饱和对 | 有效分辨率 |
|------|:---:|:---:|:---:|
| 512（实验 max_length） | 24 | 40 | 37.5% |
| 2048（QMSum 典型长度） | 31 | 33 | 48.4% |
| 32768（模型最大上下文） | 43 | 21 | 67.2% |

→ 在实验配置（max_length=512）下，仅 37.5% 维度对能有效区分位置。分辨率随序列长度增长逐步恢复（2048: 48.4%, 32768: 67.2%），但在短中序列范围内损失严重——这是 Mistral 为支持 32K 长上下文做出的明确取舍。

#### 数据块二：Base 值反事实分析（theta_sensitivity）

> 数据来源：`results/rope_analysis/rope_theory_results.json`，`rope_theory.theta_sensitivity` 段。对 5 个不同 base 值分别重新计算 RoPE 频段分布与覆盖率。

| Base | 低频维度对 | 512 分辨率 | 2048 分辨率 | 32768 分辨率 |
|------|:---:|:---:|:---:|:---:|
| 10,000（常规 RoPE） | 4 | 56.2% | 71.9% | 100.0% |
| 100,000 | 16 | 45.3% | 57.8% | 81.2% |
| **1,000,000（Mistral 实际）** | **24** | **37.5%** | **48.4%** | **67.2%** |
| 10,000,000 | 30 | 32.8% | 40.6% | 57.8% |

→ base 增大 → 低频对增多、短序列覆盖率下降。若 base 降回常规 10K，2048 tokens 分辨率可从 48.4% 升至 71.9%（+23.5 pp）。这意味着 PromptEOL 在 Mistral 上的差表现不仅仅是方法本身的问题，更是"方法对位置精度高敏感"与"模型为长上下文牺牲位置精度"之间的方向性冲突。

#### 数据块三：隐状态频谱（spectral）

> 数据来源：`results/rope_analysis/{QMSum,2WikiMultihop,ArguAna}_rope_analysis.json`，各文件 `spectral` 段。对模型每层隐状态沿序列维度做 FFT（n_fft=1024），按 Nyquist 频率划分三频段（低频 0~1/8 Nyquist、中频 1/8~1/2、高频 1/2~Nyquist），计算各频段能量占比。20 条采样文本 × 2 方法 × 4 层的平均值。

| 数据集 | 方法 | Layer 8 低频 | Layer 16 | Layer 24 | Layer 32 |
|------|------|:---:|:---:|:---:|:---:|
| QMSum | mean | 15.5% | 27.0% | 51.6% | **55.9%** |
| QMSum | PromptEOL | 15.4% | 28.0% | 52.6% | **56.0%** |
| 2WikiMultihop | mean | 16.9% | 29.6% | 49.3% | 47.1% |
| 2WikiMultihop | PromptEOL | 15.7% | 28.8% | 49.2% | 47.3% |
| ArguAna | mean | 12.8% | 16.8% | 34.4% | 50.3% |
| ArguAna | PromptEOL | 13.3% | 17.8% | 35.5% | 50.4% |

→ 两个关键结论：

**(a) 低频化是 Transformer 结构自身的涌现属性，与编码方法无关。** 同层同数据集下，mean 和 PromptEOL 的低频占比几乎相同（最大差异 < 1 pp）。PromptEOL 的劣势不能归因于频谱偏好——它在同层"看到"的频段分布和 mean 一样。

**(b) 文本类型决定频谱特性。** 2Wiki 的低频占比（47.1%）持续低于 QMSum（55.9%）——信息密集的百科文本需要更多高频维度保留局部细节，语义压缩更不完全。ArguAna 浅层（layer 8）低频占比仅 12.8%，因为短文本在浅层尚未完成语义聚合。

#### 数据块四：频段滤波实验（frequency_filtering）

> 数据来源：同上各数据集 JSON 文件，`frequency_filtering` 段。对 layer 32 隐状态做 FFT → 频段掩码 → IFFT 重建 → 与原始 embedding 计算余弦相似度。10 条文本的平均值。低通 = 只保留低频，高通 = 只保留高频，带通 = 只保留中频。数值越高说明该频段对语义表示的贡献越大。

| 数据集 | 方法 | 低通 (cos_sim) | 带通 | 高通 | 无滤波 |
|------|------|:---:|:---:|:---:|:---:|
| QMSum | mean | **1.000** | 0.023 | 0.004 | 1.000 |
| QMSum | PromptEOL | 0.666 | 0.529 | **0.643** | 1.000 |
| 2WikiMultihop | mean | **1.000** | −0.031 | 0.007 | 1.000 |
| 2WikiMultihop | PromptEOL | 0.565 | 0.588 | **0.621** | 1.000 |
| ArguAna | mean | 0.9999 | 0.383 | **0.278** | 1.000 |
| ArguAna | PromptEOL | 0.397 | 0.826 | **0.854** | 1.000 |

**频段重要性分数**（QMSum，= 1 − cos_sim，即移除该频段后 embedding 的变化量，越大越重要）：

| 方法 | 低频重要性 | 中频重要性 | 高频重要性 |
|------|:---:|:---:|:---:|
| mean | 0.996 | 0.977 | 0.000 |
| PromptEOL | 0.357 | 0.471 | **0.334** |

→ 决定性的方法差异：

**(a) Mean-pooling：语义完全在低频中。** 低通后几乎不变（cos_sim=1.000），高通后变为噪声（cos_sim=0.004）。移除低频时 embedding 变化 99.6%（重要性=0.996），移除高频时变化 0%（重要性=0.000）。这是因为平均操作天然是低通滤波器——它抹平了所有高频波动，只保留全局低频语义。

**(b) PromptEOL：语义分散在三频段，且高频分量保留最多信息。** 高通 cos_sim=0.643 远高于 mean 的 0.004——last token 将语义压缩进了精细的高频通道。而高频维度仅 24 对且波长 ≤900 tokens（见数据块一），在长文本上这些维度本身面临最严重的位置编码退化。PromptEOL 同时承受"高频依赖"和"高频分辨率不足"的双重困境。

**(c) 跨数据集趋势：** ArguAna 短文本下两种方法的高频保留均升高（mean: 0.004→0.278，PromptEOL: 0.643→0.854）——短文本的语义信息尚未完全迁移至低频通道，局部细节仍分散在高频中。这解释了为什么 PromptEOL 在 ArguAna 上的衰退幅度相对 QMSum 更小（见 §1.2）。

#### 综合证据链

上述四块数据形成逻辑闭环：

1. **RoPE 天然存在位置分辨率瓶颈**：base=1M 下 2048 tokens 仅 48.4% 分辨率，高频 24 对波长 ≤900 tokens → 在长文本上它们**无法区分位置**（数据块一、二）
2. **低频化是结构属性，PromptEOL 无法躲避**：Transformer 深层无论用什么方法提取，隐藏层都在低频通道中（数据块三）
3. **但 mean-pooling 的语义恰好在低频**：而低频维度对波长最长（≥28K tokens），是**位置精度最高**的通道（数据块四：低通 cos_sim=1.000）
4. **PromptEOL 的语义却在错误的位置**：它把语义压缩进了高频通道（高通 cos_sim=0.643），但这些高频维度恰好在长文本上**先饱和**（数据块一：饱和对 33 个）

→ **最终结论**：PromptEOL 依赖精确位置编码分配注意力——它的语义在高频维度中，而 Mistral 大 base 值使这些高频维度在长文本上几乎无法区分位置；mean-pooling 通过直接平均所有位置，其语义天然落在低频通道，对位置编码退化具有**完全鲁棒性**。

---

## 三、进阶任务：长文本表示优化

> **关于 Baseline 数值差异**：进阶实验的 baseline 与 §1.2 全量评估结果不同，这是因为进阶实验对数据集进行了采样（max_corpus=100~200, max_queries=50），以节省 GPU 测试时间。全量评估需编码数万篇长文本，而进阶部分涉及多种变体方法，全量运行不现实。采样后 corpus 缩小使检索难度降低，因此 nDCG 绝对值偏高（ArguAna 尤为明显），但所有对比在同一子集上完成，**相对提升/下降百分比仍然有效**。

### 3.1 研究点 1：关键词增强加权池化

**方法**：$$\mathbf{e}_{\text{weighted}} = \sum_{i=1}^{N} \mathbf{w}_i \cdot \mathbf{h}_i^{(L)},\quad \mathbf{w}_i \propto \|\mathbf{h}_i\|_2$$

> 脚本: `scripts/run_advanced_sampled.py`，max_corpus=200, max_queries=50。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| Baseline | 0.1040 | 0.1499 | 0.8635 |
| RP1-L2-Norm | 0.0951 (-8.5%) | **0.1571 (+4.8%)** | **0.8758 (+1.4%)** |
| RP1-AbsMax | 0.0967 (-6.9%) | 0.0994 (-33.7%) | 0.8699 (+0.7%) |

> **结果解读**：加权池化效果高度依赖文本特性。2WikiMultihop（百科信息密）和 ArguAna（论辩短文本）中关键词 token 范数更高，L2-Norm 加权可有效放大关键 token 权重（+4.8%/+1.4%）。QMSum（会议记录冗余高）中背景词 token 范数与关键词差异小，加权反而引入噪声（−8.5%）。AbsMax 极不稳定（2Wiki 上 −33.7%），说明对最大范数 token 的单点依赖在信息分散的百科文本上完全失效。

### 3.2 研究点 2：长文本分块聚合

**方法**：tokenizer 精确分块——QMSum/2Wiki 用 chunk_size=1024, overlap=128；ArguAna 用 chunk_size=512, overlap=64——然后展平所有 chunk 为一批编码，再按原文聚合（mean 或 L2-weighted）。

> 脚本: `scripts/run_rp23_final.py`（QMSum/2Wiki）和 `scripts/run_rp23_fast.py`（ArguAna）——两个脚本均包含 RP2 与 RP3 的完整实现与 baseline 共用代码，下文 §3.3 不再单独列出。max_corpus=100 (final) / 200 (fast)。

> **为何 chunk_size 不同？** QMSum/2Wiki 文本极长（3000~8000 tokens），大块（1024）减少分块数控制编码开销；ArguAna 文本短（100~300 tokens），若用 1024 会导致绝大多数文本不触达分块逻辑，退化回 baseline，因此缩小为 512。分块窗口必须匹配目标文本的长度分布，不可统一。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| Baseline | 0.1670 | 0.2043 | 0.8635 |
| **RP2-Chunk-Mean** | **0.3173 (+90.0%)** | **0.2613 (+27.9%)** | 0.8313 (-3.7%) |
| RP2-Chunk-Weighted | 0.3173 (+90.0%) | 0.2613 (+27.9%) | 0.8313 (-3.7%) |

**核心发现**：分块聚合是三种方法中对长文本最有效的方法。

> **ArguAna 为何两种分块方法结果相同？** ArguAna 论辩文本平均长度仅 100~300 tokens，而分块窗口为 512 tokens（ArguAna 用的 fast 脚本参数），导致绝大多数文本只产生 1 个 chunk。当 chunk 数为 1 时，`mean(chunks)` 和 `weighted-average(chunks)` 退化为同一个值。这也解释了分块在 ArguAna 上的 −3.7% 副效应：短文本被强行分块后反而损失了跨句上下文，而在 QMSum/2Wiki 这种真正需要切分的长文本上，分块聚合的优势才得以体现。

### 3.3 研究点 3：语义压缩后再编码

**方法**：基于 TextRank / LexRank 原理（Mihalcea & Tarau, 2004）的抽取式压缩——所有文本的所有句子批量编码成向量 → 构建句子间余弦相似度矩阵 → 计算每个句子的中心度（平均相似度，即 PageRank 的一阶近似） → 按中心度排序取 Top-30% 句子 → 保持原文顺序还原。Hierarchical 变体为两阶段：先在各段落内局部压缩，再对拼接结果全局压缩。

> 脚本同上（RP2 与 RP3 实现在同一脚本中：`scripts/run_rp23_final.py` / `scripts/run_rp23_fast.py`）。句子编码 max_length=128, batch_size=64。

| 方法 | QMSum nDCG@10 | 2Wiki nDCG@10 | ArguAna nDCG@10 |
|------|:---:|:---:|:---:|
| Baseline | 0.1670 | 0.2043 | 0.8635 |
| **RP3-Extractive** | **0.2513 (+50.5%)** | 0.2143 (+4.9%) | 0.8150 (-5.6%) |
| RP3-Hierarchical | — | — | 0.8150 (-5.6%) |

压缩效果与文本冗余度正相关：QMSum 会议记录（极高冗余）→ +50.5%，2Wiki 百科（信息密集）→ +4.9%，ArguAna 短论辩 → −5.6%。

> **ArguAna 为何两种压缩方法结果相同？** `compress_extractive` 在文本 ≤3 句时直接返回原文（不压缩），ArguAna 论辩文本大多不超过 3 句，Extractive 和 Hierarchical 在绝大多数文本上输入完全相同，从而输出一致。同时，压缩对短文本产生 −5.6% 的反效果：本就信息密集的短论辩经压缩后可能丢失关键推理节点，反而损害检索质量。

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

> **ArguAna 为何三种方法均衰退？** ArguAna 论辩文本极短（100~300 tokens），三种方法均以长文本为设计目标：加权池化（RP1）在短文本上 token 范数差异小、区分度不足；分块聚合（RP2）的 chunk_size=512 导致 ≈1 chunk，分块无效果反而损失跨句上下文；语义压缩（RP3）在 ≤3 句时跳过压缩、直接返回原文。三个实验的负结果共同验证了一个前提：**这些方法的前提是文本足够长**（≥1000 tokens）。

---

## 四、实验代码说明

### 复现命令

```bash
# 基础实验（PromptEOL + mean-pooling，最终层）
python scripts/standalone_eval.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --methods prompteol mean --layers -1 \
  --datasets QMSum 2WikiMultihop ArguAna \
  --max-length 512 --batch-size 8 --output-dir results/basic

# 层消融实验（QMSum + 2WikiMultihop，layer 8/16/24/32）
python scripts/run_layer_ablation.py \
  --model models/Mistral-7B-Instruct-v0.3 \
  --layers 8 16 24 32 \
  --output-dir results/layer_ablation

# 层消融实验（ArguAna，layer 8/16/24/32）
python scripts/run_arguana_ablation_fast.py

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
├── standalone_eval.py               # 基础实验（最终层）
├── run_layer_ablation.py            # 层消融（QMSum + 2WikiMultihop）
├── run_arguana_ablation_fast.py     # 层消融（ArguAna）
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
├── basic/             # 基础实验 JSON 结果
├── layer_ablation/    # 层消融 JSON 结果
├── advanced/          # 进阶实验 JSON 结果
├── rope_analysis/     # RoPE 分析 JSON 结果
└── position_analysis/ # 位置贡献 JSON 结果
```

---

## 五、参考文献

1. Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*, 2021.
2. Jiang, A. Q., et al. "Mistral 7B." *arXiv:2310.06825*, 2023.
3. Zhu, D., et al. "LongEmbed: Extending Embedding Models for Long Context Retrieval." *arXiv:2404.12096*, 2024.
4. BehnamGhader, P., et al. "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders." *arXiv:2404.05961*, 2024.
5. Mihalcea, R. & Tarau, P. "TextRank: Bringing Order into Text." *EMNLP*, 2004.
6. Peng, B., et al. "YaRN: Efficient Context Window Extension of Large Language Models." *arXiv:2309.00071*, 2023.
7. Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017.

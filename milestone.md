# 进展报告（Milestone）

**课题**：从大语言模型中抽取长文本表示   
**小组**：组长 王涵，组员 任奕弛  
**报告日期**：2026 年 6 月 2 日  

---

## 一、对照作业任务的完成情况

### 1.1 简单任务（§3.1，20 分）

| 要求项                            | 状态      | 说明                                                               |
| ------------------------------ | ------- | ---------------------------------------------------------------- |
| 使用 **Mistral-Instruct-7B-0.3** | **已完成** | 模型已下载至 `models/Mistral-7B-Instruct-v0.3/`，4-bit 量化运行             |
| 实现 **PromptEOL**               | **已完成** | 模板：`This sentence : "{text}" means in one word:"`；forward hook   |
| 实现 **mean-pooling**            | **已完成** | 对目标层 token 做 attention mask 加权平均                                 |
| 在三个数据集上 **检索评估**               | **已完成** | 三数据集 × 两方法完整指标表已产出，详见 §4.4                                       |
| **不同 Transformer 层** 的嵌入质量探索   | **已完成** | **两者**（PromptEOL + mean-pooling）均完成 layers 8/16/24/32 消融，详见 §4.5 |

**小结**：§3.1 **全部完成**。mean-pooling 全面领先 PromptEOL（nDCG@10 提升 5.9–6.7 倍）。层消融揭示各数据集最优层不同：**QMSum** 两者最优层=24；**2Wiki** mean-pooling 最优层=32、PromptEOL 最优层=24；**ArguAna** mean-pooling 最优层=24（0.3253，层间差异最显著，layer 8→24 提升 2.6 倍）、PromptEOL 最优层=32（0.0427，layer 16 异常降至 0.0073）。PromptEOL 全局最佳（0.0427 @ ArguAna layer 32）仍低于 mean-pooling 全局最差（0.0542 @ QMSum layer 8），方法差异远超层次差异。

---

### 1.2 分析任务（§3.2，20 分）

| 要求项                                | 状态      | 说明                                         |
| ---------------------------------- | ------- | ------------------------------------------ |
| 对比 PromptEOL 与 mean-pooling，分析优劣原因 | **进行中** | 基础数据已产出，分析写入 `reports/basic_experiment.md` |
| RoPE 频率对结果的影响及机理分析                 | **未开始** | `proposal` 已规划，代码未实现                       |

**小结**：§3.2 的对比分析已完成初稿（基于 §3.1 主实验数据），RoPE 分析待进行。

---

### 1.3 进阶任务（§3.3，60 分）

| 要求项                  | 状态          | 说明                                   |
| -------------------- | ----------- | ------------------------------------ |
| 至少 **3 个** 可提升性能的研究点 | **未开始（实现）** | `proposal` 已选定：关键词加权池化、分块聚合、语义压缩后再编码 |
| 研究点可组合、有实验表现         | **未开始**     | 需在与基线对比后迭代                           |
| 完整代码（含进阶）            | **部分**      | 当前仓库以基础阶段代码为主                        |

**小结**：§3.3 处于**方案设计阶段**，未进入编码与实验。

---

## 二、当前进度总览

```
基础实验阶段  ██████████████████  100%  （全部完成）
分析实验阶段  ███░░░░░░░░░░░░░    20%  （对比分析已有数据，RoPE 未做）
进阶优化阶段  █░░░░░░░░░░░░░░░     5%  （方向已定，未实现）
```

**已完成工作**：环境搭建、方法实现（含 forward hook OOM 修复）、评估框架、数据与模型下载、主实验三数据集 × 两方法跑分、层消融实验、报告分析。

**尚未完成工作**：RoPE 分析实验、进阶方法及最终 PDF。

---

## 三、实验成果

### 3.1 工程与代码结构

已建立可复现实验仓库，最终方案文件清单如下：

| 层级 | 文件 | 用途 |
|------|------|------|
| 核心库 | `src/llm_encoder.py` | 编码器，forward hook 捕获单层 hidden states（节省 ~65% 显存） |
| 核心库 | `src/prompts.py` | PromptEOL 模板构建 |
| 核心库 | `src/pooling.py` | mean-pooling / last-token-pooling |
| 实验脚本 | `scripts/standalone_eval.py` | 主实验 + QMSum/2Wiki 层消融，手动编码→检索→四指标计算 |
| 实验脚本 | `scripts/run_arguana_ablation_fast.py` | ArguAna 层消融，单次前传提取 4 层 |
| 下载 | `scripts/download_assets.py` | 下载模型 + 三数据集（一次性） |
| 验证 | `scripts/smoke_test.py` | 池化 + Prompt 模板单元测试 |
| 验证 | `scripts/verify_basic.py` | 综合验证，产出 `results/verify_report.json` |
| 验证 | `scripts/verify_offline_encoder.py` | 随机小模型离线编码管线验证 |

技术要点：

- 支持 **可配置抽取层**（`layer=-1` 或指定 1…N 层）。
- **显存优化**：用 forward hook 替代 `output_hidden_states=True`，仅捕获目标层输出，节省 ~1 GB VRAM，使 RTX 4060 8GB 可运行 7B 4-bit 模型。
- **评估方式**：绕过 MTEB 框架，使用独立评估管线，手动实现编码→检索→nDCG/Recall/MRR/MAP 全流程。数据集通过 HuggingFace `datasets` 直接加载：
  - QMSum → `dwzhu/LongEmbed`（`qmsum`）
  - 2WikiMultihop → `dwzhu/LongEmbed`（`2wikimqa`）
  - ArguAna → `mteb/arguana`

### 3.2 数据与模型资源

| 资源                       | 来源                            | 状态                                      |
| ------------------------ | ----------------------------- | --------------------------------------- |
| QMSum                    | `dwzhu/LongEmbed`（`qmsum`）    | 已缓存；corpus 197 条，queries 1527 条         |
| 2WikiMultihop            | `dwzhu/LongEmbed`（`2wikimqa`） | 已缓存；corpus / queries 各 300 条            |
| ArguAna                  | `mteb/arguana`                | 已缓存；corpus 8674 条，queries 1406 条        |
| Mistral-7B-Instruct-v0.3 | Hugging Face                  | 已落盘至 `models/Mistral-7B-Instruct-v0.3/` |

### 3.3 质量验证（非正式实验指标）

已通过自动化检查（`results/verify_report.json`）：

- 项目文件与能力点检查：**PASS**
- 依赖与单元级池化 / Prompt 模板：**PASS**
- 离线编码管线（随机小模型 + MTEB 任务注册）：**PASS**

### 3.4 主实验结果（§3.1，已完成）

**实验配置**：Mistral-7B-Instruct-v0.3，4-bit 量化，max_length=512，batch_size=2，抽取层=最后一层。

| 方法               | QMSum      | 2WikiMultihop | ArguAna    |
| ---------------- | ---------- | ------------- | ---------- |
| PromptEOL        | 0.0191     | 0.0182        | 0.0427     |
| **mean-pooling** | **0.1133** | **0.1134**    | **0.2871** |

**关键发现**：mean-pooling 全面优于 PromptEOL（nDCG@10 提升 5.9–6.7 倍）。

### 3.5 层消融实验结果（§3.1，已完成）

> nDCG@10 / Recall@10 / MRR@10 / MAP@10 ，最优值加粗。

**QMSum**（长会议文本，corpus 197 / queries 1527）：

| layer | 方法        | nDCG@10    | Recall@10  | MRR@10     | MAP@10     |
| ----- | --------- | ---------- | ---------- | ---------- | ---------- |
| 8     | mean      | 0.0542     | 0.1009     | 0.0402     | 0.0402     |
| 8     | PromptEOL | 0.0209     | 0.0472     | 0.0131     | 0.0131     |
| 16    | mean      | 0.1107     | 0.2102     | 0.0805     | 0.0805     |
| 16    | PromptEOL | 0.0236     | 0.0517     | 0.0153     | 0.0153     |
| 24    | mean      | **0.1250** | **0.2187** | **0.0968** | **0.0968** |
| 24    | PromptEOL | **0.0278** | **0.0629** | **0.0175** | **0.0175** |
| 32    | mean      | 0.1133     | 0.1984     | 0.0876     | 0.0876     |
| 32    | PromptEOL | 0.0191     | 0.0426     | 0.0122     | 0.0122     |

**2WikiMultihop**（多跳问答，corpus/queries 各 300）：

| layer | 方法        | nDCG@10    | Recall@10  | MRR@10     | MAP@10     |
| ----- | --------- | ---------- | ---------- | ---------- | ---------- |
| 8     | mean      | 0.0224     | 0.0433     | 0.0163     | 0.0163     |
| 8     | PromptEOL | 0.0160     | 0.0333     | 0.0108     | 0.0108     |
| 16    | mean      | 0.0237     | 0.0433     | 0.0176     | 0.0176     |
| 16    | PromptEOL | 0.0174     | 0.0433     | 0.0100     | 0.0100     |
| 24    | mean      | 0.0605     | 0.1033     | 0.0468     | 0.0468     |
| 24    | PromptEOL | **0.0231** | **0.0467** | **0.0163** | **0.0163** |
| 32    | mean      | **0.1134** | **0.1900** | **0.0902** | **0.0902** |
| 32    | PromptEOL | 0.0182     | 0.0367     | 0.0126     | 0.0126     |

**ArguAna**（论辩检索，corpus 8674 / queries 1406）：

| layer | 方法        | nDCG@10    | Recall@10  | MRR@10     | MAP@10     |
| ----- | --------- | ---------- | ---------- | ---------- | ---------- |
| 8     | mean      | 0.1253     | 0.2696     | 0.0802     | 0.0802     |
| 8     | PromptEOL | 0.0328     | 0.0711     | 0.0211     | 0.0211     |
| 16    | mean      | 0.2487     | 0.5349     | 0.1594     | 0.1594     |
| 16    | PromptEOL | 0.0073     | 0.0142     | 0.0052     | 0.0052     |
| 24    | mean      | **0.3253** | **0.6871** | **0.2117** | **0.2117** |
| 24    | PromptEOL | 0.0378     | 0.0818     | 0.0242     | 0.0242     |
| 32    | mean      | 0.2871     | 0.6166     | 0.1845     | 0.1845     |
| 32    | PromptEOL | **0.0427** | **0.0910** | **0.0277** | **0.0277** |

**关键发现**：

**（1）方法差异远超层次差异（跨数据集一致）**
无论选择哪一层，mean-pooling 的四项指标均远超 PromptEOL：mean-pooling 全局最差（QMSum layer 8: nDCG@10=0.0542, Recall@10=0.1009）仍 > PromptEOL 全局最佳（ArguAna layer 32: nDCG@10=0.0427, Recall@10=0.0910）。PromptEOL 的"取最后 token"策略存在根本性信息瓶颈。

**（2）最优层因数据集与方法而异**

- **QMSum**（长会议文本）：两方法最优层均为 24（中深层）。会议文本语义扁平，中深层在全局语义与局部细节间达到最佳平衡。mean-pooling 最优 nDCG@10=0.1250，为 PromptEOL 最优（0.0278）的 4.5 倍。
- **2WikiMultihop**（多跳问答）：mean-pooling 最优层=32（最后一层，nDCG@10=0.1134），多跳推理需要最深层的语义抽象与推理能力。PromptEOL 最优层=24（nDCG@10=0.0231），但绝对值极低。浅层（layer 8）两方法均表现最差，说明浅层句法特征无法支撑语义检索。
- **ArguAna**（短文本论辩检索）：mean-pooling 最优层=24（nDCG@10=0.3253, Recall@10=0.6871），层间差异三数据集最大（layer 8→24 nDCG 提升 2.6 倍，Recall 提升 2.5 倍），高层语义抽象对论辩匹配至关重要。PromptEOL 最优层=32（nDCG@10=0.0427），但 layer 16 异常崩塌（nDCG@10=0.0073, 仅为 layer 24 的 1/5），推测中间层对单 token 压缩特别敏感。

**（3）Recall@10 与 nDCG@10 变化一致，MRR@10 ≡ MAP@10 因数据集特性**
三项数据集每个 query 均仅匹配 1 个相关文档（binary relevance, single-positive），此时 average precision ≡ reciprocal rank，因此 MAP@10 与 MRR@10 恒等。Recall@10 与 nDCG@10 在各消融实验中变化趋势一致（如 ArguAna mean-pooling: layer 24 Recall 达到 0.6871，nDCG 同时达到峰值 0.3253），说明层选择对排序质量和召回覆盖产生同步影响。

---

## 四、下一步计划

### 4.1 近期（§3.2 分析实验）

1. 在基础表基础上完善 **PromptEOL vs mean-pooling** 对比分析。
2. 实现并实验 **RoPE 频率相关** 干预或分析。

### 4.2 中期（§3.3 进阶任务）

按 `proposal` 推进三个研究点（满足「可组合」要求）：

| 研究点 | 方向        | 与基线关系                |
| --- | --------- | -------------------- |
| R1  | 关键词增强加权池化 | 改进 mean-pooling 分支   |
| R2  | 长文本分块二次聚合 | 可与 R1 或 PromptEOL 结合 |
| R3  | 语义压缩后再编码  | 可与分块或加权组合            |

### 4.3 提交前 checklist

- [x] `results/basic/` 完整 JSON  
- [x] `results/layer_ablation/` 完整 JSON
- [ ] 实验报告 PDF（含参考文献）  
- [ ] 进阶代码与消融表  
- [ ] 代码打包与 README 复现说明终稿  

---

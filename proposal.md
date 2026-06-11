选题：从大语言模型中抽取长文本表示
=============

* * *

一、研究目标
======

本项目主要研究以下内容：

1. 复现并比较 PromptEOL 与 mean-pooling 两种长文本表示方法；

2. 探索不同隐藏层对句子嵌入质量的影响；

3. 分析 RoPE 位置编码频率对长文本表示效果的影响；

4. 提出并验证多种长文本表示优化方案；

5. 在 QMSum、2WikiMultihop、ArguAna 数据集上完成实验评估。

* * *

二、研究内容与计划
=========

（1）基础实验阶段
---------

### 研究内容

* 使用 Mistral-Instruct-7B-0.3 模型；

* 实现：
  
  * PromptEOL 方法；
  
  * mean-pooling 方法；

* 比较不同 Transformer layer 的表示效果。

### 实验数据集

* QMSum

* 2WikiMultihop

* ArguAna

### 评估指标

根据数据集任务采用相应检索或相似度评价指标，例如：

* Recall

* MAP

* nDCG

* MRR

* * *

（2）分析实验阶段
---------

### 研究内容

#### ① PromptEOL 与 mean-pooling 对比分析

重点分析：

* 为什么 mean-pooling 在长文本下可能更稳定；

* PromptEOL 的语义压缩优势与信息损失问题；

* 长文本中 token 数量对表示质量的影响。

#### ② RoPE 位置编码研究

探索：

* 不同频率成分对长文本建模的影响；

* 长距离 token 的位置衰减问题；

* 高频/低频信息对语义聚合效果的贡献。

* * *

（3）进阶优化阶段
---------

计划从以下方向中进行优化：

### 研究点1：关键词增强加权池化

核心思想：

* 对重要 token 提高权重；

* 减少无关 token 对平均池化的干扰。

可能方法：

* attention score 加权；

* TF-IDF 权重；

* token saliency。

* * *

### 研究点2：长文本分块聚合（Chunk-based Aggregation）

核心思想：

* 将长文本切分为多个语义块；

* 分别编码后进行二次聚合。

目的：

* 缓解长距离语义衰减；

* 保留局部语义信息。

* * *

### 研究点3：语义压缩后再编码

核心思想：

* 先对长文本进行摘要压缩；

* 再提取文本表示。

可能方法：

* Prompt-based summarization；

* sliding summarization；

* hierarchical compression。

* * * 

三、技术路线
======

整体流程如下：

1. 数据集预处理；

2. 长文本输入模型；

3. 提取 hidden states；

4. 不同 pooling 方法生成 embedding；

5. 相似度计算与检索评估；

6. 分析 layer 与 RoPE 影响；

7. 引入改进策略并进行实验对比。

* * *

四、分组情况
==========

组长：王涵

组员：任奕弛

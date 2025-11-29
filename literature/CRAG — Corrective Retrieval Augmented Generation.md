## CRAG — Corrective Retrieval Augmented Generation

----
Yan S Q, Gu J C, Zhu Y, et al. Corrective Retrieval Augmented Generation[J]. arXiv preprint arXiv:2401.15884, 2024. 
----

### 1. 背景与动机 (Background & Motivation)
尽管大型语言模型（LLMs）表现出色，但由于仅依赖参数化知识，不可避免地会出现幻觉 。检索增强生成（RAG）通过引入外部相关文档来补充知识，是一种有效的解决方案 。然而，RAG 存在以下关键问题：
1.  **严重依赖检索质量**：RAG 的有效性取决于检索文档的相关性和准确性。如果检索器返回无关或误导性文档，可能会加剧模型的事实错误 。
2.  **无差别使用检索结果**：大多数现有 RAG 方法不加区分地合并检索到的文档，无论其是否相关 。
3.  **文档利用粒度粗糙**：现有方法通常将整个文档作为参考，包含大量非必要信息，干扰生成。

**CRAG** 旨在解决检索器返回不准确结果的情况，通过设计**纠正策略（Corrective Strategies）**来提高生成的鲁棒性 。

---

### 2. 核心方法 (Core Methodology)

CRAG 设计了一个即插即用的轻量级**检索评估器（Retrieval Evaluator）**，根据检索质量触发不同的知识检索动作。

#### 2.1 检索评估器 (Retrieval Evaluator)
* **模型**：采用轻量级的 T5-large 模型进行微调，参数量远小于通用 LLM 。
* **功能**：评估查询与检索文档的相关性，并输出一个置信度分数。
* **动作触发**：根据置信度分数设定阈值，触发三种动作之一：**Correct（正确）**、**Incorrect（错误）** 或 **Ambiguous（模糊）** 。

#### 2.2 三种纠正动作 (Action Triggers)

1.  **Correct (正确)**：
    * **触发条件**：至少有一个检索文档的置信度高于上限阈值 。
    * **处理策略**：执行**知识精炼（Knowledge Refinement）**。虽然文档总体相关，但仍可能包含噪声。CRAG 采用“分解-再重组”（decompose-then-recompose）算法，将文档分割为细粒度的知识条带（strips），过滤掉无关条带，仅保留关键信息 。

2.  **Incorrect (错误)**：
    * **触发条件**：所有检索文档的置信度均低于下限阈值。
    * **处理策略**：放弃检索到的文档，转而进行**网络搜索（Web Search）**。使用 ChatGPT 将查询重写为关键词，利用搜索引擎获取更广泛的外部知识 。

3.  **Ambiguous (模糊)**：
    * **触发条件**：置信度处于中间区间。
    * **处理策略**：结合 **Correct** 和 **Incorrect** 的处理方式，同时使用精炼后的内部知识和网络搜索获得的外部知识，以补充信息的广度和准确性。

---

### 3. 推理算法 (Inference Algorithm)

#### 3.1 核心流程
输入为查询 $x$ 和检索到的文档集 $D$。

1.  **评估**：评估器 $E$ 计算每对 $(x, d_i)$ 的相关性分数 。
2.  **判断**：根据分数判定置信度状态（Correct / Incorrect / Ambiguous）。
3.  **执行动作**：
    * 若 **Correct**：$k = \text{Knowledge\_Refine}(x, D)$ 。
    * 若 **Incorrect**：$k = \text{Web\_Search}(\text{Rewrite}(x))$ 。
    * 若 **Ambiguous**：$k = \text{Knowledge\_Refine}(x, D) + \text{Web\_Search}(\text{Rewrite}(x))$ 。
4.  **生成**：生成器 $G$ 基于原始查询 $x$ 和最终确定的知识 $k$ 生成回复 $y$ 。

#### 3.2 知识精炼逻辑 (Knowledge Refinement Logic)
* **分解**：将文档分割为细粒度的知识条带（strips，通常为一两句话）。
* **过滤**：再次利用评估器计算每个条带的相关性，过滤掉得分低的条带 。
* **重组**：将剩余的相关条带拼接作为内部知识。

---

### 4. 实验与结果 (Experiments & Results)

#### 4.1 实验设置
* **数据集**：PopQA (短文本生成), Biography (长文本生成), PubHealth (判断题), Arc-Challenge (选择题)。
* **基线模型**：Standard RAG, Self-RAG, LLaMA2, Alpaca。
* **生成器**：主要使用 LLaMA2-7B 。

#### 4.2 主要结果
1.  **性能显著提升**：CRAG 在所有四个数据集上均显著优于标准的 RAG 和最先进的 Self-RAG。
    * 在 PopQA 上，CRAG 基于 SelfRAG-LLaMA2-7b 的准确率比 RAG 提高了 7.0% 。
    * 与 Self-RAG 相比，Self-CRAG 在 Biography 上的 FactScore 提升了 5.0% 。
2.  **通用性强**：CRAG 适用于多种任务类型（短文本、长文本）和底层模型（即插即用） 。
3.  **鲁棒性**：当人为降低检索质量时，Self-CRAG 的性能下降幅度小于 Self-RAG，证明了其对低质量检索的鲁棒性 。
4.  **计算开销低**：评估器非常轻量（T5-large），且仅增加了适度的计算开销 。

---

### 5. 适用场景 (Applicable Scenarios)

| 场景 | CRAG 优势 |
| :--- | :--- |
| **检索质量不稳定的系统** | 当检索器容易返回无关文档时，CRAG 能自动识别并丢弃错误信息，通过网络搜索补救 。 |
| **需要高事实准确性的任务** | 通过知识精炼（Knowledge Refinement）过滤文档中的噪声，只保留精确证据，减少幻觉 。 |
| **静态知识库受限场景** | 当内部知识库无法覆盖查询需求时（Incorrect 模式），自动扩展至网络搜索，获取最新或缺失信息 。 |
| **现有 RAG 系统升级** | 作为即插即用模块，无需重新训练大型生成模型即可集成到现有 RAG 框架中 。 |

## MRAG — A Survey on Multimodal Retrieval-Augmented Generation

----
Mei L, Mo S, Yang Z, et al. A Survey on Multimodal Retrieval-Augmented Generation[J]. arXiv preprint arXiv:2504.08748, 2025. 
----

### 1. 背景与动机 (Background & Motivation)
尽管大型语言模型（LLMs）在文本任务中表现出色，但仍面临特定领域知识匮乏和幻觉（hallucination）等限制 。传统的检索增强生成（RAG）通过引入外部文本知识缓解了这些问题，但忽略了现实世界中丰富的多模态数据（如图像、视频） 。

**MRAG (Multimodal RAG)** 旨在通过整合多模态数据来扩展 RAG 框架，利用多模态大模型（MLLMs）在检索和生成过程中处理多种数据类型，从而提供更全面、准确且基于事实的响应 。

---

### 2. MRAG 的演进阶段 (Evolution of MRAG)

MRAG 的发展经历了三个主要阶段，从简单的文本转换到真正的端到端多模态处理：

#### 2.1 MRAG 1.0: 伪多模态 (Pseudo-MRAG)
* **核心逻辑**：将多模态数据转换为文本（如图像字幕），然后复用传统的文本 RAG 流程 。
* **流程**：文档解析（OCR/Captioning）$\rightarrow$ 文本索引 $\rightarrow$ 文本检索 $\rightarrow$ LLM 生成 。
* **局限性**：文档解析繁琐，转换过程中存在严重的信息丢失（如字幕无法捕捉图像细节），检索准确率受限 。

#### 2.2 MRAG 2.0: 真多模态 (True Multimodal)
* **核心逻辑**：保留原始多模态数据，支持多模态输入，并利用 MLLM 直接处理多模态数据 。
* **改进**：
    * **解析**：使用统一的 MLLM 提取字幕，简化流程 。
    * **检索**：支持跨模态检索（Cross-modal retrieval），允许文本查询直接检索图像等数据 。
    * **生成**：使用 MLLM 结合多模态提示生成答案，减少模态转换带来的损失 。

#### 2.3 MRAG 3.0: 端到端与规划 (End-to-End & Planning)
* **核心逻辑**：引入多模态**输出**能力和**搜索规划**模块，实现从输入到输出的全流程多模态交互 。
* **关键特性**：
    * **文档解析**：保留文档截图（Screenshots）以最小化信息损失 。
    * **搜索规划**：引入规划模块（Search Planning），动态决定是否检索以及如何重构查询（Query Reformulation） 。
    * **多模态输出**：支持生成包含文本、图像或视频的混合模态答案 。

---

### 3. 核心组件与技术 (Key Components & Technologies)

#### 3.1 多模态文档解析与索引 (Document Parsing & Indexing)
* **基于提取的方法 (Extraction-based)**：
    * **纯文本提取**：使用 OCR 或规则提取文本，忽略视觉信息 。
    * **多模态提取**：保留图像、表格等原始格式，或将其转换为字幕/代码（如 HTML、LaTeX） 。
* **基于表示的方法 (Representation-based)**：
    * 直接对文档截图或整体进行编码（Embedding），保留布局和视觉结构，避免解析错误传播 。

#### 3.2 多模态搜索规划 (Multimodal Search Planning)
* **固定规划 (Fixed Planning)**：预定义的检索流程，如强制对所有含图查询进行图像检索，缺乏灵活性 。
* **自适应规划 (Adaptive Planning)**：
    * **检索分类**：动态判断当前查询是否需要检索外部知识（$a_{none}, a_{text}, a_{image}$） 。
    * **查询重构**：利用视觉和文本线索优化查询语句，甚至将复杂问题分解为子查询 。

#### 3.3 多模态检索 (Multimodal Retrieval)
* **检索器 (Retriever)**：
    * **双流架构 (Dual-stream)**：分别编码视觉和语言特征，通过对比学习对齐，效率高 。
    * **生成式检索 (Generative)**：直接生成文档标识符（DocIDs）来检索文档，分为静态 DocID 和可学习 DocID 。
* **重排序器 (Reranker)**：
    * **微调 (Fine-tuning)**：在特定领域数据上微调模型以学习相关性评分]。
    * **提示 (Prompting)**：利用 LLM/MLLM 的零样本能力，通过 Point-wise、Pair-wise 或 List-wise 的方式进行排序 。
* **精炼器 (Refiner)**：
    * **硬提示 (Hard Prompt)**：过滤无关内容，保留关键 token 。
    * **软提示 (Soft Prompt)**：将提示信息压缩为连续的向量表示（Soft tokens） 。

#### 3.4 多模态生成 (Multimodal Generation)
* **模态输入**：支持任意模态组合（如文本+图像、文本+视频）的灵活输入架构 。
* **模态输出**：
    * **原生 MLLM 输出**：模型直接生成多模态内容 。
    * **增强型输出**：先生成文本，再通过位置识别（Position Identification）、候选检索和匹配插入（Matching and Insertion）来嵌入图像或视频 。

---

### 4. 数据集与评估 (Datasets & Evaluation)

#### 4.1 数据集分类
1.  **检索与生成联合数据集 (Retrieval & Generation)**：评估系统检索外部知识并生成响应的能力。例如：InfoSeek, Encyclopedic-VQA, MMSearch, MRAG-bench 。
2.  **纯生成数据集 (Generation)**：评估模型内在知识和推理能力。例如：MMMU (多学科), MathVista (数学), Video-MME (视频理解)。

#### 4.2 评估指标 (Evaluation Metrics)
* **基于规则的指标 (Rule-based)**：
    * EM (Exact Match), BLEU, ROUGE (文本重叠) 。
    * CIDEr, SPICE (图像描述质量)。
* **基于 LLM/MLLM 的指标**：
    * **Answer Precision/Recall**：答案与事实的匹配度 。
    * **Context Precision/Recall**：检索内容与事实的对齐度 。
    * **Faithfulness & Hallucination**：生成内容是否忠实于检索到的证据，以及是否存在幻觉 。

---

### 5. 挑战与未来方向 (Challenges & Future Directions)

#### 5.1 主要挑战
* **数据准确性**：上游解析错误会传播到下游，且长文档中的跨页面关系难以保留 。
* **异构数据对齐**：文本（离散）与图像（连续）的特征对齐困难，难以构建统一的语义空间 。
* **评估缺失**：缺乏针对指令遵循、多轮对话和复杂推理的综合基准测试 。

#### 5.2 未来建议
* **统一多模态表示**：开发更强大的跨模态对齐框架（如 Cross-modal attention），实现真正的统一表示学习。
* **智能代理协作**：利用多智能体（Multi-Agent）系统处理复杂查询，分工进行推理和检索 。
* **人机协同 (HITL)**：在规划和解析阶段引入人类反馈，以处理模糊或高难度的多模态任务。
* **多样化输出控制**：提升生成输出的多样性与相关性的平衡，特别是在生成多媒体内容时 。

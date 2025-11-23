# **3 方法实现（Methodology）**

本研究提出了一种面向真实场景的 **混合式检索增强多智能体系统（Hybrid RAG–driven Multi-Agent System）**，以解决传统 RAG 系统难以处理跨模态文档、实时网页信息以及领域特定检索偏置的问题。本节将详细介绍系统架构、核心模块实现以及关键技术创新。

系统方法包括五大部分：
(1) 多智能体体系结构
(2) Hybrid RAG 检索引擎
(3) 多模态文档解析与处理模块
(4) 实时网页检索模块
(5) 基于查询具体性的动态权重融合（Specificity-Aware Fusion, SAF-RRF）

---

# **3.1 系统总体架构**

本系统基于京东 OxyGent 多智能体框架构建，将系统能力拆解为多个职能单一的 Agent，并通过 Master Agent 进行调度与协同。整体架构如下：

```
                          ┌───────────────────────────────┐
                          │           Master Agent         │
                          │   任务解析 | 工具选择 | 检索融合 | 推理   │
                          └───────────────┬─────────────────┘
                                          │
      ┌───────────────────────────────────┼──────────────────────────────────┐
      │                                   │                                  │
┌──────────────┐                ┌───────────────────┐              ┌────────────────────┐
│ Browser Agent│                │ Hybrid RAG Engine │              │  Multimodal Agent  │
│（实时网页检索）│                │ Dense+Sparse+SAF-RRF │              │（PDF/OCR/视频/图片） │
└──────────────┘                └───────────────────┘              └────────────────────┘
                                          │
                                   ┌──────┴──────────┐
                                   │ Vector DB + BM25 │
                                   │   (本地知识库)    │
                                   └──────────────────┘
```

Master Agent 负责任务分类、工具调用顺序规划与最终推理结果生成；Browser Agent 和 Multimodal Agent 为外部知识来源；Hybrid RAG 提供统一的检索增强能力。

---

# **3.2 Hybrid RAG：混合式检索增强模块**

为应对语义检索、关键词检索、多模态检索和实时性知识的综合需求，我们构建了一个 **Dense + Sparse + Multimodal + Live Web** 的全面检索体系。

## **3.2.1 稠密语义检索（Dense Retrieval）**

使用 Sentence-BERT / GTE / BGE 等模型生成向量表示：

[
$$E(q), \quad E(d)$$
]

通过余弦相似度进行检索：

[
$$Score_{\text{dense}}(q,d)=\cos(E(q),E(d))$$
]

适用于：

* 参数化文档问答
* 长文本语义匹配
* 同义表达、多语言变体

---

## **3.2.2 稀疏关键词检索（Sparse Retrieval, BM25）**

针对专业术语密集的领域（如粮食存储、气象条件、害虫防治），稀疏检索往往更加准确。

BM25 得分为：

[
$$Score_{\text{sparse}}(q,d)=\sum_{t\in q}IDF(t)\cdot\frac{tf(t,d)(k+1)}{tf(t,d)+k(1-b+b\frac{|d|}{avg_d})}$$
]

适用于：

* 数值类问题
* 专业术语匹配
* 表格字段查询

---

## **3.2.3 多模态文本植入（Multimodal-to-Text Retrieval）**

为支持 PDF、图片、PPT、视频等非文本文件，本研究通过 Multimodal Agent（`server.py`）将所有文件转为文本：

* **PDF**：文本解析 + OCR 自动补全
* **图片**：OCR 或图像理解 API
* **视频**：关键帧抽取 + 帧级 OCR
* **Excel**：表格结构化文本化
* **PPT**：逐页文字提取

所有提取的文本被切分为 chunk 并进入 Dense/Sparse 检索库。

---

## **3.2.4 实时网页检索（Live Web Retrieval）**

通过 Browser Agent（`server1.py`）实现：

* Playwright 真实浏览器
* 自动登录京东/百度
* 自动发起查询
* 结构化解析页面正文
* 自动跳转抓取第一条结果
* 抽取标题/正文/超链接

实时网页内容与本地检索结果合并，实现知识补全。

---

# **3.3 Specificity-Aware Fusion：基于查询具体性的动态权重融合（创新点）**

Hybrid RAG 的关键挑战在于：**不同查询对 Dense 与 Sparse 的依赖不同**。

例如：

| 查询                    | 最优检索方式             |
| --------------------- | ------------------ |
| “粮仓温湿度的控制方法”          | Dense（语义泛化）        |
| “磷化氢最低熏蒸温度是多少？”       | Sparse（精确字段）       |
| “如下 PDF 的第 10 页怎么解释？” | Multimodal + Dense |

为解决该问题，本研究提出 **Specificity-Aware Fusion (SAF)**，通过分析查询词的 TF-IDF 具体性自动调节权重。

---

## **3.3.1 查询具体性评分**

$$给定查询 ( Q = {q_i} )，TF-IDF 基于整个知识库计算。$$

[
$$Score_{\text{spec}}=\frac{\sum_{i=1}^{|Q|}TFIDF(q_i)}{|Q|}$$
]

意义：

* **高具体性** → 多为专业词，适合 Sparse
* **低具体性** → 多为通用词，适合 Dense

系统实现于 `compute_specificity_score()`。

---

## **3.3.2 动态权重（Dense 与 Sparse 的自适应权重）**

通过 sigmoid 平滑动态调整权重：

[
$$w_{\text{sparse}}=\sigma(Score_{\text{spec}})$$
]
[
$$w_{\text{dense}}=1 - w_{\text{sparse}}$$
]

保证：

* 通用查询 → Dense 权重高
* 技术词密集查询 → Sparse 权重高

---

## **3.3.3 倒数排名融合（Reciprocal Rank Fusion, RRF）**

最终得分融合公式：

[
$$RRF_{\text{score}}(d)=
\frac{w_{\text{dense}}}{k+r_{\text{dense}}(d)}
+
\frac{w_{\text{sparse}}}{k+r_{\text{sparse}}(d)}
$$
]

其中：

* ( k ) = 平滑参数（常取 60）
* ( r(d) ) = 文档排名（越小越好）

使 Dense 与 Sparse 在**排名层**进行统一融合，更稳健，更抗异常。

---

# **3.4 Multimodal Agent：非文本文档解析模块**

基于 `/mnt/data/server.py`，实现了统一的文件解析能力。

## **3.4.1 PDF 混合解析**

* 使用 PyMuPDF 抽取可用文本
* 如果文本密度低则OCR（Tesseract + 百度识别）补齐
* 页面级结构输出
* 自动过滤页眉页脚

提升对扫描件、图像型 PDF 的召回率。

---

## **3.4.2 图片解析**

两种策略：

1. OCR → 提取文字
2. 图像理解 → 适用于包含视觉实体的问题

系统自动选取识别路径。

---

## **3.4.3 视频解析**

流程：

1. 以固定间隔抽关键帧
2. 帧级 OCR
3. 合并帧文本形成时序描述

可用于监控视频分析等场景。

---

## **3.4.4 Excel 表格解析**

* Sheet 枚举
* 行列结构化抽取
* 转换成自然语言文本

供 RAG 检索使用。

---

# **3.5 Browser Agent：实时网页检索模块**

Browser Agent 基于 `/mnt/data/server1.py` 实现：

### **3.5.1 真浏览器上下文（Playwright）**

* 持久上下文（Persistent Context）
* 自动复用 cookies（京东/百度登录）
* 支持页面，跳转、输入、点击等操作

### **3.5.2 网页结构化提取**

* 页面标题
* 正文
* 剔除广告和噪声节点
* 自动定位搜索结果页中的主要内容

### **3.5.3 Live 内容注入 RAG**

从网页获取的实时内容被：

* 直接加入上下文
* 或即时编码进入临时向量库
* 或作为第二轮检索的 rerank 来源

---

# **3.6 Master Agent：任务调度与推理协调模块**

Master Agent 决定系统的“推理策略”，包括：

---

## **3.6.1 任务解析（Intent Recognition）**

基于启发式规则与 LLM 自分类：

| 特征                | 触发的工具            |
| ----------------- | ---------------- |
| 包含“现在/最新/今日/实时”等  | Browser Agent    |
| 输入含文件（PDF/图片/视频）  | Multimodal Agent |
| 包含精确字段（数值/化学名/术语） | Sparse 优先        |
| 模糊查询、抽象问题         | Dense 优先         |

---

## **3.6.2 工具调用顺序规划**

任务执行顺序示例：

```
if query need live info:
    web_docs = BrowserAgent.search(query)

if input contains multimedia:
    mm_docs = MultimodalAgent.extract(files)

dense_docs = DenseRetrieval(query)
sparse_docs = SparseRetrieval(query)

hybrid_docs = SAF_RRF(dense_docs, sparse_docs)

context = merge(web_docs, mm_docs, hybrid_docs)
```

---

## **3.6.3 不确定性判断与二阶段推理**

第一轮生成后，LLM 会输出置信度（基于 self-evaluation）：

```
if confidence < threshold:
    再次调用 Browser Agent 或 Multimodal Agent
    第二轮生成（refine）
```

---

## **3.6.4 输出结构化答案（Final Answer）**

* 回答
* 使用的证据信息来源
* Web、PDF、OCR 的引用
* 解释性说明（LLM 自动生成）

---

# **3.7 整体算法流程（Pseudo-code）**

```
Algorithm: Multi-Agent Hybrid RAG with SAF-RRF

Input: query Q, optional files F
Output: final answer A

1: intent ← classify_intent(Q)
2: if intent indicates multimedia:
3:        M ← MultimodalAgent.extract(F)
4: if intent indicates realtime:
5:        W ← BrowserAgent.search(Q)
6: D_dense ← DenseRetrieval(Q)
7: D_sparse ← SparseRetrieval(Q)
8: spec ← compute_specificity_score(Q)
9: w_sparse ← sigmoid(spec)
10: w_dense ← 1 - w_sparse
11: R ← SAF_RRF(D_dense, D_sparse, w_dense, w_sparse)
12: context ← merge(R, M, W)
13: draft, conf ← LLM.generate(context, Q)
14: if conf < threshold:
15:        W2 ← BrowserAgent.search(Q)
16:        context ← merge(context, W2)
17:        A ← LLM.generate(context, Q)
18: else:
19:        A ← draft
20: return A
```



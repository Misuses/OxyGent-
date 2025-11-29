## Self-RAG — Learning to Retrieve, Generate, and Critique through Self-Reflection

-----

## Asai A, Wu Z, Wang Y, et al. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection[J]. arXiv preprint arXiv:2310.11511, 2023.

### 1\. 背景与动机 (Background & Motivation)

尽管大型语言模型（LLMs）能力卓越，但由于仅依赖参数化知识，常产生包含事实错误的响应。现有的检索增强生成（RAG）虽然引入了相关知识，但存在两个主要局限性：

1.  **无差别检索（Indiscriminate Retrieval）**：传统 RAG 往往固定检索一定数量的段落，而不论查询是否需要外部知识。引入不必要或不相关的段落可能会导致生成质量下降，甚至阻碍模型的通用性 。
2.  **生成一致性不足**：模型并未被显式训练去利用和遵循检索到的事实，导致输出可能与检索到的相关段落不一致（不支持）。

**Self-RAG** 提出了一种新框架，通过**按需检索（On-demand Retrieval）和自我反思（Self-reflection）来提升生成的质量和事实准确性。其核心思想是训练一个单一的 LM，使其能够生成反思令牌（Reflection Tokens）**，从而在推理阶段动态控制检索行为并自我评价生成内容。

-----

### 2\. 反思令牌与推理框架 (Reflection Tokens & Inference Framework)

#### 2.1 核心目标

Self-RAG 不仅生成任务文本，还生成特殊的反思令牌来指导流程。

  * **输入**：提示词 $x$ 和之前的生成内容 $y_{<t}$。
  * **输出**：包含反思令牌的任务文本段落 $y_t$。
  * **关键机制**：模型在生成过程中动态决定是否检索，并在生成后对自身输出的相关性、支持度和效用进行打分。

#### 2.2 四种反思令牌 (Reflection Tokens)

Self-RAG 引入了四种类型的反思令牌，作为词表扩展进行训练：

| 类型 | 令牌名称 | 定义与功能 | 典型值 |
| :--- | :--- | :--- | :--- |
| **检索决策** | `Retrieve` | 决定是否需要检索外部文档。 | `{Yes, No, Continue}` |
| **相关性检查** | `IsREL` | 判断检索到的文档 $d$ 是否提供了解决 $x$ 的有用信息。 | `{Relevant, Irrelevant}` |
| **证据支持度** | `IsSUP` | 判断生成内容 $y$ 中的陈述是否被文档 $d$ 支持（归因）。 | `{Fully supported, Partially supported, No support}` |
| **效用评估** | `IsUSE` | 评估生成内容 $y$ 对查询 $x$ 的有用程度。 | `{1, 2, 3, 4, 5}` |

#### 2.3 数学描述 (推理评分)

在推理时，模型并行生成多个候选段落，并基于批评令牌的线性加权得分选择最佳段落。段落 $y_t$ 相对于文档 $d$ 的得分 $f$ 计算如下 ：
$f(y_t, d, \text{critique}) = p(y_t | x, d, y_{<t}) + \mathcal{S}(\text{critique}) $

其中 $\mathcal{S}$ 是批评令牌概率的加权和：

$$
\mathcal{S}(\text{critique}) = \sum_{G \in \{\text{IsREL, IsSUP, IsUSE}\}} w^G s_t^G
$$

$w^G$ 是可在推理时调整的权重，用于定制模型行为（例如，提高 `IsSUP` 的权重以生成更具事实依据的内容）。

#### 2.4 代码实现示例 (推理逻辑)

```python
def self_rag_inference(model, retriever, x, context_history):
    # 1. 动态检索决策
    # 模型预测是否需要检索 (Retrieve Token)
    retrieve_token = model.predict_retrieve_token(x, context_history)
    
    if retrieve_token == "Yes":
        # 2. 检索文档
        docs = retriever.retrieve(x, context_history)
        candidates = []
        
        # 3. 并行生成与自我批评
        for d in docs:
            # 生成候选段落 y_t
            segment = model.generate_segment(x, d, context_history)
            
            # 生成批评令牌 (Self-Reflection)
            is_rel = model.predict_is_rel(x, d)         # 相关性
            is_sup = model.predict_is_sup(x, d, segment) # 支持度
            is_use = model.predict_is_use(x, segment)    # 有用性
            
            # 计算综合得分 (Equation 3 & 4)
            # 通过调整 weights 可以控制模型偏好 (如更看重事实支持)
            score = calculate_score(segment_prob, is_rel, is_sup, is_use, weights)
            candidates.append((segment, score))
            
        # 4. 选择最佳段落 (Beam Search Selection)
        best_segment = max(candidates, key=lambda c: c[1])
        return best_segment
        
    else:
        # 5. 标准生成模式 (无检索)
        segment = model.generate_standard(x, context_history)
        return segment
```

-----

### 3\. 训练框架 (Training Framework)

Self-RAG 的训练分为两个阶段：先训练评论家模型（Critic），再训练生成器模型（Generator）。

#### 3.1 评论家模型训练 (Critic Training)

  * **目标**：创建一个能够自动标注反思令牌的模型，用于后续扩充生成器的训练数据。
  * **数据收集**：使用 GPT-4 针对特定指令生成反思令牌（`Retrieve`, `IsREL`, `IsSUP`, `IsUSE`）作为“软标签”。收集了约 4k-20k 条监督数据 。
  * **训练**：初始化一个与生成器相同的 LM（如 Llama 2-7B），在收集的数据 $\mathcal{D}_{critic}$ 上进行标准条件语言建模训练。

$$
\max_{\mathcal{C}} \mathbb{E}_{((x,y),r) \sim \mathcal{D}_{critic}} \log p_{\mathcal{C}}(r|x,y)
$$

#### 3.2 生成器模型训练 (Generator Training)

  * **数据扩充**：利用训练好的 Critic 模型 $C$ 和检索器 $R$，将原始的指令-输出对 $(x, y)$ 扩充为包含检索段落和反思令牌的训练数据 $\mathcal{D}_{gen}$ 。
      * 如果 Critic 认为需要检索，则插入 `Retrieve=Yes` 和检索到的段落。
      * Critic 进一步评价检索段落的相关性 (`IsREL`) 和生成的支持度 (`IsSUP`)。
      * 最终评价整体效用 (`IsUSE`)。
  * **训练目标**：在扩充后的语料 $\mathcal{D}_{gen}$ 上训练生成器 $M$，使其学会生成文本的同时也能生成反思令牌。

$$
\max_{\mathcal{M}} \mathbb{E}_{(x,y,r) \sim \mathcal{D}_{gen}} \log p_{\mathcal{M}}(y,r|x)
$$

-----

### 4\. 实验与效果 (Experiments & Results)

#### 4.1 数据集与任务

实验涵盖了 6 个多样化的任务：

  * **闭集任务**：PubHealth (事实验证), ARC-Challenge (科学推理)。
  * **短文本生成**：PopQA, TriviaQA-unfiltered (开放域问答)。
  * **长文本生成**：Bio (传记生成), ALCE-ASQA (长文本问答)。

#### 4.2 对比基线

  * **无检索基线**：Llama2, Alpaca, ChatGPT, CoVE (Chain-of-Verification)。
  * **检索增强基线**：Llama2-chat (RAG), Ret-ChatGPT, Toolformer, SAIL 。

#### 4.3 核心结果

  * **性能超越**：Self-RAG (7B/13B) 在所有任务上显著优于参数量更大的基线模型（如 Llama2-chat, Alpaca），并在 PubHealth, PopQA, Bio, ASQA 等任务上超越了基于 ChatGPT 的 RAG 系统。
  * **引用准确率**：在 ASQA 任务中，Self-RAG 展现了极高的引用准确率（Citation Precision），甚至在精度上超越了 ChatGPT 。
  * **抗干扰能力**：当检索内容不相关时，Self-RAG 能识别 `IsREL=Irrelevant` 并忽略错误信息，而其他指令微调模型往往会被误导 。

-----

### 5\. 适用场景 (Applicable Scenarios)

| 场景 | 优势描述 |
| :--- | :--- |
| **高事实性要求的 QA** | 利用 `IsSUP` 令牌确保生成的每一个断言都有证据支持，显著减少幻觉 。 |
| **长文本生成/写作** | 在生成长文（如传记）时，能够按需多次检索，保证整篇文章的事实一致性 。 |
| **开放域问答** | 自适应地决定何时检索：面对未知问题自动触发检索，面对已知常识问题直接回答，兼顾效率与准确性 。 |
| **可控生成** | 用户可根据偏好（如更看重准确性还是创造性）在推理阶段调整反思令牌的权重，无需重新训练模型 。 |

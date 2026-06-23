# RNN：CNN-DailyMail 文本摘要课程报告草稿

## 1. 引言

文本摘要是自然语言处理中的序列生成任务，目标是从较长文档中生成简洁、保留关键信息的短文本。本实验使用 CNN/DailyMail 数据集进行新闻摘要建模。需要强调的是，本任务是文本摘要任务，不是翻译任务：输入为新闻正文 `article`，输出为对应摘要 `highlights`。

CNN/DailyMail 数据集由新闻正文和人工整理的摘要要点组成，常用于自动摘要研究。由于 article 通常较长而 highlights 较短，本任务属于长文本到短文本的序列生成问题，适合采用 Encoder-Decoder 框架。

本实验目标是实现并比较多种基于 RNN/LSTM 的摘要模型，包括基础 LSTM Encoder-Decoder、BiLSTM Encoder-Decoder，以及作为主模型的 BiLSTM Encoder + Attention + LSTM Decoder，并使用 ROUGE-1、ROUGE-2 和 ROUGE-L 进行评价。

## 2. 数据集与预处理

实验使用 CNN/DailyMail 数据集，主要字段如下：

| 字段 | 含义 |
|---|---|
| `article` | 新闻正文，作为模型输入 |
| `highlights` | 新闻摘要，作为生成目标 |

预处理流程如下：

1. 使用 HuggingFace `datasets` 加载 `abisee/cnn_dailymail` 数据集。
2. 使用 `transformers` 中的 `t5-small` tokenizer 对 article 和 highlights 进行分词。
3. 对 article 进行截断和 padding，默认最大长度为 400。
4. 对 highlights 进行截断和 padding，默认最大长度为 100。
5. 为降低从零训练 RNN 的计算成本，默认使用数据子集：训练集 5000 条，验证集 500 条，测试集 500 条。

使用子集的原因是完整 CNN/DailyMail 数据集规模较大，RNN 从零训练成本较高；课程作业重点在于完成模型设计、训练流程、评价指标实现和结果分析。

## 3. 模型方法

### 3.1 Seq2Seq 框架

Seq2Seq 模型由 Encoder 和 Decoder 组成。Encoder 将输入 article 编码为隐状态表示，Decoder 根据编码结果逐步生成 highlights。训练时采用 teacher forcing，即在一定概率下使用真实目标词作为下一步输入，以稳定训练过程。

### 3.2 LSTM Encoder-Decoder Baseline

基础模型使用单向 LSTM Encoder 和 LSTM Decoder，不使用 Attention。Encoder 将整篇输入压缩到最终隐状态，Decoder 仅依赖该隐状态生成摘要。该模型作为 baseline，用于观察普通 Seq2Seq 在长文本摘要任务上的表现。

### 3.3 BiLSTM Encoder

BiLSTM Encoder 同时从正向和反向读取输入文本，将两个方向的隐状态拼接后作为文章表示。相比单向 LSTM，BiLSTM 可以利用更完整的上下文信息，因此理论上对 article 编码更充分。

### 3.4 Attention 机制

Attention 机制在 Decoder 生成每个词时，根据当前 Decoder 隐状态对 Encoder 每个时间步的输出计算权重，并得到动态上下文向量。这样 Decoder 不必只依赖单个固定向量，可以在生成不同摘要词时关注原文不同位置，适合处理 CNN/DailyMail 这类长文本摘要任务。

### 3.5 主模型

主模型为 BiLSTM Encoder + Attention + LSTM Decoder。选择该模型的原因如下：

1. 符合课程对 RNN、LSTM、双向结构和 Attention 的要求。
2. Encoder-Decoder 适合从 article 到 highlights 的序列生成过程。
3. BiLSTM 提升长文本上下文编码能力。
4. Attention 缓解长文本压缩为单个向量造成的信息损失。

## 4. 实验设置

默认超参数如下：

| 参数 | 默认值 |
|---|---:|
| embedding_dim | 256 |
| hidden_dim | 512 |
| encoder_layers | 1 |
| decoder_layers | 1 |
| dropout | 0.1 |
| batch_size | 8 |
| learning_rate | 1e-3 |
| epochs | 5 |
| teacher_forcing_ratio | 0.5 |
| optimizer | Adam |
| max_article_len | 400 |
| max_summary_len | 100 |

如果显存不足，可减小 batch size、article 截断长度、hidden_dim 或训练样本数量。训练中使用 gradient clipping 防止梯度爆炸。

评价指标包括：

1. ROUGE-1：unigram 级别重合度，反映关键词覆盖情况。
2. ROUGE-2：bigram 级别重合度，反映短语级匹配情况。
3. ROUGE-L：基于最长公共子序列，反映生成摘要和参考摘要的整体结构相似度。

## 5. 实验结果

请在实际运行训练和评价脚本后填写真实结果，不要虚构 ROUGE 数值。

本次实际运行设置：train 5000 条，validation 500 条，test 500 条，训练 5 epoch。当前 conda 环境未安装 `rouge_score`，以下 ROUGE 使用 `evaluate.py` 中的本地 fallback F1 实现计算。

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | 说明 |
|---|---:|---:|---:|---|
| LSTM Seq2Seq | 2.53 | 0.00 | 2.53 | baseline |
| BiLSTM Seq2Seq | 3.23 | 0.00 | 3.23 | 验证双向编码 |
| BiLSTM + Attention | 6.86 | 0.59 | 5.96 | 主模型 |
| DistilBART-CNN-12-6 | 43.12 | 20.76 | 29.69 | 开源预训练模型对比 |

生成摘要样例格式：

```text
Article:
(CNN) I see signs of a revolution everywhere. I see it in the op-ed pages of the newspapers, and on the state ballots in nearly half the country. I see it in politicians who once preferred to play it safe with this explosive issue but are now willing to stake their political futures on it. I see the revolution in the eyes of sterling scientists, previously reluctant to dip a toe into this heavily stigmatized world, who are diving in head first.

Reference highlights:
CNN's Dr. Sanjay Gupta says we should legalize medical marijuana now. He says he knows how easy it is do nothing "because I did nothing for too long".

RNN main model generated summary:
The:,,,,,,,  . . .  a   a   a  . . .  a  . . . .,,,,,,  . . .  .  .

DistilBART generated summary:
John Sutter: I see signs of a medical marijuana revolution everywhere in the U.S. He says it's burning white hot among young people, but also shows up among parents and grandparents. Sutter says he has even seen the revolution in his own family.
```

## 6. 结果分析

根据本次 5000/500/500 子集实验结果，可以得到以下观察：

1. Attention 明显提升 ROUGE：BiLSTM + Attention 的 ROUGE-1/ROUGE-2/ROUGE-L 为 6.86/0.59/5.96，高于 BiLSTM Seq2Seq 的 3.23/0.00/3.23，说明 Decoder 动态关注原文位置有助于长文本摘要。
2. BiLSTM 相比单向 LSTM 有小幅提升：BiLSTM Seq2Seq 的 ROUGE-1 从 2.53 提升到 3.23，ROUGE-L 从 2.53 提升到 3.23，但提升有限，说明只增强 Encoder 不能完全解决长文本生成问题。
3. 长文本截断会影响摘要质量：article 被截断到 400 token，可能丢失后文信息，导致摘要覆盖不足。
4. RNN 模型局限明显：从生成样例看，模型容易重复短语和标点，语言流畅性不足，对实体、人名、地点的复制能力较弱。这与训练子集较小、从零训练成本高、RNN 长程依赖建模能力有限有关。
5. 开源预训练模型差距明显：DistilBART-CNN-12-6 在同一 500 条测试子集上达到 43.12/20.76/29.69，远高于从零训练的 RNN 主模型。这说明预训练 Transformer 的优势来自大规模语料预训练、CNN/DailyMail 任务微调和更强的长程依赖建模能力。

## 7. 总结

本实验完成了 CNN/DailyMail 文本摘要任务的 RNN 系列模型实现，包括 LSTM Encoder-Decoder baseline、BiLSTM Encoder-Decoder，以及主模型 BiLSTM Encoder + Attention + LSTM Decoder。训练流程包含 teacher forcing、CrossEntropyLoss 忽略 padding、gradient clipping、验证集 loss 输出，评价阶段实现 greedy decoding 并计算 ROUGE-1、ROUGE-2 和 ROUGE-L。

总体而言，RNN 摘要模型结构清晰，能够展示 Encoder-Decoder、双向编码和 Attention 的作用；但在长文本新闻摘要上仍存在长程依赖建模能力有限、生成重复、实体复制能力弱和训练成本较高等问题。未来可以尝试 beam search、coverage mechanism、pointer-generator，或微调 T5/BART 等预训练模型进一步提升摘要质量。


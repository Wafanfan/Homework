# RNN：CNN-DailyMail 文本摘要课程报告

## 摘要

本文围绕 CNN/DailyMail 新闻摘要数据集完成文本摘要实验。需要明确的是，本实验任务是自动文本摘要，不是机器翻译：模型输入为新闻正文 `article`，输出为人工摘要 `highlights`。由于新闻正文通常较长、摘要较短，该任务属于长文本到短文本的序列生成问题，适合采用 Encoder-Decoder 框架建模。

实验首先实现三种基于 RNN/LSTM 的摘要模型：LSTM Encoder-Decoder baseline、BiLSTM Encoder-Decoder，以及作为主模型的 BiLSTM Encoder + Attention + LSTM Decoder。模型使用 PyTorch 从零训练，训练过程中采用 teacher forcing，损失函数为忽略 padding token 的 CrossEntropyLoss，并使用 gradient clipping 缓解梯度爆炸。评价阶段采用 greedy decoding 生成摘要，并计算 ROUGE-1、ROUGE-2 和 ROUGE-L。为了在课程作业环境中可运行，本实验使用 CNN/DailyMail 子集训练和测试：训练集 5000 条，验证集 500 条，测试集 500 条。

在 RNN 系列模型中，BiLSTM + Attention 的 ROUGE-1/ROUGE-2/ROUGE-L 分别为 6.86/0.59/5.96，高于 LSTM baseline 的 2.53/0.00/2.53 和 BiLSTM baseline 的 3.23/0.00/3.23，说明 Attention 对长文本摘要有明显帮助。进一步地，实验加入开源预训练模型 DistilBART-CNN-12-6 作为性能上限参考，在相同 500 条测试子集上达到 43.12/20.76/29.69，显著高于从零训练的 RNN 模型。结果表明，RNN 结构能够用于展示 Encoder-Decoder、BiLSTM 和 Attention 的作用，但在 CNN/DailyMail 这种长文本摘要任务上，从零训练的 RNN 仍存在长程依赖建模弱、重复生成、实体复制能力不足等局限。

**关键词：** 文本摘要；CNN/DailyMail；Seq2Seq；LSTM；BiLSTM；Attention；ROUGE；DistilBART

## 1. 引言

自动文本摘要是自然语言处理中的重要生成任务，目标是将较长文本压缩为简洁、准确、保留核心信息的短文本。新闻摘要是文本摘要中最常见的应用场景之一：新闻正文包含大量背景、细节和引用，而摘要需要突出事件主体、关键事实和主要结论。

本实验使用 CNN/DailyMail 数据集进行新闻摘要建模。该数据集中每条样本包含新闻正文 `article` 和人工摘要 `highlights`。因此，本实验不是翻译任务，不涉及源语言到目标语言的语义转换，而是同一语言内部的长文本压缩和重写。模型需要从 article 中理解主要信息，并生成与 highlights 接近的摘要。

从建模角度看，CNN/DailyMail 属于典型的长文本到短文本序列生成任务。输入序列较长，输出序列较短，且输出不是简单抽取单个标签，而是逐词生成。因此，Encoder-Decoder 框架是合理选择：Encoder 将输入文本编码为隐状态表示，Decoder 根据编码结果逐步生成摘要。考虑课程主题为 RNN，本实验重点实现 LSTM、BiLSTM 和 Attention 机制，并通过对比实验分析它们在摘要任务中的作用。

本实验的主要目标如下：

1. 完成 CNN/DailyMail 文本摘要任务的数据处理、训练、推理和评价流程。
2. 实现 LSTM Encoder-Decoder、BiLSTM Encoder-Decoder 和 BiLSTM + Attention 三种 RNN 模型。
3. 使用 ROUGE-1、ROUGE-2 和 ROUGE-L 对生成摘要进行评价。
4. 分析 BiLSTM 和 Attention 对长文本摘要质量的影响。
5. 加入开源预训练模型 DistilBART 作为性能上限参考，说明从零训练 RNN 与预训练 Transformer 的差距。

## 2. 数据集与预处理

### 2.1 数据集简介

CNN/DailyMail 是自动摘要领域常用数据集，原始数据来自 CNN 和 DailyMail 新闻网页。每条样本主要包括两个字段：

| 字段 | 含义 | 本实验用途 |
|---|---|---|
| `article` | 新闻正文 | 模型输入 |
| `highlights` | 新闻摘要要点 | 模型生成目标 |

本实验通过 HuggingFace `datasets` 加载 `abisee/cnn_dailymail` 数据集，配置版本为 `3.0.0`。由于完整训练集规模较大，从零训练 RNN 模型的成本较高，本实验使用子集完成课程实验：

| Split | 样本数 |
|---|---:|
| Train | 5000 |
| Validation | 500 |
| Test | 500 |

使用子集的原因主要有三点。第一，完整 CNN/DailyMail 训练集约数十万条样本，从零训练 RNN 需要较长时间。第二，课程作业重点是完整实现模型结构、训练流程、评价指标和实验分析，而不是追求最优榜单结果。第三，使用固定子集可以让实验在普通 GPU 环境中更容易复现。

### 2.2 分词与截断

本实验使用 `transformers` 中的 `t5-small` tokenizer 对 article 和 highlights 进行分词。虽然主模型不是 T5，但直接复用成熟的预训练 tokenizer 可以避免手写词表、低频词过滤和特殊符号处理带来的额外复杂度。

RNN 主模型的数据处理设置如下：

| 项目 | 设置 |
|---|---:|
| tokenizer | `t5-small` |
| `max_article_len` | 400 |
| `max_summary_len` | 100 |
| padding | `max_length` |
| truncation | enabled |

正文长度截断为 400 token，摘要长度截断为 100 token。这样做可以控制显存和训练时间，但也会带来信息损失：如果新闻关键信息出现在正文后半部分，截断会导致模型无法看到相关内容。这是后续结果分析中的一个重要因素。

### 2.3 Dataset 与 DataLoader

预处理后的样本包含：

| 字段 | 含义 |
|---|---|
| `input_ids` | article 分词后的 token id |
| `attention_mask` | article padding mask |
| `labels` | highlights 分词后的 token id |

代码中使用 `SummaryDataset` 封装 HuggingFace Dataset，再通过 PyTorch `DataLoader` 构造 batch。训练时 batch size 设为 8。

## 3. 模型方法

### 3.1 Seq2Seq 建模

文本摘要可形式化为条件序列生成问题。给定输入正文序列：

```text
x = (x_1, x_2, ..., x_m)
```

模型需要生成摘要序列：

```text
y = (y_1, y_2, ..., y_n)
```

Seq2Seq 模型将生成概率分解为：

```text
P(y | x) = Π_t P(y_t | y_<t, x)
```

Encoder 负责读取输入序列并生成上下文表示，Decoder 在每个时间步根据已生成 token 和 Encoder 信息预测下一个 token。本实验的 RNN 主模型均遵循这一框架。

### 3.2 LSTM Encoder-Decoder Baseline

第一个实验为单向 LSTM Encoder-Decoder，不使用 Attention。Encoder 将 article 的 token embedding 输入 LSTM，得到最终隐状态和细胞状态；Decoder 使用该状态作为初始状态，逐步生成摘要 token。

该模型结构简单，是基础 baseline。它的主要问题是需要将整篇文章压缩到固定长度的最终隐状态中。对于 CNN/DailyMail 这种长文本任务，固定向量很容易丢失细节信息，导致摘要覆盖不足。

### 3.3 BiLSTM Encoder-Decoder

第二个实验将 Encoder 改为双向 LSTM。BiLSTM 同时从正向和反向读取输入：

```text
forward:  x_1 -> x_2 -> ... -> x_m
backward: x_m -> x_{m-1} -> ... -> x_1
```

每个位置的表示由前向和后向隐状态拼接得到。相比单向 LSTM，BiLSTM 可以同时利用左侧和右侧上下文，对输入文章的编码能力更强。本实验使用该模型验证双向编码是否能提升摘要质量。

### 3.4 Attention 机制

第三个实验为主模型：BiLSTM Encoder + Attention + LSTM Decoder。Attention 的核心思想是：Decoder 在生成每个摘要词时，不只依赖 Encoder 的最终状态，而是动态关注 Encoder 的所有时间步输出。

本实验使用加性 Attention。设 Encoder 输出为 `h_i`，Decoder 当前隐状态为 `s_t`，则注意力分数可表示为：

```text
e_{t,i} = v^T tanh(W_h h_i + W_s s_t)
```

对所有输入位置做 softmax 得到权重：

```text
α_{t,i} = softmax(e_{t,i})
```

上下文向量为 Encoder 输出的加权和：

```text
c_t = Σ_i α_{t,i} h_i
```

Decoder 在生成第 `t` 个 token 时同时使用当前输入 embedding、Decoder 隐状态和上下文向量 `c_t`。这样模型可以在生成不同摘要词时关注原文不同片段，缓解长文本被压缩为单个向量造成的信息损失。

### 3.5 Teacher Forcing

训练时使用 teacher forcing。即在 Decoder 的下一步输入中，以一定概率使用真实目标 token，而不是模型上一步预测 token。本实验设置：

```text
teacher_forcing_ratio = 0.5
```

Teacher forcing 可以让训练更稳定、收敛更快。但在推理阶段，模型无法看到真实前文，只能使用自己生成的 token，因此训练和推理之间存在 exposure bias。这也是 RNN 生成质量不稳定的原因之一。

### 3.6 开源预训练模型对比

除 RNN 主实验外，本实验加入 `sshleifer/distilbart-cnn-12-6` 作为开源预训练 Transformer 对比模型。该模型是针对 CNN/DailyMail fine-tuned 的 DistilBART。需要强调的是，它不是本实验的主模型，也不替代 RNN 主模型；它的作用是作为性能上限参考，展示从零训练 RNN 与大规模预训练模型之间的差距。

DistilBART 评估时使用 beam search：

| 参数 | 值 |
|---|---:|
| `max_source_len` | 1024 |
| `max_target_len` | 128 |
| `min_target_len` | 30 |
| `num_beams` | 4 |
| `length_penalty` | 2.0 |
| `no_repeat_ngram_size` | 3 |

与 RNN 主模型相比，DistilBART 有三个明显优势：第一，它已经经过大规模预训练，具备较强语言建模能力；第二，它已在 CNN/DailyMail 上微调；第三，Transformer 结构比普通 RNN 更擅长建模长距离依赖。

## 4. 代码实现

项目结构如下：

```text
cnn_dm_summary/
├── data_prepare.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── evaluate_pretrained.py
├── utils.py
├── requirements.txt
├── README.md
├── report_draft.md
└── course_report.md
```

各文件功能如下：

| 文件 | 功能 |
|---|---|
| `data_prepare.py` | 加载 CNN/DailyMail，分词、截断、padding，保存处理后的数据 |
| `dataset.py` | 封装 PyTorch Dataset 和 DataLoader |
| `model.py` | 实现 Encoder、Attention、Decoder、Seq2Seq |
| `train.py` | 训练 RNN 模型，保存 best/last checkpoint |
| `evaluate.py` | 对 RNN checkpoint 进行 greedy decoding 并计算 ROUGE |
| `evaluate_pretrained.py` | 评估 HuggingFace 开源预训练摘要模型 |
| `utils.py` | 随机种子、设备选择、checkpoint、JSON 写入等工具 |

模型训练使用 Adam 优化器，损失函数为：

```text
CrossEntropyLoss(ignore_index=pad_token_id)
```

忽略 padding token 可以避免模型因为大量 padding 位置而产生无意义损失。训练时还使用：

```text
grad_clip = 1.0
```

用于缓解 RNN 训练中的梯度爆炸问题。

## 5. 实验设置

### 5.1 RNN 实验超参数

| 参数 | 值 |
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
| grad_clip | 1.0 |
| max_article_len | 400 |
| max_summary_len | 100 |

### 5.2 模型规模

| Model | Trainable Parameters |
|---|---:|
| LSTM Seq2Seq | 36.58M |
| BiLSTM Seq2Seq | 38.68M |
| BiLSTM + Attention | 82.66M |

主模型参数量显著大于两个 baseline。由于训练集只有 5000 条，主模型后期更容易过拟合，这一点在训练 loss 和 validation loss 中也能观察到。

### 5.3 评价指标

实验使用 ROUGE-1、ROUGE-2 和 ROUGE-L 评价摘要质量：

1. ROUGE-1：unigram 级别重合度，反映关键词覆盖情况。
2. ROUGE-2：bigram 级别重合度，反映短语级匹配情况。
3. ROUGE-L：基于最长公共子序列，反映生成摘要与参考摘要的整体结构相似度。

本次运行环境中未安装 `rouge_score` 包，因此 `evaluate.py` 使用本地 fallback 实现计算 ROUGE F1。该实现用于本实验模型间横向对比是有效的；如果安装 `rouge-score`，代码会自动优先使用官方库，绝对数值可能有小幅差异。

## 6. 实验结果

### 6.1 Validation Loss

| Model | 训练情况 | Validation Loss |
|---|---|---:|
| LSTM Seq2Seq | epoch 5 | 5.6677 |
| BiLSTM Seq2Seq | epoch 5 | 5.6746 |
| BiLSTM + Attention | best epoch 3 | 5.6785 |
| BiLSTM + Attention | epoch 5 | 5.7864 |

从 validation loss 看，LSTM baseline 略低于主模型 best checkpoint，但差异只有约 0.01，不能说明 LSTM 的摘要生成质量更好。CrossEntropy loss 衡量的是 teacher forcing 条件下的逐 token 预测能力，而 ROUGE 衡量的是模型自由生成完整摘要后的质量。二者相关但不完全一致。

主模型第 5 个 epoch 的训练 loss 降到 4.5316，但 validation loss 上升到 5.7864，说明 Attention 模型后期出现过拟合。因此最终评估使用第 3 个 epoch 保存的 best checkpoint。

### 6.2 ROUGE 结果

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | 说明 |
|---|---:|---:|---:|---|
| LSTM Seq2Seq | 2.53 | 0.00 | 2.53 | baseline |
| BiLSTM Seq2Seq | 3.23 | 0.00 | 3.23 | 验证双向编码 |
| BiLSTM + Attention | 6.86 | 0.59 | 5.96 | RNN 主模型 |
| DistilBART-CNN-12-6 | 43.12 | 20.76 | 29.69 | 开源预训练模型对比 |

从 RNN 三组实验看，结果呈现清晰趋势：

```text
LSTM Seq2Seq < BiLSTM Seq2Seq < BiLSTM + Attention
```

这说明双向编码有一定帮助，而 Attention 对长文本摘要的提升更加明显。虽然 RNN 主模型的绝对 ROUGE 不高，但相对 baseline 的提升符合预期。

DistilBART 结果显著高于从零训练的 RNN 模型，达到 ROUGE-1 43.12、ROUGE-2 20.76、ROUGE-L 29.69，说明预训练摘要模型可以稳定达到二三十以上的 ROUGE 水平。该结果也与公开资料中 CNN/DailyMail 上 BART/DistilBART 通常显著优于从零训练 RNN 的结论一致。

### 6.3 生成样例

```text
Article:
(CNN) I see signs of a revolution everywhere. I see it in the op-ed pages
of the newspapers, and on the state ballots in nearly half the country. I
see it in politicians who once preferred to play it safe with this explosive
issue but are now willing to stake their political futures on it. I see the
revolution in the eyes of sterling scientists, previously reluctant to dip a
toe into this heavily stigmatized world, who are diving in head first.

Reference highlights:
CNN's Dr. Sanjay Gupta says we should legalize medical marijuana now. He
says he knows how easy it is do nothing "because I did nothing for too long".

RNN main model generated summary:
The:,,,,,,,  . . .  a   a   a  . . .  a  . . . .,,,,,,  . . .  .  .

DistilBART generated summary:
John Sutter: I see signs of a medical marijuana revolution everywhere in
the U.S. He says it's burning white hot among young people, but also shows
up among parents and grandparents. Sutter says he has even seen the
revolution in his own family.
```

该样例能够直观看出两类模型差距。RNN 主模型虽然在 ROUGE 上优于 baseline，但实际生成中仍存在严重重复和标点异常；DistilBART 则能生成结构较完整、语义较连贯的新闻摘要。

## 7. 结果分析

### 7.1 Attention 的作用

BiLSTM + Attention 的 ROUGE-1 从 BiLSTM baseline 的 3.23 提升到 6.86，ROUGE-L 从 3.23 提升到 5.96。Attention 的提升原因在于 Decoder 生成每个 token 时可以动态访问 Encoder 的不同时间步输出，不必完全依赖固定长度隐状态。对于 CNN/DailyMail 这种长文本任务，固定向量瓶颈非常明显，因此 Attention 的收益比单纯改为 BiLSTM 更大。

### 7.2 BiLSTM 的作用

BiLSTM Seq2Seq 相比 LSTM Seq2Seq 有小幅提升：ROUGE-1 从 2.53 提升到 3.23，ROUGE-L 从 2.53 提升到 3.23。这说明双向上下文有助于 Encoder 表示文章内容。但提升有限，原因是无 Attention 的 Decoder 仍然主要依赖 Encoder 最终状态，长文本信息依然被压缩到有限向量中。

### 7.3 Loss 与 ROUGE 不完全一致

实验中 LSTM baseline 的 validation loss 略低于 BiLSTM + Attention best checkpoint，但 ROUGE 明显更低。这是合理现象。Validation loss 是 teacher forcing 条件下的逐 token 交叉熵，模型每一步可以看到真实前文；而 ROUGE 是推理阶段模型自己生成完整摘要后的匹配分数。一个模型可能在局部 token 预测上更保守、loss 更低，但自由生成时摘要质量较差。

因此，本任务最终应以 ROUGE 和生成样例为主要评价依据，而不是只看 validation loss。结合 ROUGE 结果，BiLSTM + Attention 仍然是三种 RNN 模型中最好的主模型。

### 7.4 RNN 模型的局限

从零训练的 RNN 模型在本实验中表现较弱，主要原因包括：

1. **训练数据有限。** 本实验只使用 5000 条训练样本，而 CNN/DailyMail 完整训练集规模更大。从零训练生成模型需要更多数据。
2. **长程依赖建模能力有限。** LSTM 虽然缓解了普通 RNN 的梯度问题，但面对 400 token 甚至更长的新闻正文，仍难以稳定保留关键信息。
3. **缺少复制机制。** 新闻摘要中人名、地名、机构名非常重要，但普通 Seq2Seq 模型没有 pointer/copy mechanism，复制实体能力较弱。
4. **容易重复生成。** 生成样例中出现明显重复和标点异常，说明模型缺少 coverage mechanism，也没有在 RNN 解码中加入重复惩罚。
5. **曝光偏差。** 训练时使用 teacher forcing，推理时使用模型自身预测，错误会逐步累积。

### 7.5 预训练模型的优势

DistilBART-CNN-12-6 在同一测试子集上达到 43.12/20.76/29.69，远高于 RNN 主模型。这种差距主要来自以下方面：

1. **大规模预训练。** DistilBART 在大量文本上学习过语言表示和生成能力。
2. **任务微调。** 该模型已在 CNN/DailyMail 摘要任务上 fine-tune。
3. **Transformer 架构。** Self-attention 更适合建模长距离依赖。
4. **更长输入。** DistilBART 使用 1024 token 输入长度，而 RNN 主模型使用 400 token。
5. **Beam search。** 预训练模型评估使用 beam search 和 `no_repeat_ngram_size=3`，生成质量优于简单 greedy decoding。

因此，DistilBART 不应被视为对 RNN 主模型设计的否定，而应作为性能上限参考，说明现代预训练 Transformer 在文本摘要任务上的优势。

## 8. 复现实验命令

### 8.1 安装依赖

```bash
cd /data/nHome/wangyifan/homework/RNN/cnn_dm_summary
pip install -r requirements.txt
```

### 8.2 数据准备

```bash
python data_prepare.py \
  --train_size 5000 \
  --val_size 500 \
  --test_size 500 \
  --max_article_len 400 \
  --max_summary_len 100 \
  --output_dir data/processed
```

### 8.3 训练 RNN 模型

```bash
python train.py \
  --model_type lstm \
  --processed_dir data/processed \
  --batch_size 8 \
  --epochs 5 \
  --output_dir checkpoints/lstm \
  --device cuda:0
```

```bash
python train.py \
  --model_type bilstm \
  --processed_dir data/processed \
  --batch_size 8 \
  --epochs 5 \
  --output_dir checkpoints/bilstm \
  --device cuda:0
```

```bash
python train.py \
  --model_type bilstm_attention \
  --processed_dir data/processed \
  --batch_size 8 \
  --epochs 5 \
  --output_dir checkpoints/bilstm_attention \
  --device cuda:0
```

### 8.4 评价 RNN 模型

```bash
python evaluate.py \
  --checkpoint checkpoints/bilstm_attention/best.pt \
  --test_size 500 \
  --batch_size 8 \
  --device cuda:0
```

### 8.5 评价 DistilBART

```bash
python evaluate_pretrained.py \
  --model_name sshleifer/distilbart-cnn-12-6 \
  --test_size 500 \
  --batch_size 8 \
  --device cuda:0 \
  --fp16 \
  --output_jsonl checkpoints/distilbart_cnn_12_6/test_predictions.jsonl \
  --output_metrics checkpoints/distilbart_cnn_12_6/test_rouge.json
```

## 9. 总结

本实验完成了 CNN/DailyMail 文本摘要任务的完整课程项目。首先，实现了数据加载、分词、截断、padding、Dataset/DataLoader、模型训练、checkpoint 保存、摘要生成和 ROUGE 评价。其次，实现并比较了三种 RNN 模型：LSTM Encoder-Decoder、BiLSTM Encoder-Decoder 和 BiLSTM + Attention + LSTM Decoder。实验结果显示，BiLSTM + Attention 在 RNN 系列中效果最好，说明 Attention 对长文本摘要任务具有实际价值。

同时，实验加入 DistilBART-CNN-12-6 作为开源预训练模型对比。该模型在相同测试子集上取得远高于 RNN 的 ROUGE，说明当前摘要任务中预训练 Transformer 具有明显优势。由于课程要求主模型围绕 RNN 展开，本文仍以 BiLSTM + Attention + LSTM Decoder 作为主模型，DistilBART 仅作为性能上限参考。

未来可以从以下方向改进 RNN 摘要模型：

1. 使用完整 CNN/DailyMail 训练集训练更久。
2. 在 RNN Decoder 中加入 beam search。
3. 引入 pointer-generator，提高实体和关键词复制能力。
4. 引入 coverage mechanism，减少重复生成。
5. 尝试预训练词向量或预训练 Encoder。
6. 微调 T5-small、BART 或 PEGASUS，与 RNN 结果进行更系统对比。

总体而言，本实验展示了 RNN、BiLSTM 和 Attention 在文本摘要任务中的基本作用，也通过 DistilBART 对比说明了现代预训练模型在长文本摘要上的优势和必要性。

## 参考资料

[1] CNN/DailyMail dataset repository: https://github.com/abisee/cnn-dailymail

[2] HuggingFace `abisee/cnn_dailymail` dataset: https://huggingface.co/datasets/abisee/cnn_dailymail

[3] DistilBART CNN/DailyMail model card: https://huggingface.co/sshleifer/distilbart-cnn-12-6

[4] BART large CNN/DailyMail model card: https://huggingface.co/facebook/bart-large-cnn

[5] Lewis et al. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. https://arxiv.org/abs/1910.13461

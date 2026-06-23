# CNN/DailyMail RNN Text Summarization

本项目实现课程大作业“RNN：CNN-DailyMail 文本摘要”。任务是文本摘要，不是翻译：输入为 CNN/DailyMail 数据集中的 `article` 新闻正文，输出为 `highlights` 摘要。该任务属于长文本到短文本的序列生成任务，适合使用 Encoder-Decoder 框架。

## 模型设计

本项目包含 3 个 RNN 实验：

| 实验 | 命令参数 | 目的 |
|---|---|---|
| LSTM Encoder-Decoder | `--model_type lstm` | 基础 baseline，无 Attention |
| BiLSTM Encoder-Decoder | `--model_type bilstm` | 验证双向编码的作用 |
| BiLSTM + Attention + LSTM Decoder | `--model_type bilstm_attention` | 主模型，验证 Attention 对长文本摘要的作用 |

主模型选择 BiLSTM + Attention + LSTM Decoder 的原因：

1. 符合 RNN、LSTM、双向结构和 Attention 的课程要求。
2. CNN/DailyMail 是长文本摘要任务，Encoder-Decoder 适合建模 article 到 highlights 的生成过程。
3. BiLSTM 能同时利用前向和后向上下文，比单向 LSTM 编码能力更强。
4. Attention 让 Decoder 在生成每个词时动态关注原文不同位置，缓解长文本压缩为单个向量的信息损失。

## 环境安装

```bash
pip install -r requirements.txt
```

如果网络或 HuggingFace 访问受限，数据集和 `t5-small` tokenizer 可能下载失败。可以提前配置 `HF_HOME` 或使用 `--cache_dir` 指向已有缓存。

## 数据准备

默认使用 HuggingFace `abisee/cnn_dailymail` 数据集的 `3.0.0` 配置，并使用 `t5-small` tokenizer。默认截断长度为：

- `max_article_len = 400`
- `max_summary_len = 100`

为了适合课程作业和普通 GPU/CPU，默认只使用子集。完整数据集较大，从零训练 RNN 成本高；课程重点是完成模型设计、训练流程、评价和分析。

可先单独准备数据：

```bash
python data_prepare.py \
  --train_size 5000 \
  --val_size 500 \
  --test_size 500 \
  --max_article_len 400 \
  --max_summary_len 100 \
  --output_dir data/processed
```

也可以直接运行训练脚本，训练脚本会自动下载并分词。

## 训练

主模型：

```bash
python train.py \
  --model_type bilstm_attention \
  --processed_dir data/processed \
  --batch_size 8 \
  --epochs 5 \
  --learning_rate 1e-3 \
  --teacher_forcing_ratio 0.5 \
  --output_dir checkpoints/bilstm_attention
```

两个对比模型：

```bash
python train.py --model_type lstm --processed_dir data/processed --output_dir checkpoints/lstm
python train.py --model_type bilstm --processed_dir data/processed --output_dir checkpoints/bilstm
```

显存不足时可依次减小：

1. `--batch_size`
2. `--max_article_len`
3. `--hidden_dim`
4. `--train_size`

代码已使用 gradient clipping，默认 `--grad_clip 1.0`，用于缓解梯度爆炸。

## 评价

```bash
python evaluate.py \
  --checkpoint checkpoints/bilstm_attention/best.pt \
  --test_size 500 \
  --batch_size 8
```

开源预训练模型对比，可直接复现 DistilBART 结果：

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

`evaluate.py` 使用 greedy decoding；`evaluate_pretrained.py` 使用 beam search。两个脚本都会计算：

- ROUGE-1：unigram 重合度，反映关键词覆盖情况。
- ROUGE-2：bigram 重合度，反映短语级匹配情况。
- ROUGE-L：基于最长公共子序列，反映整体结构相似度。

评价输出文件默认保存在 checkpoint 目录：

- `test_predictions.jsonl`
- `test_rouge.json`

## 实验结果表格

不要填写未实际运行的分数。运行后将真实 ROUGE 结果填入下表：

本次实际运行设置：train 5000 条，validation 500 条，test 500 条，训练 5 epoch。当前 conda 环境未安装 `rouge_score`，以下 ROUGE 使用 `evaluate.py` 中的本地 fallback F1 实现计算。

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | 说明 |
|---|---:|---:|---:|---|
| LSTM Seq2Seq | 2.53 | 0.00 | 2.53 | baseline |
| BiLSTM Seq2Seq | 3.23 | 0.00 | 3.23 | 验证双向编码 |
| BiLSTM + Attention | 6.86 | 0.59 | 5.96 | 主模型 |
| DistilBART-CNN-12-6 | 43.12 | 20.76 | 29.69 | 开源预训练模型对比 |

## 生成摘要样例格式

```text
Article:
待粘贴原文片段

Reference highlights:
待粘贴参考摘要

Generated summary:
待粘贴模型生成摘要
```


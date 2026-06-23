"""Generate summaries and compute ROUGE for a trained checkpoint."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import DatasetDict
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

from data_prepare import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    get_tokenizer,
    load_raw_splits,
    tokenize_splits,
)
from dataset import create_dataloader
from model import build_model
from utils import get_device, load_checkpoint, write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CNN/DailyMail summarizer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--max_article_len", type=int, default=None)
    parser.add_argument("--max_summary_len", type=int, default=None)
    parser.add_argument("--max_decode_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_metrics", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tokens(text: str):
    return re.findall(r"\w+", text.lower())


def _ngram_counts(tokens, n: int):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a, b) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _compute_rouge_fallback(predictions, references):
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        pred_tokens = _tokens(pred)
        ref_tokens = _tokens(ref)
        for key, n in (("rouge1", 1), ("rouge2", 2)):
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts = _ngram_counts(ref_tokens, n)
            overlap = sum((pred_counts & ref_counts).values())
            totals[key] += _f1(overlap, sum(pred_counts.values()), sum(ref_counts.values()))
        lcs = _lcs_length(pred_tokens, ref_tokens)
        totals["rougeL"] += _f1(lcs, len(pred_tokens), len(ref_tokens))
    count = max(len(predictions), 1)
    return {key: 100.0 * value / count for key, value in totals.items()}


def compute_rouge(predictions, references):
    if rouge_scorer is None:
        return _compute_rouge_fallback(predictions, references)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    count = max(len(predictions), 1)
    return {key: 100.0 * value / count for key, value in totals.items()}


def main():
    args = parse_args()
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    train_args = checkpoint.get("train_args", {})

    dataset_name = args.dataset_name or train_args.get("dataset_name") or DEFAULT_DATASET_NAME
    dataset_config = args.dataset_config or train_args.get("dataset_config") or DEFAULT_DATASET_CONFIG
    tokenizer_name = args.tokenizer_name or checkpoint.get("tokenizer_name") or "t5-small"
    max_article_len = args.max_article_len or train_args.get("max_article_len", 400)
    max_summary_len = args.max_summary_len or train_args.get("max_summary_len", 100)
    max_decode_len = args.max_decode_len or max_summary_len
    test_size = args.test_size if args.test_size is not None else train_args.get("test_size", 500)

    tokenizer = get_tokenizer(tokenizer_name)
    raw = load_raw_splits(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        cache_dir=args.cache_dir,
        train_size=0,
        val_size=test_size if args.split == "validation" else 0,
        test_size=test_size if args.split == "test" else 0,
        seed=args.seed,
    )
    raw_split = raw[args.split]
    tokenized = tokenize_splits(
        DatasetDict({args.split: raw_split}),
        tokenizer=tokenizer,
        max_article_len=max_article_len,
        max_summary_len=max_summary_len,
    )[args.split]
    dataloader = create_dataloader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    references = []
    rows = []
    seen = 0

    for batch in tqdm(dataloader, desc="generate"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_len=max_decode_len,
        )
        preds = tokenizer.batch_decode(generated_ids.cpu().tolist(), skip_special_tokens=True)
        refs = tokenizer.batch_decode(batch["labels"].tolist(), skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(refs)

        batch_size = len(preds)
        for i in range(batch_size):
            raw_item = raw_split[seen + i]
            rows.append(
                {
                    "article": raw_item["article"],
                    "reference": raw_item["highlights"],
                    "prediction": preds[i],
                }
            )
        seen += batch_size

    metrics = compute_rouge(predictions, references)
    print(
        "ROUGE - "
        f"R1: {metrics['rouge1']:.2f}, "
        f"R2: {metrics['rouge2']:.2f}, "
        f"RL: {metrics['rougeL']:.2f}"
    )

    checkpoint_dir = Path(args.checkpoint).parent
    output_jsonl = args.output_jsonl or checkpoint_dir / f"{args.split}_predictions.jsonl"
    output_metrics = args.output_metrics or checkpoint_dir / f"{args.split}_rouge.json"
    write_jsonl(output_jsonl, rows)
    write_json(
        output_metrics,
        {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "num_examples": len(predictions),
            "rouge": metrics,
        },
    )
    print(f"Saved predictions to {output_jsonl}")
    print(f"Saved metrics to {output_metrics}")


if __name__ == "__main__":
    main()


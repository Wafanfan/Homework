"""Evaluate an open-source pretrained summarization model on CNN/DailyMail."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data_prepare import DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_NAME, load_raw_splits
from evaluate import compute_rouge
from utils import get_device, write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a HuggingFace seq2seq summarization model on CNN/DailyMail."
    )
    parser.add_argument("--model_name", default="sshleifer/distilbart-cnn-12-6")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--min_target_len", type=int, default=30)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=2.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_metrics", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def batched(items, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main():
    args = parse_args()
    device = get_device(args.device)

    raw = load_raw_splits(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        train_size=0,
        val_size=args.test_size if args.split == "validation" else 0,
        test_size=args.test_size if args.split == "test" else 0,
        seed=args.seed,
    )[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model.half()
    model.eval()

    predictions = []
    references = []
    rows = []

    for _, batch in tqdm(list(batched(list(raw), args.batch_size)), desc="generate"):
        articles = [item["article"] for item in batch]
        refs = [item["highlights"] for item in batch]
        encoded = tokenizer(
            articles,
            max_length=args.max_source_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=args.max_target_len,
                min_length=args.min_target_len,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
            )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(preds)
        references.extend(refs)
        for item, pred, ref in zip(batch, preds, refs):
            rows.append(
                {
                    "article": item["article"],
                    "reference": ref,
                    "prediction": pred,
                }
            )

    metrics = compute_rouge(predictions, references)
    print(
        "ROUGE - "
        f"R1: {metrics['rouge1']:.2f}, "
        f"R2: {metrics['rouge2']:.2f}, "
        f"RL: {metrics['rougeL']:.2f}"
    )

    safe_name = args.model_name.replace("/", "_")
    output_dir = Path("checkpoints") / safe_name
    output_jsonl = args.output_jsonl or output_dir / f"{args.split}_predictions.jsonl"
    output_metrics = args.output_metrics or output_dir / f"{args.split}_rouge.json"
    write_jsonl(output_jsonl, rows)
    write_json(
        output_metrics,
        {
            "model_name": args.model_name,
            "split": args.split,
            "num_examples": len(predictions),
            "generation": {
                "max_source_len": args.max_source_len,
                "max_target_len": args.max_target_len,
                "min_target_len": args.min_target_len,
                "num_beams": args.num_beams,
                "length_penalty": args.length_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            },
            "rouge": metrics,
        },
    )
    print(f"Saved predictions to {output_jsonl}")
    print(f"Saved metrics to {output_metrics}")


if __name__ == "__main__":
    main()

"""Data loading and tokenization for CNN/DailyMail summarization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


DEFAULT_DATASET_NAME = "abisee/cnn_dailymail"
DEFAULT_DATASET_CONFIG = "3.0.0"
DEFAULT_TOKENIZER = "t5-small"


def get_tokenizer(tokenizer_name: str = DEFAULT_TOKENIZER):
    """Load a HuggingFace tokenizer and make sure it has a pad token."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def _select_subset(split, size: Optional[int], seed: int):
    if size is None or size < 0:
        return split
    if size == 0:
        return split.select(range(0))
    size = min(size, len(split))
    return split.shuffle(seed=seed).select(range(size))


def load_raw_splits(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    cache_dir: Optional[str] = None,
    train_size: Optional[int] = 5000,
    val_size: Optional[int] = 500,
    test_size: Optional[int] = 500,
    seed: int = 42,
) -> DatasetDict:
    """Download/load CNN-DailyMail and return selected splits."""
    raw = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    return DatasetDict(
        {
            "train": _select_subset(raw["train"], train_size, seed),
            "validation": _select_subset(raw["validation"], val_size, seed),
            "test": _select_subset(raw["test"], test_size, seed),
        }
    )


def _tokenize_targets(tokenizer, summaries, max_summary_len: int):
    try:
        return tokenizer(
            text_target=summaries,
            max_length=max_summary_len,
            truncation=True,
            padding="max_length",
        )
    except TypeError:
        return tokenizer(
            summaries,
            max_length=max_summary_len,
            truncation=True,
            padding="max_length",
        )


def tokenize_splits(
    raw_splits: DatasetDict,
    tokenizer,
    max_article_len: int = 400,
    max_summary_len: int = 100,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    """Tokenize article as source and highlights as target."""

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["article"],
            max_length=max_article_len,
            truncation=True,
            padding="max_length",
        )
        labels = _tokenize_targets(tokenizer, batch["highlights"], max_summary_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = None
    for split_name in raw_splits:
        if len(raw_splits[split_name]):
            remove_columns = raw_splits[split_name].column_names
            break
    if remove_columns is None:
        for split_name in raw_splits:
            remove_columns = raw_splits[split_name].column_names
            break

    return raw_splits.map(
        preprocess,
        batched=True,
        num_proc=num_proc,
        remove_columns=remove_columns,
        desc="Tokenizing CNN/DailyMail",
    )


def load_or_prepare_splits(
    tokenizer,
    processed_dir: Optional[str] = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    cache_dir: Optional[str] = None,
    train_size: Optional[int] = 5000,
    val_size: Optional[int] = 500,
    test_size: Optional[int] = 500,
    max_article_len: int = 400,
    max_summary_len: int = 100,
    seed: int = 42,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    """Load tokenized splits from disk, or download and tokenize them."""
    if processed_dir and Path(processed_dir).exists():
        return load_from_disk(processed_dir)

    raw = load_raw_splits(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        cache_dir=cache_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    tokenized = tokenize_splits(
        raw,
        tokenizer=tokenizer,
        max_article_len=max_article_len,
        max_summary_len=max_summary_len,
        num_proc=num_proc,
    )
    if processed_dir:
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(processed_dir)
    return tokenized


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CNN/DailyMail tokenized data.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--max_article_len", type=int, default=400)
    parser.add_argument("--max_summary_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = get_tokenizer(args.tokenizer_name)
    tokenized = load_or_prepare_splits(
        tokenizer=tokenizer,
        processed_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        max_article_len=args.max_article_len,
        max_summary_len=args.max_summary_len,
        seed=args.seed,
        num_proc=args.num_proc,
    )
    print("Prepared tokenized splits:")
    for split_name, split in tokenized.items():
        print(f"  {split_name}: {len(split)} examples")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


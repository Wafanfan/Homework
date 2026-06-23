"""PyTorch dataset wrappers for tokenized summarization examples."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class SummaryDataset(Dataset):
    """Wrap a tokenized HuggingFace Dataset split."""

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[int(idx)]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def create_dataloader(
    split,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        SummaryDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


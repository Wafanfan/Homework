"""Train RNN Seq2Seq summarization models on CNN/DailyMail."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from data_prepare import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_TOKENIZER,
    get_tokenizer,
    load_or_prepare_splits,
)
from dataset import create_dataloader
from model import build_model
from utils import (
    AverageMeter,
    count_parameters,
    get_device,
    save_checkpoint,
    set_seed,
    write_json,
)


MODEL_TYPES = ("lstm", "bilstm", "bilstm_attention")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN/DailyMail RNN summarizer.")
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="bilstm_attention")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--max_article_len", type=int, default=400)
    parser.add_argument("--max_summary_len", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    return parser.parse_args()


def make_model_config(args, tokenizer) -> dict:
    bidirectional = args.model_type in {"bilstm", "bilstm_attention"}
    use_attention = args.model_type == "bilstm_attention"
    start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = tokenizer.pad_token_id
    return {
        "vocab_size": len(tokenizer),
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "dropout": args.dropout,
        "bidirectional": bidirectional,
        "use_attention": use_attention,
        "pad_token_id": tokenizer.pad_token_id,
        "start_token_id": start_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    teacher_forcing_ratio: float = 0.5,
    grad_clip: float = 1.0,
):
    is_train = optimizer is not None
    model.train(is_train)
    meter = AverageMeter()
    iterator = tqdm(dataloader, leave=False, desc="train" if is_train else "valid")

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        meter.update(loss.item(), input_ids.size(0))
        iterator.set_postfix(loss=f"{meter.avg:.4f}")
    return meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    tokenizer = get_tokenizer(args.tokenizer_name)

    tokenized = load_or_prepare_splits(
        tokenizer=tokenizer,
        processed_dir=args.processed_dir,
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

    pin_memory = device.type == "cuda"
    train_loader = create_dataloader(
        tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = create_dataloader(
        tokenized["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = make_model_config(args, tokenizer)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    output_dir = Path(args.output_dir or f"checkpoints/{args.model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"args": vars(args), "model_config": model_config})

    print(f"Device: {device}")
    print(f"Model: {args.model_type}")
    print(f"Train examples: {len(tokenized['train'])}")
    print(f"Validation examples: {len(tokenized['validation'])}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            grad_clip=args.grad_clip,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            teacher_forcing_ratio=1.0,
            grad_clip=0.0,
        )
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_loss,
            model_config,
            args.tokenizer_name,
            vars(args),
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_loss,
                model_config,
                args.tokenizer_name,
                vars(args),
            )
            print(f"Saved new best checkpoint to {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()


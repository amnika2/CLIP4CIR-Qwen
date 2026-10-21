import json
import math
import random
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from cli import PROJECT_ROOT, add_model_arguments
from contrastive import contrastive_loss
from data_utils import CIRRDataset
from lora_encoder import QwenLoraEncoder
from retrieval import evaluate_cirr


def collate_items(items):
    return items


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    encoder: QwenLoraEncoder,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: torch.amp.GradScaler,
    args,
    epoch: int,
) -> float:
    encoder.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    progress = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")

    for batch_index, items in enumerate(progress, start=1):
        with torch.autocast(
            encoder.model.device.type,
            dtype=getattr(torch, args.dtype),
            enabled=args.dtype != "float32",
        ):
            query_features = encoder.forward_queries(
                [item["reference_image"] for item in items],
                [item["caption"] for item in items],
            )
            target_features = encoder.forward_images(
                [item["target_image"] for item in items]
            )
            loss = contrastive_loss(
                query_features, target_features, logit_scale=args.logit_scale
            )
            group_start = (
                (batch_index - 1) // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            accumulation_size = min(
                args.gradient_accumulation_steps, len(train_loader) - group_start
            )
            scaled_loss = loss / accumulation_size

        scaler.scale(scaled_loss).backward()
        running_loss += loss.detach().item()

        if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(
            train_loader
        ):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        progress.set_postfix(
            loss=f"{running_loss / batch_index:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    return running_loss / len(train_loader)


def train(args) -> None:
    set_seed(args.seed)
    encoder = QwenLoraEncoder(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        query_instruction=args.query_instruction,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainable, total = encoder.trainable_parameter_counts()
    print(
        f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)"
    )

    train_dataset = CIRRDataset("train", "relative", args.cirr_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_items,
    )

    optimizer = AdamW(
        (parameter for parameter in encoder.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=encoder.model.device.type == "cuda" and args.dtype == "float16"
    )

    metadata = vars(args).copy()
    metadata["cirr_root"] = str(metadata["cirr_root"])
    metadata["output_dir"] = str(metadata["output_dir"])
    metadata["effective_batch_size"] = (
        args.batch_size * args.gradient_accumulation_steps
    )
    best_score = -math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            encoder, train_loader, optimizer, scheduler, scaler, args, epoch
        )
        metrics = evaluate_cirr(encoder, args.cirr_root, args.eval_batch_size)
        selection_score = (metrics["recall_at5"] + metrics["group_recall_at1"]) / 2
        epoch_result = {
            "epoch": epoch,
            "global_step": epoch * updates_per_epoch,
            "train_loss": train_loss,
            "selection_score": selection_score,
            **metrics,
        }
        print(json.dumps(epoch_result, indent=2))

        checkpoint_metadata = {**metadata, **epoch_result}
        encoder.save_adapter(args.output_dir / "last", checkpoint_metadata)
        if selection_score > best_score:
            best_score = selection_score
            encoder.save_adapter(args.output_dir / "best", checkpoint_metadata)


def parse_args():
    parser = ArgumentParser(description="LoRA fine-tuning Qwen3-VL-Embedding on CIRR")
    add_model_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models/qwen3_vl_embedding_2b_cirr_lora",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing",
        action=BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

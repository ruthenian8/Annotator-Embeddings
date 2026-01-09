#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from src.dataset import DataModule  # noqa: E402
from src.transformer_models.encoder_module import EncoderModule  # noqa: E402
from src.utils.utils import Tasks, set_up_tokenizers  # noqa: E402


MODEL_TYPE = "deberta-multichoice"
MODEL_NAME_OR_PATH = "microsoft/deberta-v3-base"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect dataset cartography statistics for DeBERTa multi-choice models."
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--annotator_id_path", required=True)
    parser.add_argument("--annotation_label_path", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--method", choices=["add", "concat"], default="add")
    parser.add_argument("--include_pad_annotation", action="store_true")
    parser.add_argument("--broadcast_annotator_embedding", action="store_true")
    parser.add_argument("--broadcast_annotation_embedding", action="store_true")
    parser.add_argument("--use_naiive_concat", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="cartography_deberta_multichoice.csv")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.num_train_epochs * args.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def iter_train_batches(train_loader):
    if isinstance(train_loader, dict):
        for task_name, loader in train_loader.items():
            for batch in loader:
                yield task_name, batch
    else:
        for batch in train_loader:
            yield None, batch


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    args.model_type = MODEL_TYPE
    args.model_name_or_path = MODEL_NAME_OR_PATH
    args.use_annotator_embed = True
    args.use_annotation_embed = True
    args.training_paradigm = "learn_from_scratch"
    args.test_mode = "normal"
    args.pad_token_id = 0
    args.add_output_tokens = [True for _ in args.tasks]
    args.eval_batch_size = args.train_batch_size

    with open(args.annotator_id_path, "r") as f:
        annotator_ids = json.load(f)
    args.num_annotators = len(annotator_ids)

    with open(args.annotation_label_path, "r") as f:
        annotation_labels = json.load(f)
    args.num_labels = len(annotation_labels)

    tasks = Tasks(args.tasks)
    args.tasks = tasks

    encoder_tokenizer, decoder_tokenizers, _ = set_up_tokenizers(args, annotation_labels=annotation_labels)

    data_module = DataModule(
        args,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizers,
        tasks=tasks,
        annotator_id_path=args.annotator_id_path,
        annotation_label_path=args.annotation_label_path,
        use_naiive_concat=args.use_naiive_concat,
    )
    train_loader = data_module.train_dataloader()

    model = EncoderModule(decoder_tokenizer=decoder_tokenizers, **args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    steps_per_epoch = 0
    if isinstance(train_loader, dict):
        steps_per_epoch = sum(len(loader) for loader in train_loader.values())
    else:
        steps_per_epoch = len(train_loader)
    args.steps_per_epoch = steps_per_epoch

    optimizer, scheduler = build_optimizer(model, args)
    loss_fn = torch.nn.CrossEntropyLoss()

    task_list = list(tasks)
    if len(task_list) != 1:
        raise ValueError("This script currently supports a single task at a time.")
    task = task_list[0]
    label_names = decoder_tokenizers[task].labels

    rows = []

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        for _, batch in iter_train_batches(train_loader):
            optimizer.zero_grad()
            input_ids = batch["question_ids"].to(device)
            attention_mask = batch["question_mask"].to(device)
            annotator_ids = batch["annotator_id"].to(device)
            annotations = batch["annotations"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
                annotator_ids=annotator_ids,
                annotations=annotations,
            )

            loss = loss_fn(logits.view(-1, decoder_tokenizers[task].num_labels), answer_ids)
            loss.backward()
            optimizer.step()
            scheduler.step()

            probs = torch.softmax(logits, dim=-1)
            data_ids = batch["data_id"]

            for idx, data_id in enumerate(data_ids):
                row = {
                    "data_id": data_id,
                    "epoch": epoch,
                    "gold_label_id": int(answer_ids[idx].detach().cpu().item()),
                    "gold_label": decoder_tokenizers[task].id2label(int(answer_ids[idx].detach().cpu().item())),
                }
                for label_idx, _ in enumerate(label_names):
                    row[f"label_{label_idx}"] = float(probs[idx, label_idx].detach().cpu().item())
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved cartography statistics to {args.output_csv}")


if __name__ == "__main__":
    main()

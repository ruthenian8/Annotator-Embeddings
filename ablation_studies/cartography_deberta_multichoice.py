#!/usr/bin/env python
"""Train a DeBERTa multi-choice model and export dataset cartography statistics.

This script fine-tunes a DeBERTa-based multi-choice encoder on the provided
training data while tracking example-level training dynamics (often referred
to as "dataset cartography" statistics). It then writes these statistics to a
CSV file for downstream analysis.

Typical usage (from the repository root):

    python -m ablation_studies.cartography_deberta_multichoice \\
        --train_data_path PATH/TO/train.jsonl \\
        --annotator_id_path PATH/TO/annotator_ids.jsonl \\
        --annotation_label_path PATH/TO/annotation_labels.jsonl \\
        --tasks TASK1 TASK2 ... \\
        --output_csv cartography_deberta_multichoice.csv

Required inputs:
  * --train_data_path: Path to the training set used for multi-choice
    modeling. The expected format must match the `DataModule` implementation
    in `src.dataset`.
  * --annotator_id_path: Path to per-example annotator identifiers.
  * --annotation_label_path: Path to per-example annotation labels.
  * --tasks: One or more task names (see `Tasks` in `src.utils.utils`).

Additional flags control training hyperparameters (batch size, number of
epochs, learning rate, weight decay, warmup steps, sequence length, etc.) and
how annotator / annotation embeddings are incorporated.

Output:
  The script writes a CSV file to the location provided by `--output_csv`
  (default: "cartography_deberta_multichoice.csv"). Each row corresponds to
  an example observed during training and includes cartography-related
  metrics such as training epoch/step, model confidence, correctness, and
  related statistics, enabling analysis of example difficulty and stability
  across epochs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterator, Optional

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.add_argument(
        "--use_naive_concat",
        "--use_naiive_concat",
        dest="use_naive_concat",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="cartography_deberta_multichoice.csv")
    args = parser.parse_args()

    # Validate that required path arguments point to existing files.
    if not Path(args.train_data_path).is_file():
        parser.error(f"--train_data_path does not exist or is not a file: {args.train_data_path}")
    if not Path(args.annotator_id_path).is_file():
        parser.error(f"--annotator_id_path does not exist or is not a file: {args.annotator_id_path}")
    if not Path(args.annotation_label_path).is_file():
        parser.error(
            f"--annotation_label_path does not exist or is not a file: {args.annotation_label_path}"
        )

    return args


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace, steps_per_epoch: int):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.num_train_epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def iter_train_batches(train_loader) -> Iterator[tuple[Optional[str], dict]]:
    """Iterate over training batches from single or multi-task data loader.

    Args:
        train_loader: Either a DataLoader or a dict of task names to DataLoaders.

    Yields:
        A tuple of (task_name, batch) where task_name is None for single-task
        loaders or the task name string for multi-task loaders.
    """
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
    args.embed_wo_weight = False
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
        use_naiive_concat=args.use_naive_concat,
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

    optimizer, scheduler = build_optimizer(model, args, steps_per_epoch)
    loss_fn = torch.nn.CrossEntropyLoss()

    task_list = list(tasks)
    if len(task_list) != 1:
        raise ValueError("This script currently supports a single task at a time.")
    task = task_list[0]
    label_names = decoder_tokenizers[task].labels

    rows = []

    for epoch in range(1, args.num_train_epochs + 1):
        # Training phase
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

            loss = loss_fn(
                logits.view(-1, decoder_tokenizers[task].num_labels),
                answer_ids.view(-1),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        # After training for this epoch, collect cartography statistics in eval mode
        model.eval()
        with torch.no_grad():
            for _, batch in iter_train_batches(train_loader):
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

                probs = torch.softmax(logits, dim=-1)
                data_ids = batch["data_id"]

                for idx, data_id in enumerate(data_ids):
                    # Ensure data_id is converted to native Python types for safe DataFrame/CSV usage
                    data_id_value = data_id
                    if isinstance(data_id_value, torch.Tensor):
                        data_id_value = data_id_value.detach().cpu()
                        if data_id_value.dim() == 0 or data_id_value.numel() == 1:
                            data_id_value = data_id_value.item()
                        else:
                            data_id_value = data_id_value.tolist()
                    elif isinstance(data_id_value, np.ndarray):
                        data_id_value = data_id_value.tolist()
                    elif isinstance(data_id_value, np.generic):
                        data_id_value = data_id_value.item()

                    gold_label_id = int(answer_ids[idx].detach().cpu().item())
                    row = {
                        "data_id": data_id_value,
                        "epoch": epoch,
                        "gold_label_id": gold_label_id,
                        "gold_label": decoder_tokenizers[task].id2label(gold_label_id),
                    }
                    for label_idx, _ in enumerate(label_names):
                        row[f"label_{label_idx}"] = float(
                            probs[idx, label_idx].detach().cpu().item()
                        )
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved cartography statistics to {args.output_csv}")


if __name__ == "__main__":
    main()

"""Entry point for GRU4Rec training with PyTorch Lightning (local or Colab)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.colab.setup import prepare_colab_session
from src.config.train_config import TrainConfig
from src.data.gru4rec import GRU4RecDataset, gru4rec_collate_fn
from src.data.meta import gru4rec_vocab_size, load_meta, split_examples_path
from src.evaluation.baselines import evaluate_pop_baseline, popularity_top_k_gru_indices
from src.training.lit_modules.gru4rec import GRU4RecLitModule
from src.training.trainer_factory import build_lightning_trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_path = split_examples_path(config.processed_dir, "train", "gru4rec")
    val_path = split_examples_path(config.processed_dir, "val", "gru4rec")
    pin = config.device.startswith("cuda") and torch.cuda.is_available()

    train_loader = DataLoader(
        GRU4RecDataset(train_path),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=gru4rec_collate_fn,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        GRU4RecDataset(val_path),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=gru4rec_collate_fn,
        pin_memory=pin,
    )
    return train_loader, val_loader


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GRU4Rec on processed Yoochoose data.")
    parser.add_argument(
        "--drive-project-dir",
        type=Path,
        default=None,
        help="Google Drive project root (contains data/processed.zip). Enables Colab setup.",
    )
    parser.add_argument(
        "--processed-variant",
        type=str,
        default="subsample_1_32_clicks_only",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser


def train_gru4rec(config: TrainConfig) -> None:
    config = prepare_colab_session(config)
    set_seed(config.seed)

    meta = load_meta(config.processed_dir)
    num_embeddings = gru4rec_vocab_size(meta)
    train_loader, val_loader = build_dataloaders(config)

    pop_indices = popularity_top_k_gru_indices(config.processed_dir, k=20)
    pop_baseline_metrics = evaluate_pop_baseline(val_loader, pop_indices, ks=(5, 10, 20))

    module = GRU4RecLitModule(
        num_embeddings=num_embeddings,
        learning_rate=config.learning_rate,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pop_baseline_metrics=pop_baseline_metrics,
    )

    trainer = build_lightning_trainer(config)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.drive_project_dir is not None:
        config = TrainConfig.for_colab(
            args.drive_project_dir,
            processed_variant=args.processed_variant,
            wandb_run_name=args.run_name,
            run_name=args.run_name,
        )
    else:
        config = TrainConfig(
            processed_variant=args.processed_variant,
            run_name=args.run_name,
            wandb_run_name=args.run_name,
        )

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.no_wandb:
        config.wandb_enabled = False
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity

    train_gru4rec(config)


if __name__ == "__main__":
    main()

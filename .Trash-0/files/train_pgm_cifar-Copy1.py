#train_pgm_cifar.py

#!/usr/bin/env python3
import os
import sqlite3
import argparse
import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

from torch.utils.data import DataLoader
from tqdm import tqdm


# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser(description="PGM Ex4: CIFAR-10 ResNet baseline")
    # experiment metadata
    p.add_argument("--exp_name", type=str, default="cifar10_baseline")
    p.add_argument("--env_id", type=str, default="CIFAR10-ResNet18")  # keeps RL-style naming
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--notes", type=str, default="")

    # training core
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--cosine", action="store_true", help="use cosine LR schedule")
    p.add_argument("--warmup_epochs", type=int, default=0)

    # augmentation
    p.add_argument("--augment", type=str, default="basic", choices=["basic", "auto", "rand"])

    # paths / db
    p.add_argument("--runs_dir", type=str, default=os.path.join("..", "runs"))
    p.add_argument("--db_name", type=str, default="pgm.db")
    p.add_argument("--data_dir", type=str, default=os.path.join("..", "data"))

    # eval
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# -------------------- Utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep fast kernels
    torch.backends.cudnn.benchmark = True


def db_connect(db_path):
    conn = sqlite3.connect(db_path, timeout=30.0)
    # keep same schema as RL runs table for compatibility
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        exp_name TEXT,
        env_id TEXT,
        seed INTEGER,
        steps INTEGER,
        mean_reward REAL,
        std_reward REAL,
        lr REAL,
        gamma REAL,
        clip_range REAL,
        batch_size INTEGER,
        n_steps INTEGER,
        device TEXT,
        notes TEXT,
        extra TEXT
    );
    """)
    conn.commit()
    return conn


def log_run(conn, row):
    cols = ",".join(row.keys())
    qmarks = ",".join(["?"] * len(row))
    conn.execute(f"INSERT INTO runs ({cols}) VALUES ({qmarks})", list(row.values()))
    conn.commit()
    cur = conn.execute("SELECT last_insert_rowid();")
    return cur.fetchone()[0]


# -------------------- Data --------------------
def build_transforms(kind="basic"):
    # CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if kind == "basic":
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif kind == "auto":
        train_tf = T.Compose([
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:  # "rand"
        train_tf = T.Compose([
            T.RandAugment(num_ops=2, magnitude=9),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, test_tf


def get_loaders(data_dir, batch_size, num_workers, augment):
    train_tf, test_tf = build_transforms(augment)

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# -------------------- Model / Train --------------------
def build_model(device):
    model = resnet18(num_classes=10)
    # ResNet expects 224x224 by default; CIFAR-10 uses 32x32.
    # resnet18 still works fine on 32x32 inputs (conv1 is 7x7 stride 2 → okay for CIFAR).
    return model.to(device)


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model, loader, device):
    model.eval()
    accs = []
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            accs.append(accuracy(logits, targets))
            losses.append(loss.item())
    mean_acc = float(np.mean(accs)) * 100.0
    std_acc = float(np.std(accs)) * 100.0
    mean_loss = float(np.mean(losses))
    return mean_acc, std_acc, mean_loss


def main():
    args = parse_args()
    set_seed(args.seed)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # paths & db
    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    db_path = os.path.join(args.runs_dir, args.db_name)
    conn = db_connect(db_path)

    # data
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers, args.augment)

    # model & optim
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    else:
        scheduler = None

    # optional warmup
    def lr_lambda(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        return 1.0

    warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) if args.warmup_epochs > 0 else None

    # train
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        if warmup is not None and epoch < args.warmup_epochs:
            warmup.step()
        elif scheduler is not None:
            scheduler.step()

    train_time = time.time() - start

    # evaluate on test set
    test_acc, test_acc_std, test_loss = evaluate(model, test_loader, device)

    # log (reuse RL schema: mean_reward := accuracy, std_reward := per-batch acc std)
    row = dict(
        ts=int(time.time()),
        exp_name=args.exp_name,
        env_id=args.env_id,
        seed=args.seed,
        steps=args.epochs,                 # epochs as "steps" to stay consistent
        mean_reward=float(test_acc),       # Top-1 test accuracy (%)
        std_reward=float(test_acc_std),    # std across test batches (%)
        lr=float(args.lr),
        gamma=float("nan"),                # not used here
        clip_range=float("nan"),           # not used here
        batch_size=int(args.batch_size),
        n_steps=int(args.epochs),          # epochs again for PGM compatibility
        device=device + (f":{torch.cuda.get_device_name(0)}" if device == "cuda" else ""),
        notes=args.notes,
        extra=json.dumps({
            "dataset": "CIFAR-10",
            "model": "resnet18",
            "augment": args.augment,
            "optimizer": "SGD",
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "cosine": args.cosine,
            "warmup_epochs": args.warmup_epochs,
            "test_loss": test_loss,
            "train_time_sec": train_time
        })
    )
    run_id = log_run(conn, row)
    conn.close()

    print(f"OK: logged run_id={run_id}, exp={args.exp_name}, seed={args.seed}, acc={test_acc:.2f}% ± {test_acc_std:.2f}%")


if __name__ == "__main__":
    # keep pygame banner away if SDL present on the box
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
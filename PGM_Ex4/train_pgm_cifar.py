#!/usr/bin/env python3
import os, sqlite3, argparse, time, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Ex4 – CIFAR-10 single run (baseline/proposal)")
    # exp/meta
    p.add_argument("--exp_name", type=str, default="cifar10_baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--runs_dir", type=str, default="../runs")
    p.add_argument("--db_name", type=str, default="pgm.db")
    p.add_argument("--data_dir", type=str, default="../data")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--augment", type=str, default="basic", choices=["basic","none"])
    p.add_argument("--cosine", action="store_true")
    # dataloader/system
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def db_connect(db_path):
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs(
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


def make_loaders(data_dir, batch_size, augment, num_workers):
    normalize = T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    if augment == "basic":
        train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), normalize])
    else:
        train_tf = T.Compose([T.ToTensor(), normalize])
    test_tf = T.Compose([T.ToTensor(), normalize])

    train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=(num_workers>0),
                              prefetch_factor=2 if num_workers>0 else None)
    test_loader  = DataLoader(test, batch_size=512, shuffle=False,
                              num_workers=max(0, num_workers//2), pin_memory=pin,
                              persistent_workers=(num_workers//2>0),
                              prefetch_factor=2 if num_workers//2>0 else None)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    corr, tot = 0, 0
    for x,y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        corr += (pred==y).sum().item()
        tot  += y.numel()
    acc = 100.0 * corr / tot
    std = 0.0  # schema compat (std_reward)
    return acc, std


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device / CuDNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # IO
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.runs_dir, args.db_name)
    conn = db_connect(db_path)

    # Data
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.augment, args.num_workers)

    # Model
    model = resnet18(num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None

    # Train
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x,y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if scheduler: scheduler.step()
    train_time = time.time() - start

    # Eval
    acc, std = evaluate(model, test_loader, device)

    # Log row (env_id uses “CIFAR10”; mean_reward stores accuracy)
    row = dict(
        ts=int(time.time()),
        exp_name=args.exp_name,
        env_id="CIFAR10",
        seed=args.seed,
        steps=args.epochs,
        mean_reward=float(acc),
        std_reward=float(std),
        lr=float(args.lr),
        gamma=None,
        clip_range=None,
        batch_size=int(args.batch_size),
        n_steps=None,
        device=(str(device) + (f":{torch.cuda.get_device_name(0)}" if device.type=="cuda" else "")),
        notes=args.notes,
        extra=json.dumps({
            "augment": args.augment,
            "cosine": bool(args.cosine),
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "train_time_sec": train_time
        })
    )
    cols = ",".join(row.keys()); q = ",".join(["?"]*len(row))
    conn.execute(f"INSERT INTO runs({cols}) VALUES({q})", list(row.values()))
    conn.commit()
    run_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
    conn.close()

    print(f"OK: logged run_id={run_id}, exp={args.exp_name}, seed={args.seed}, acc={acc:.2f}% ± {std:.2f}%")

if __name__ == "__main__":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    main()
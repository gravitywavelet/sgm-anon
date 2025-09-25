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
import numpy as np

# ----------------- DB helpers -----------------
def ensure_runs_table(conn):
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
    conn.execute("""CREATE INDEX IF NOT EXISTS runs_exp_seed_steps
                    ON runs(exp_name, seed, steps, ts);""")
    conn.commit()

def ensure_columns(conn, table, row_dict):
    import numbers
    cur = conn.execute(f'PRAGMA table_info({table})')
    have = {r[1] for r in cur.fetchall()}
    added = []
    for k, v in row_dict.items():
        if k not in have:
            # best-effort type
            t = "TEXT"
            try:
                if isinstance(v, (bool, np.bool_, int, np.integer)): t = "INTEGER"
                elif isinstance(v, (float, np.floating)): t = "REAL"
            except Exception: pass
            conn.execute(f'ALTER TABLE {table} ADD COLUMN {k} {t}')
            added.append((k, t))
    if added:
        conn.commit()

# ----------------- Args -----------------
def parse_args():
    p = argparse.ArgumentParser("CIFAR-10/100 trainer with label smoothing for SGM demos")
    # exp/meta
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--runs_dir", type=str, required=True)
    p.add_argument("--db_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    # dataset
    p.add_argument("--dataset", type=str, choices=["cifar10","cifar100"], required=True)
    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine","none"])
    p.add_argument("--warmup_epochs", type=int, default=0)  # ignored (kept for compatibility)
    # dataloader/system
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()

# ----------------- Utils -----------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_loaders(dataset, data_dir, batch_size, num_workers):
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        Train = torchvision.datasets.CIFAR10
        Test  = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        Train = torchvision.datasets.CIFAR100
        Test  = torchvision.datasets.CIFAR100
        num_classes = 100

    normalize = T.Normalize(mean, std)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf  = T.Compose([T.ToTensor(), normalize])

    train = Train(root=data_dir, train=True,  download=True, transform=train_tf)
    test  = Test (root=data_dir, train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=(num_workers>0))
    test_loader  = DataLoader(test, batch_size=512, shuffle=False,
                              num_workers=max(0, num_workers//2), pin_memory=pin,
                              persistent_workers=(num_workers//2>0))
    return train_loader, test_loader, num_classes

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    corr, tot = 0, 0
    for x,y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x); pred = logits.argmax(1)
        corr += (pred==y).sum().item(); tot += y.numel()
    return 100.0 * corr / tot

# ----------------- Main -----------------
def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # IO
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.runs_dir, args.db_name)
    conn = sqlite3.connect(db_path, timeout=60.0)
    ensure_runs_table(conn)

    # Data & model
    train_loader, test_loader, num_classes = make_loaders(args.dataset, args.data_dir, args.batch_size, args.num_workers)
    model = resnet18(num_classes=num_classes).to(device)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.sched=="cosine" else None
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Train
    t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        for x,y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if scheduler: scheduler.step()
    train_time = time.time() - t0

    # Eval
    acc = evaluate(model, test_loader, device)

    # Log row
    row = dict(
        ts=int(time.time()),
        exp_name=args.exp_name,
        env_id=args.dataset.upper(),
        seed=int(args.seed),
        steps=int(args.epochs),
        mean_reward=float(acc),
        std_reward=0.0,
        lr=float(args.lr),
        gamma=None, clip_range=None,
        batch_size=int(args.batch_size),
        n_steps=None,
        device=str(device) + (f":{torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""),
        notes=args.notes,
        extra=json.dumps({
            "dataset": args.dataset,
            "label_smoothing": float(args.label_smoothing),
            "weight_decay": float(args.weight_decay),
            "sched": args.sched,
            "train_time_sec": train_time
        })
    )
    ensure_columns(conn, "runs", row)
    cols = ",".join(row.keys()); q = ",".join(["?"]*len(row))
    conn.execute(f"INSERT INTO runs({cols}) VALUES({q})", list(row.values()))
    conn.commit()
    run_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
    conn.close()
    print(f"OK: run_id={run_id}, exp={args.exp_name}, seed={args.seed}, acc={acc:.2f}% (ls={args.label_smoothing})")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse, os, time, sqlite3, json, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision as tv
import torchvision.transforms as T


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_tables(db_path: str):
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER, exp_name TEXT, env_id TEXT, seed INTEGER, steps INTEGER,
      mean_reward REAL, std_reward REAL, lr REAL, gamma REAL, clip_range REAL,
      batch_size INTEGER, n_steps INTEGER, device TEXT, notes TEXT, extra TEXT
    );""")
    conn.execute("CREATE INDEX IF NOT EXISTS runs_exp_seed_steps ON runs(exp_name, seed, steps, ts);")
    conn.commit()
    conn.close()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def get_loaders(data_dir, batch_size, num_workers):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train = tv.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_tf)
    test  = tv.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test,  batch_size=512, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def main():
    ap = argparse.ArgumentParser("CIFAR-10 trainer (ICLR mini stress-test friendly)")
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--runs_dir", default="../runs")
    ap.add_argument("--db_name", default="pgm_cifar10.db")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--num_workers", type=int, default=4)

    # recipe knobs (kept minimal & transparent)
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--warmup_epochs", type=int, default=0)

    args = ap.parse_args()

    # filesystem
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.runs_dir, args.db_name)
    ensure_tables(db_path)

    # seed & device
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    # model
    model = tv.models.resnet18(num_classes=10)
    model.to(device)

    # loss, opt, sched
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.sched == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = None

    # warmup (simple linear)
    def lr_warmup(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            for pg in opt.param_groups:
                pg["lr"] = args.lr * float(epoch + 1) / float(args.warmup_epochs)

    # train
    start = time.time()
    for ep in range(args.epochs):
        model.train()
        lr_warmup(ep)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        if sched is not None and ep >= args.warmup_epochs:
            sched.step()

    test_acc = evaluate(model, test_loader, device)

    # log to sqlite (match your schema)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("""
        INSERT INTO runs(ts, exp_name, env_id, seed, steps, mean_reward, std_reward,
                         lr, gamma, clip_range, batch_size, n_steps, device, notes, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(time.time()),
        args.exp_name,
        "cifar10",
        int(args.seed),
        int(args.epochs),
        float(test_acc),  # mean_reward = accuracy %
        0.0,              # std_reward (NA here)
        float(args.lr),
        0.0, 0.0,
        int(args.batch_size),
        0,
        device,
        "",
        json.dumps({
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "sched": args.sched,
            "warmup_epochs": args.warmup_epochs
        })
    ))
    conn.commit()
    conn.close()

    dur = (time.time() - start) / 60.0
    print(f"[CIFAR10] seed={args.seed} epochs={args.epochs} acc={test_acc:.2f}% time={dur:.1f}m")


if __name__ == "__main__":
    main()
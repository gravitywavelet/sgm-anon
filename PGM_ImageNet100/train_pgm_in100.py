#!/usr/bin/env python3
import argparse, os, json, time, math, random, sqlite3
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform, resolve_data_config, Mixup
from timm.utils import ModelEmaV2
from timm.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
from torchvision import datasets
from timm.scheduler import CosineLRScheduler

# robust timm loss import (falls back to PyTorch if absent)
try:
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    _HAS_TIMM_LOSS = True
except Exception:
    LabelSmoothingCrossEntropy = None
    SoftTargetCrossEntropy = None
    _HAS_TIMM_LOSS = False

# ---------- DB ----------
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
    );""")
    conn.execute("CREATE INDEX IF NOT EXISTS runs_exp_seed_steps ON runs(exp_name, seed, steps, ts);")
    conn.commit()

def log_row(db_path, row):
    conn = sqlite3.connect(db_path, timeout=60.0)
    ensure_runs_table(conn)
    cols = ",".join(row.keys()); q = ",".join(["?"] * len(row))
    conn.execute(f"INSERT INTO runs({cols}) VALUES({q})", list(row.values()))
    conn.commit(); conn.close()

# ---------- Utils ----------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x,y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total

def main():
    ap = argparse.ArgumentParser("IN-100 + DeiT-S (timm) – PGM trainer")
    # meta
    ap.add_argument("--exp_name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs_dir", type=str, required=True)
    ap.add_argument("--db_name", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=8)
    # training
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_effective", type=int, default=512)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--model", type=str, default="deit_small_patch16_224")
    ap.add_argument("--optimizer", type=str, default="adamw")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--sched", type=str, default="cosine")
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--randaug_m", type=int, default=9)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.0, help="0 to disable; else decay, e.g., 0.9999")
    args = ap.parse_args()

    # device & seed
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"): torch.set_float32_matmul_precision("high")

    # DATA — assumes ImageNet style dir with 100 classes in train/ and val/
    # You should prepare ImageNet-100 with the usual structure:
    # data_dir/train/<class>/*.jpeg, data_dir/val/<class>/*.jpeg
    train_tf = transforms_imagenet_train(
        img_size=args.img_size,
        auto_augment=f"rand-m{args.randaug_m}-mstd0.5",
        interpolation="bicubic",
        mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    val_tf = transforms_imagenet_eval(
        img_size=args.img_size,
        crop_pct=0.875,
        mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tf)

    # choose per-GPU batch that fits, and compute grad-accum for effective 512
    # simple heuristic: try 128, then fallback
    per_gpu = 128
    grad_accum = max(1, math.ceil(args.batch_effective / per_gpu))

    train_loader = DataLoader(
        train_ds, batch_size=per_gpu, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=(args.num_workers>0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=max(0, args.num_workers//2), pin_memory=True, persistent_workers=(args.num_workers//2>0)
    )

    # MODEL
    model = timm.create_model(
        args.model, pretrained=False, num_classes=100, drop_path_rate=args.drop_path
    ).to(device)

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=100,
        drop_path_rate=args.drop_path,
        img_size=args.img_size,     # <- let ViT expect 192 for screen, 224 for confirm
    ).to(device)

    use_soft_targets = (getattr(args, "mixup", 0.0) > 0) or (getattr(args, "cutmix", 0.0) > 0)
    ls = float(getattr(args, "label_smoothing", 0.0))

    if use_soft_targets and _HAS_TIMM_LOSS and SoftTargetCrossEntropy is not None:
        criterion = SoftTargetCrossEntropy()
    elif ls > 0:
        # Prefer native torch if available
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        except TypeError:
            # fallback to timm if torch is too old
            if _HAS_TIMM_LOSS and LabelSmoothingCrossEntropy is not None:
                criterion = LabelSmoothingCrossEntropy(smoothing=ls)
            else:
                criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    mixup_fn = None
    if args.mixup > 0.0 or args.cutmix > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup if args.mixup>0 else 0.0,
            cutmix_alpha=args.cutmix if args.cutmix>0 else 0.0,
            label_smoothing=args.label_smoothing, num_classes=100
        )

    # OPTIM / SCHED
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.999))
    else:
        raise ValueError("Only AdamW intended for this recipe.")

    if args.sched == "cosine":
        lr_sched = CosineLRScheduler(
            optimizer, t_initial=args.epochs, lr_min=args.lr*1e-2, warmup_t=args.warmup_epochs, warmup_lr_init=args.lr*0.1
        )
    else:
        lr_sched = None

    # EMA
    ema = None
    if args.ema and args.ema > 0:
        ema = ModelEmaV2(model, decay=args.ema)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    # TRAIN
    iters_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
    
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
    
            scaler.scale(loss).backward()
    
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
    
        if lr_sched is not None:
            lr_sched.step(epoch + 1)



    # EVAL (EMA if present)
    eval_model = ema.module if ema is not None else model
    top1 = evaluate(eval_model, val_loader, device)

    # LOG
    db_path = os.path.join(args.runs_dir, args.db_name)
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    row = dict(
        ts=int(time.time()),
        exp_name=args.exp_name,
        env_id="ImageNet-100",
        seed=int(args.seed),
        steps=int(args.epochs),
        mean_reward=float(top1),
        std_reward=0.0,
        lr=float(args.lr),
        gamma=None, clip_range=None,
        batch_size=int(args.batch_effective),
        n_steps=None,
        device=str(device) + (f":{torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""),
        notes="timm-deitS",
        extra=json.dumps(dict(
            model=args.model, wd=args.weight_decay, sched=args.sched, warmup=args.warmup_epochs,
            drop_path=args.drop_path, randaug_m=args.randaug_m, mixup=args.mixup, cutmix=args.cutmix,
            ls=args.label_smoothing, ema=args.ema, per_gpu=per_gpu, grad_accum=grad_accum
        ))
    )
    log_row(db_path, row)
    print(f"OK: {args.exp_name} seed={args.seed} top1={top1:.2f}%")

if __name__ == "__main__":
    main()
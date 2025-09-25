#!/usr/bin/env python3
# Ex5 – PGM outer loop for CIFAR-100 (multi-knob proposals + two-stage PAC gate)

import argparse, json, math, os, sqlite3, subprocess, time
from pathlib import Path
import torch
from copy import deepcopy
import random, zlib  # single import set

# ---------- utils ----------
def cfg_hash(cfg: dict) -> str:
    return format(zlib.adler32(json.dumps(cfg, sort_keys=True).encode()), "08x")

def _clip(x, lo, hi): 
    return float(min(hi, max(lo, x)))

def _maybe_add(cfg, key, default):
    if key not in cfg:
        cfg[key] = default

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return v

def _trust_region_lr(old_lr, mult, lo=1e-5, hi=1.0, max_ratio=1.6):
    # clamp multiplicative change within [1/max_ratio, max_ratio]
    if mult <= 0:
        mult = 1.0
    mult = _clamp(mult, 1.0 / max_ratio, max_ratio)
    new_lr = _clamp(_to_float(old_lr) * mult, lo, hi)
    return new_lr

def _scale_lr_for_batch(old_bs, new_bs, old_lr, max_ratio=1.6):
    # linear scaling rule + trust-region clamp
    if old_bs <= 0:
        return old_lr
    scale = new_bs / float(old_bs)
    return _trust_region_lr(old_lr, scale, max_ratio=max_ratio)

def propose(cfg, max_changes=2, strong_moves=True, seed=None):
    """
    Make a small, safe proposal around cfg for CIFAR-100 short training runs.

    - Favors low-risk moves: cosine + warmup, label smoothing, weight decay, EMA, mild mixup/cutmix.
    - Uses an LR trust-region (×/÷1.6) to avoid cliffs.
    - Includes 'PAIR' (batch_size + scaled LR) but blocks it in early iterations and never repeats back-to-back.
    - Optionally adjusts momentum and optimizer (downweighted early).
    - Persists `_iter` and `_last_move_tag` in the config so behavior stays stateful across iterations.
    """
    if seed is not None:
        random.seed(seed)

    new = deepcopy(cfg)
    changed = []

    # ---------- ensure optional knobs exist (trainer can ignore unknown keys safely) ----------
    _maybe_add(new, "optimizer", new.get("optimizer", "SGD"))           # "SGD" | "SGD_NAG" | "AdamW"
    _maybe_add(new, "lr_schedule", new.get("lr_schedule", "step"))      # "step" | "cosine"
    _maybe_add(new, "warmup_epochs", new.get("warmup_epochs", 0))
    _maybe_add(new, "label_smoothing", new.get("label_smoothing", 0.0))
    _maybe_add(new, "weight_decay", new.get("weight_decay", 5e-4))
    _maybe_add(new, "ema_decay", new.get("ema_decay", 0.0))
    _maybe_add(new, "mixup", new.get("mixup", 0.0))
    _maybe_add(new, "cutmix", new.get("cutmix", 0.0))
    _maybe_add(new, "grad_clip", new.get("grad_clip", 0.0))
    _maybe_add(new, "momentum", new.get("momentum", 0.9))               # used for SGD variants
    _maybe_add(new, "batch_size", new.get("batch_size", 128))
    _maybe_add(new, "lr", new.get("lr", 0.1))                           # typical SGD baseline

    cur_iter = int(new.get("_iter", 0))
    last_tag = new.get("_last_move_tag", None)

    # ---------- define atomic moves ----------
    def move_lr_schedule_cosine():
        if new.get("lr_schedule") != "cosine":
            new["lr_schedule"] = "cosine"
            changed.append(("lr_schedule", "cosine"))

    def move_warmup():
        choice = random.choice([3, 5])  # short warmup fits short epochs
        if new.get("warmup_epochs") != choice:
            new["warmup_epochs"] = choice
            changed.append(("warmup_epochs", choice))

    def move_label_smoothing():
        choice = random.choice([0.05, 0.1])
        if abs(new.get("label_smoothing", 0.0) - choice) > 1e-9:
            new["label_smoothing"] = choice
            changed.append(("label_smoothing", choice))

    def move_weight_decay():
        choice = random.choice([2e-4, 5e-4, 1e-3])
        if abs(new.get("weight_decay", 5e-4) - choice) > 1e-12:
            new["weight_decay"] = choice
            changed.append(("weight_decay", choice))

    def move_ema():
        choice = random.choice([0.99, 0.999])
        if abs(new.get("ema_decay", 0.0) - choice) > 1e-12:
            new["ema_decay"] = choice
            changed.append(("ema_decay", choice))

    def move_mixup_cutmix():
        # mild regularization only (larger values often hurt at short epochs)
        mix = random.choice([0.0, 0.1, 0.2])
        cut = random.choice([0.0, 0.1, 0.2])
        before = len(changed)
        if abs(new.get("mixup", 0.0) - mix) > 1e-9:
            new["mixup"] = mix
            changed.append(("mixup", mix))
        if abs(new.get("cutmix", 0.0) - cut) > 1e-9:
            new["cutmix"] = cut
            changed.append(("cutmix", cut))
        return len(changed) > before  # indicate something changed

    def move_optimizer():
        opt_now = new.get("optimizer", "SGD")
        choice = random.choice(["SGD", "SGD_NAG", "AdamW"])
        if choice != opt_now:
            new["optimizer"] = choice
            changed.append(("optimizer", choice))
            # If switching to AdamW, step LR down to a safe range.
            if choice == "AdamW":
                lr_old = _to_float(new.get("lr", 0.1))
                lr_new = _clamp(lr_old, 3e-4, 1e-3)
                if abs(lr_new - lr_old) > 1e-12:
                    new["lr"] = lr_new
                    changed.append(("lr", lr_new))

    def move_lr_trust_region():
        # gentle LR nudge within trust region
        lr_old = _to_float(new.get("lr", 0.1))
        mult = random.choice([0.8, 0.9, 1.1, 1.25, 1.6])
        lr_new = _trust_region_lr(lr_old, mult, lo=1e-5, hi=1.0, max_ratio=1.6)
        if abs(lr_new - lr_old) > 1e-12:
            new["lr"] = lr_new
            changed.append(("lr", lr_new))

    def move_momentum():
        # small, safe momentum tweaks
        m_old = _to_float(new.get("momentum", 0.9))
        choice = random.choice([0.88, 0.9, 0.92, 0.94])
        if abs(choice - m_old) > 1e-12:
            new["momentum"] = choice
            changed.append(("momentum", choice))

    def move_pair_batch_scaled_lr():
        # risky pair; do sparsely and never back-to-back
        bs_old = int(new.get("batch_size", 128))
        lr_old = _to_float(new.get("lr", 0.1))
        bs_choice = random.choice([64, 96, 128, 160, 192, 256, 320, 384, 448, 512])
        if bs_choice != bs_old:
            lr_scaled = _scale_lr_for_batch(bs_old, bs_choice, lr_old, max_ratio=1.6)
            new["batch_size"] = bs_choice
            new["lr"] = lr_scaled
            changed.append(("batch_size", bs_choice))
            changed.append(("lr(batch-scaled)", lr_scaled))
            return True
        return False

    # ---------- catalog moves (tag, function, weight) ----------
    MOVES = [
        ("cosine",   move_lr_schedule_cosine,  1.0),
        ("warmup",   move_warmup,              1.0),
        ("ls",       move_label_smoothing,     0.9),
        ("wd",       move_weight_decay,        0.9),
        ("ema",      move_ema,                 0.8),
        ("mixcut",   move_mixup_cutmix,        0.6),
        ("opt",      move_optimizer,           0.5),
        ("mom",      move_momentum,            0.4),
        ("lr_trust", move_lr_trust_region,     0.8),
        ("PAIR",     move_pair_batch_scaled_lr,0.25),  # risky; downweight
    ]

    # ---------- steering / guardrails ----------
    # Block risky stuff in early iters; never repeat PAIR back-to-back
    block_pair_early = (cur_iter < 4)
    block_aggr_lr_early = (cur_iter < 3)

    filtered = []
    for tag, fn, w in MOVES:
        if tag == "PAIR" and (block_pair_early or last_tag == "PAIR"):
            continue
        if block_aggr_lr_early and tag in ("opt", "mom"):
            # push optimizer/momentum moves later
            continue
        if block_aggr_lr_early and tag == "lr_trust":
            # keep gentle LR moves but heavily downweight
            filtered.append((tag, fn, 0.2))
            continue
        filtered.append((tag, fn, w))
    MOVES = filtered

    # Prefer safer moves early (cosine/warmup/LS/WD/EMA)
    PREF = {"cosine": 1.4, "warmup": 1.3, "ls": 1.2, "wd": 1.2, "ema": 1.1, "mixcut": 1.0}
    MOVES = [(t, f, w * PREF.get(t, 1.0)) for (t, f, w) in MOVES]

    # ---------- sample & apply 1–2 moves ----------
    n_changes = random.choice([1, 2]) if strong_moves else 1
    pool = list(MOVES)
    weights = [w for _, _, w in pool]

    used_tag = None
    for _ in range(n_changes):
        if not pool:
            break
        idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
        tag, fn, w = pool.pop(idx)
        weights.pop(idx)

        before = len(changed)
        ok = fn()
        after = len(changed)
        if ok or after > before:
            used_tag = tag

    # ---------- persist proposer state ----------
    new["_iter"] = cur_iter + 1
    new["_last_move_tag"] = used_tag if used_tag is not None else last_tag

    return new, changed

# ---------- PAC bound (empirical-Bernstein lower bound) ----------
def eb_lower_bound(y_list, delta, R=1.0):
    """
    y_list: list of paired diffs normalized to [-1,1] (already divided by rmax_pp and clipped)
    delta: per-iteration alpha spend δ_t
    R: sub-Gaussian range proxy for normalized y (keep R=1.0 since y∈[-1,1])
    Returns: (lower_bound, sample_mean)
    """
    n = len(y_list)
    mu = sum(y_list) / n
    s2 = sum((y - mu) ** 2 for y in y_list) / n
    term1 = (2.0 * s2 * math.log(3.0 / max(delta, 1e-12)) / n) ** 0.5
    term2 = (3.0 * R * math.log(3.0 / max(delta, 1e-12)) / n)
    return mu - term1 - term2, mu

def should_confirm(deltas_pp, mean_pp_threshold=0.8, require_all_pos=True):
    """Heuristic to trigger confirmatory re-run based on per-seed Δ in %-points."""
    if not deltas_pp:
        return False
    all_pos = all(d > 0 for d in deltas_pp)
    mean_pp = sum(deltas_pp) / len(deltas_pp)
    if require_all_pos and all_pos:
        return True
    return mean_pp >= mean_pp_threshold

def make_seed_set(base_seeds, target_n, pool_start=1001):
    """Extend base_seeds deterministically to size target_n."""
    if len(base_seeds) >= target_n:
        return base_seeds[:target_n]
    extra = []
    cur = pool_start
    used = set(base_seeds)
    while len(base_seeds) + len(extra) < target_n:
        if cur not in used:
            extra.append(cur)
            used.add(cur)
        cur += 1
    return base_seeds + extra


# def propose(cfg, max_changes=2, strong_moves=True, seed=None):
#     """
#     CIFAR-100 (ResNet18) proposer that only touches knobs the trainer uses:
#     lr, weight_decay, batch_size, momentum.
#     Biases toward moves that commonly help (smaller batch, lr in 0.05..0.2,
#     wd in 3e-4..1e-3, momentum 0.9..0.95). At most 1-2 knobs per proposal.
#     """
#     if seed is not None:
#         random.seed(seed)

#     new = deepcopy(cfg)
#     changed = []

#     knobs = ["lr", "weight_decay", "batch_size", "momentum"]
#     k_changes = 1 if not strong_moves else random.choice([1, 2])

#     picked = random.sample(knobs, k_changes)

#     old_bs = new["batch_size"]
#     old_lr = new["lr"]

#     for k in picked:
#         if k == "batch_size":
#             # bias smaller batches for generalization
#             choices = [64, 96, 128, 160, 192, 256]
#             if old_bs in choices:
#                 choices.remove(old_bs)
#             new["batch_size"] = int(random.choice(choices))
#             changed.append("batch_size")

#         elif k == "lr":
#             # healthy region for ResNet18 + cosine on CIFAR-100
#             grid = [0.05, 0.075, 0.1, 0.12, 0.15, 0.2]
#             new["lr"] = float(random.choice(grid))
#             changed.append("lr")

#         elif k == "weight_decay":
#             # typical sweet spot range
#             grid = [3e-4, 5e-4, 7.5e-4, 1e-3]
#             new["weight_decay"] = float(random.choice(grid))
#             changed.append("weight_decay")

#         elif k == "momentum":
#             new["momentum"] = float(random.choice([0.85, 0.9, 0.92, 0.95]))
#             changed.append("momentum")

#     # If BS changed but LR not explicitly changed, scale LR linearly to keep stability
#     if new["batch_size"] != old_bs and "lr" not in changed:
#         scale = new["batch_size"] / float(old_bs)
#         new["lr"] = max(1e-4, min(4e-1, old_lr * scale))
#         changed.append("lr(batch-scaled)")

#     return new, changed

# ---------- Proposal generator (multi-knob, higher-leverage) ----------
# def propose(cfg, max_changes=3, strong_moves=True, seed=None):
#     if seed is not None:
#         random.seed(seed)

#     new = deepcopy(cfg)
#     changed = []

#     # Ensure optional knobs exist (no-ops if trainer ignores them)
#     _maybe_add(new, "optimizer", new.get("optimizer", "SGD"))
#     _maybe_add(new, "lr_schedule", new.get("lr_schedule", "step"))
#     _maybe_add(new, "warmup_epochs", new.get("warmup_epochs", 0))
#     _maybe_add(new, "ema_decay", new.get("ema_decay", 0.0))
#     _maybe_add(new, "label_smoothing", new.get("label_smoothing", 0.0))
#     _maybe_add(new, "mixup", new.get("mixup", 0.0))
#     _maybe_add(new, "cutmix", new.get("cutmix", 0.0))
#     _maybe_add(new, "grad_clip", new.get("grad_clip", 0.0))
#     _maybe_add(new, "nesterov", new.get("nesterov", False))
#     _maybe_add(new, "epochs", new.get("epochs", 30))

#     knobs_pool = [
#         "lr", "weight_decay", "batch_size", "momentum",
#         "optimizer", "lr_schedule", "warmup_epochs",
#         "ema_decay", "label_smoothing", "mixup", "cutmix",
#         "grad_clip", "nesterov", "epochs"
#     ]

#     # Number of knobs to change
#     if strong_moves:
#         choices = [1,2,2,3,3] if max_changes >= 3 else [1,2]
#     else:
#         choices = [1,1,2] if max_changes >= 2 else [1]
#     k_changes = random.choice([c for c in choices if c <= max_changes])

#     picked = random.sample(knobs_pool, k_changes)

#     old_bs = new.get("batch_size", None)
#     old_lr = new.get("lr", None)

#     for k in picked:
#         if k == "lr" and "lr" in new:
#             span = 1.0 if strong_moves else 0.75  # 2^±span
#             fac = 2.0 ** random.uniform(-span, span)
#             new["lr"] = _clip(new["lr"] * fac, 1e-4, 4e-1)
#             changed.append("lr")

#         elif k == "weight_decay" and "weight_decay" in new:
#             fac = random.choice([0.3, 0.5, 2.0, 3.0])
#             new["weight_decay"] = _clip(new["weight_decay"] * fac, 1e-6, 5e-3)
#             changed.append("weight_decay")

#         elif k == "batch_size" and "batch_size" in new:
#             choices = [64,96,128,160,192,256,512]
#             if new["batch_size"] in choices:
#                 choices.remove(new["batch_size"])
#             if choices:
#                 new["batch_size"] = int(random.choice(choices))
#                 changed.append("batch_size")

#         elif k == "momentum" and "momentum" in new:
#             target = random.choice([0.85, 0.9, 0.95, _clip(new["momentum"] + random.uniform(-0.06, 0.06), 0.80, 0.99)])
#             new["momentum"] = _clip(target, 0.80, 0.99)
#             changed.append("momentum")

#         elif k == "optimizer":
#             new["optimizer"] = random.choice(["SGD", "AdamW"])
#             changed.append("optimizer")

#         elif k == "lr_schedule":
#             new["lr_schedule"] = random.choice(["cosine", "step", "cosine"])
#             changed.append("lr_schedule")

#         elif k == "warmup_epochs":
#             max_wu = 10 if new.get("lr_schedule", "step") == "cosine" else 5
#             new["warmup_epochs"] = int(random.choice([0,1,2,3,4,5, max_wu]))
#             changed.append("warmup_epochs")

#         elif k == "ema_decay":
#             if new["ema_decay"] <= 0.0:
#                 new["ema_decay"] = random.choice([0.995, 0.997, 0.999, 0.9995])
#             else:
#                 new["ema_decay"] = _clip(new["ema_decay"] + random.choice([-0.002, -0.001, 0.001, 0.002]), 0.990, 0.9997)
#             changed.append("ema_decay")

#         elif k == "label_smoothing":
#             new["label_smoothing"] = float(random.choice([0.05, 0.1, 0.15, 0.2]))
#             changed.append("label_smoothing")

#         elif k == "mixup":
#             new["mixup"] = float(random.choice([0.0, 0.1, 0.2, 0.3]))
#             changed.append("mixup")

#         elif k == "cutmix":
#             new["cutmix"] = float(random.choice([0.0, 0.3, 0.5, 0.7]))
#             changed.append("cutmix")

#         elif k == "grad_clip":
#             new["grad_clip"] = float(random.choice([0.0, 0.5, 1.0, 2.0]))
#             changed.append("grad_clip")

#         elif k == "nesterov":
#             new["nesterov"] = bool(random.choice([True, False, True]))
#             changed.append("nesterov")

#         elif k == "epochs":
#             step = random.choice([0, 10, 20])
#             new["epochs"] = int(max(30, new["epochs"] + step))
#             changed.append("epochs")

#     # If batch size changed and lr present, scale lr linearly with batch ratio
#     if old_bs is not None and old_lr is not None and new.get("batch_size", old_bs) != old_bs and "lr" in new:
#         scale = new["batch_size"] / float(old_bs)
#         new["lr"] = _clip(old_lr * scale, 1e-4, 4e-1)
#         if "lr" not in changed:
#             changed.append("lr(batch-scaled)")

#     # Coherent combos: ensure AdamW not with near-zero wd
#     if new.get("optimizer") == "AdamW" and "weight_decay" in new:
#         if new["weight_decay"] < 5e-5:
#             new["weight_decay"] = 5e-5

#     # Prefer cosine + warmup when strong
#     if strong_moves and new.get("lr_schedule") == "cosine" and new.get("warmup_epochs", 0) == 0:
#         new["warmup_epochs"] = random.choice([3, 5, 10])
#         if "warmup_epochs" not in changed:
#             changed.append("warmup_epochs")

#     return new, changed

# ---------- DB helpers ----------
def ensure_tables(conn):
    # proposals
    conn.execute("""
    CREATE TABLE IF NOT EXISTS proposals(
      id INTEGER PRIMARY KEY,
      ts INTEGER,
      exp_name TEXT,
      iter INTEGER,
      proposer TEXT,
      accepted INTEGER,
      delta REAL,
      n_seeds INTEGER,
      rmax REAL,
      lb REAL,
      mu_hat REAL,
      incumbent_cfg TEXT,
      proposal_cfg TEXT,
      notes TEXT
    );""")

    # runs (for training results)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs(
      id INTEGER PRIMARY KEY,
      ts INTEGER,
      exp_name TEXT,
      seed INTEGER,
      steps INTEGER,
      mean_reward REAL
    );""")

    conn.execute("""
    CREATE INDEX IF NOT EXISTS runs_exp_seed_steps
    ON runs(exp_name, seed, steps, ts);
    """)

    conn.commit()

def fetch_acc(conn, exp_name, seed, epochs_min=1):
    cur = conn.execute("""
      SELECT mean_reward FROM runs
      WHERE exp_name = ? AND seed = ? AND steps >= ?
      ORDER BY ts DESC LIMIT 1
    """, (exp_name, seed, int(epochs_min)))
    row = cur.fetchone()
    return None if row is None else float(row[0])

# ---------- Launch / evaluate ----------
def launch_one(exp_name, seed, epochs, cfg, data_dir, runs_dir, db_name, num_workers):
    logs_dir = os.path.join(runs_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{exp_name}_seed{seed}.log")

    cmd = [
        "python", "train_pgm_cifar.py",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--momentum", str(cfg["momentum"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--augment", str(cfg.get("augment", "basic")),
        "--runs_dir", runs_dir,
        "--db_name", db_name,
        "--data_dir", data_dir,
        "--num_workers", str(num_workers),
    ]
    # Schedule toggle
    if cfg.get("lr_schedule", cfg.get("cosine") and "cosine") == "cosine" or cfg.get("cosine", False):
        cmd.append("--cosine")

    f = open(log_path, "w", buffering=1)
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p, f, log_path

def evaluate_config(conn, config_exp, seeds, epochs, cfg, label, data_dir, runs_dir,
                    db_name, max_procs, num_workers):
    accs, to_run = {}, []
    for s in seeds:
        acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
        if acc is None:
            to_run.append(s)
        else:
            accs[s] = acc

    times = {}
    if to_run:
        print(f"[PGM-Ex5] launching {len(to_run)} trainings for '{config_exp}' with max_procs={max_procs} ({label})")
        procs, queue = [], list(to_run)
        start_times = {}

        while queue or procs:
            while queue and len(procs) < max_procs:
                s = queue.pop(0)
                p, fh, lp = launch_one(config_exp, s, epochs, cfg, data_dir, runs_dir, db_name, num_workers)
                procs.append({"seed": s, "proc": p, "fh": fh, "log": lp})
                start_times[s] = time.time()

            still = []
            for item in procs:
                p = item["proc"]; s = item["seed"]
                if p.poll() is None:
                    still.append(item)
                else:
                    item["fh"].close()
                    dur = time.time() - start_times[s]
                    times[s] = dur
                    if p.returncode != 0:
                        try:
                            tail = open(item["log"], "r").read().splitlines()[-50:]
                            tail_msg = "\n".join(tail)
                        except Exception:
                            tail_msg = "<no log tail>"
                        raise RuntimeError(
                            f"Training failed for seed {s} (rc={p.returncode}). "
                            f"See log: {item['log']}\n--- LOG TAIL ---\n{tail_msg}\n---------------"
                        )
                    print(f"[PGM-Ex5] done seed {s} for '{config_exp}'  (time={dur/60.0:.1f} min)")
            procs = still
            time.sleep(2)

        for s in to_run:
            acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
            if acc is None:
                raise RuntimeError(f"Training produced no logged result for seed {s}.")
            accs[s] = acc

    if times:
        ordered = sorted(times.items())
        mean_t = sum(times.values())/len(times)
        p50 = sorted(times.values())[len(times)//2]
        print("[PGM-Ex5] per-seed times (min): " +
              ", ".join([f"{s}:{t/60.0:.1f}" for s,t in ordered]))
        print(f"[PGM-Ex5] time summary (min): mean={mean_t/60.0:.1f}, median={p50/60.0:.1f}, n={len(times)}")

    return accs, times

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../runs/pgm.db")
    ap.add_argument("--runs_dir", default="../runs")
    ap.add_argument("--data_dir", default="../data")

    # Ex5 defaults (CIFAR-100 only)
    ap.add_argument("--baseline_exp", default="cifar100_baseline")  # kept for legacy, not used after hashing
    ap.add_argument("--exp_name", default="cifar100_pgm_v1")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--rmax_pp", type=float, default=1.0, help="max plausible gain in %-points")
    ap.add_argument("--max_procs", type=int, default=2, help="parallel trainings")
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per proc")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)

    ap.add_argument("--confirm_n_seeds", type=int, default=10,
                    help="seeds used in confirm stage (>= len(--seeds))")
    ap.add_argument("--confirm_epochs", type=int, default=100,
                    help="epochs in confirm stage")
    ap.add_argument("--confirm_mean_pp", type=float, default=0.8,
                    help="screening mean Δ (pp) to trigger confirm")
    ap.add_argument("--confirm_all_pos", action="store_true",
                    help="also trigger confirm if all screening deltas > 0")
    ap.add_argument("--seed_pool_start", type=int, default=1001,
                    help="where to start generating extra seeds for confirm stage")

    # initial incumbent config
    ap.add_argument("--init_cfg", type=str,
        default='{"lr":0.1,"momentum":0.9,"weight_decay":0.0005,"batch_size":128,"augment":"basic","lr_schedule":"cosine"}')
    args = ap.parse_args()

    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    conn = sqlite3.connect(args.db, timeout=60.0)
    ensure_tables(conn)

    seeds = args.seeds
    print(f"[PGM-Ex5] seeds={seeds} budget={args.budget} delta={args.delta} rmax_pp={args.rmax_pp}")

    # ----- Incumbent (SCREEN cache) -----
    inc_cfg = json.loads(args.init_cfg)
    inc_exp = f"{args.exp_name}_inc_{cfg_hash(inc_cfg)}"
    inc_acc, inc_times = evaluate_config(
        conn, inc_exp, seeds, args.epochs, inc_cfg,
        "incumbent (baseline cached)",
        data_dir=args.data_dir, runs_dir=args.runs_dir,
        db_name=os.path.basename(args.db),
        max_procs=args.max_procs, num_workers=args.num_workers
    )
    inc_mean = sum(inc_acc.values()) / len(seeds)
    print(f"[PGM-Ex5] incumbent baseline mean acc = {inc_mean:.2f}%")

    # ----- Alpha-spending schedule: δ_t = δ / (t * H_B) -----
    B = args.budget
    H_B = sum(1.0/i for i in range(1, B+1))

    for it in range(1, args.budget + 1):
        delta_t = args.delta * (1.0 / (it * H_B))
        print(f"[PGM-Ex5] iteration {it}/{args.budget}: alpha spend δ_t = {delta_t:.6f} (total δ = {args.delta})")

        # ----- PROPOSE -----
        prop_cfg, changed = propose(inc_cfg)
        prop_exp = f"{args.exp_name}_prop_it{it}_{cfg_hash(prop_cfg)}"

        # ----- SCREEN (cheap) -----
        prop_acc_scr, prop_times_scr = evaluate_config(
            conn, prop_exp, seeds, args.epochs, prop_cfg,
            f"proposal iter {it} (screen)",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers
        )

        # Paired deltas in %-points
        deltas_pp_scr = [(prop_acc_scr[s] - inc_acc[s]) for s in seeds]
        mean_pp_scr = sum(deltas_pp_scr) / len(deltas_pp_scr)

        # EB lower bound (for logging on screen set)
        y_scr = [max(-1.0, min(1.0, (d / args.rmax_pp))) for d in deltas_pp_scr]
        lb_scr, mu_scr = eb_lower_bound(y_scr, delta_t, R=1.0)
        prop_mean_scr = sum(prop_acc_scr.values()) / len(seeds)
        print(f"[iter {it:02d}][screen] change={changed} μ={mu_scr:.4f} "
              f"LB_scr={lb_scr:.4f} inc_mean={inc_mean:.2f}% "
              f"prop_mean={prop_mean_scr:.2f}% meanΔ={mean_pp_scr:+.2f}pp")

        # ----- DECIDE WHETHER TO CONFIRM -----
        trigger_confirm = should_confirm(
            deltas_pp_scr,
            mean_pp_threshold=args.confirm_mean_pp,
            require_all_pos=args.confirm_all_pos
        )

        if not trigger_confirm:
            conn.execute("""
              INSERT INTO proposals(ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
                                    incumbent_cfg, proposal_cfg, notes)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (int(time.time()), args.exp_name, it, "multi-knob",
                  0, float(delta_t), len(seeds), float(args.rmax_pp),
                  float(lb_scr), float(mu_scr),
                  json.dumps(inc_cfg), json.dumps(prop_cfg),
                  f"[screen-only reject] changed={changed}; meanΔ_pp={mean_pp_scr:.3f}; "
                  f"prop_times_scr={ {k: round(v,2) for k,v in (prop_times_scr or {}).items()} }"))
            conn.commit()
            continue

        # ----- CONFIRM (expensive) -----
        seeds_conf = make_seed_set(seeds, args.confirm_n_seeds, args.seed_pool_start)

        # Incumbent cache at CONFIRM settings (use incumbent's hashed name)
        inc_exp_conf = f"{args.exp_name}_inc_{cfg_hash(inc_cfg)}_confirm"
        inc_acc_conf, _ = evaluate_config(
            conn, inc_exp_conf, seeds_conf, args.confirm_epochs, inc_cfg,
            "incumbent (confirm cache)",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers
        )

        # Proposal at CONFIRM settings
        prop_exp_conf = f"{prop_exp}_confirm"
        prop_acc_conf, prop_times_conf = evaluate_config(
            conn, prop_exp_conf, seeds_conf, args.confirm_epochs, prop_cfg,
            f"proposal iter {it} (confirm)",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers
        )

        # Final PAC decision on confirm set
        y_conf = [max(-1.0, min(1.0, ((prop_acc_conf[s] - inc_acc_conf[s]) / args.rmax_pp))) for s in seeds_conf]
        lb_conf, mu_conf = eb_lower_bound(y_conf, delta_t, R=1.0)
        prop_mean_conf = sum(prop_acc_conf.values()) / len(seeds_conf)
        inc_mean_conf  = sum(inc_acc_conf.values()) / len(seeds_conf)
        decision = "ACCEPT" if lb_conf >= 0 else "REJECT"

        print(f"[iter {it:02d}][confirm] μ={mu_conf:.4f} LB_conf={lb_conf:.4f} "
              f"inc_mean={inc_mean_conf:.2f}% prop_mean={prop_mean_conf:.2f}% decision={decision}")

        conn.execute("""
          INSERT INTO proposals(ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
                                incumbent_cfg, proposal_cfg, notes)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (int(time.time()), args.exp_name, it, "multi-knob",
              1 if decision=="ACCEPT" else 0, float(delta_t), len(seeds_conf), float(args.rmax_pp),
              float(lb_conf), float(mu_conf),
              json.dumps(inc_cfg), json.dumps(prop_cfg),
              f"[screen meanΔ_pp={mean_pp_scr:.3f}] changed={changed}; "
              f"prop_times_conf={ {k: round(v,2) for k,v in (prop_times_conf or {}).items()} }"))
        conn.commit()

        if decision == "ACCEPT":
            # Promote proposal to new incumbent, then re-cache incumbent at SCREEN settings (fair next iteration)
            inc_cfg = prop_cfg
            inc_exp = f"{args.exp_name}_inc_{cfg_hash(inc_cfg)}"
            inc_acc, inc_times = evaluate_config(
                conn, inc_exp, seeds, args.epochs, inc_cfg,
                "incumbent (post-accept screen cache)",
                data_dir=args.data_dir, runs_dir=args.runs_dir,
                db_name=os.path.basename(args.db),
                max_procs=args.max_procs, num_workers=args.num_workers
            )
            inc_mean = sum(inc_acc.values()) / len(seeds)

    print("[PGM-Ex5] done.")

if __name__ == "__main__":
    main()
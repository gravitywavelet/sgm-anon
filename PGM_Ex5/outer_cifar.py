#!/usr/bin/env python3
import argparse, os, sqlite3, subprocess, time, math, json
from pathlib import Path

# ----------------- CLI -----------------
ap = argparse.ArgumentParser("Minimal SGM outer loop for CIFAR (CTHS vs Harmonic)")

# data / io
ap.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
ap.add_argument("--data_dir", default="./data")
ap.add_argument("--runs_dir", default="../runs")
ap.add_argument("--db_name", default="pgm_cifar.db")  # must match your trainer's DB name
ap.add_argument("--num_workers", type=int, default=8)

# training
ap.add_argument("--epochs_screen", type=int, default=3)
ap.add_argument("--epochs_confirm", type=int, default=8)
ap.add_argument("--batch_size", type=int, default=128)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--weight_decay", type=float, default=5e-4)
ap.add_argument("--momentum", type=float, default=0.9)
ap.add_argument("--sched_cosine", action="store_true")

# seeds / loop
ap.add_argument("--seeds_screen", type=int, nargs="+", default=[101,103,107,109])
ap.add_argument("--confirm_n", type=int, default=12)
ap.add_argument("--seed_pool_start", type=int, default=1001)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--delta", type=float, default=0.10)
ap.add_argument("--schedule", default="cths", choices=["cths","harmonic"])

# screen trigger
ap.add_argument("--mean_pp_threshold", type=float, default=0.40, help="pp threshold (+) to trigger confirm")
ap.add_argument("--rmax_pp", type=float, default=100.0, help="pp normalization for Hoeffding (percent -> 100)")

# proposal choice (simple fixed demo: label smoothing 0.1)
ap.add_argument("--proposal", default="ls01", choices=["ls01"])

# exp naming
ap.add_argument("--exp_prefix", default="cifar_sgm_demo")

ap.add_argument("--pp_boost_confirm", type=float, default=0.0,
                help="Add this many percentage points to proposal accuracies at CONFIRM (synthetic true gain).")
ap.add_argument("--pp_boost_screen", type=float, default=0.0,
                help="Optional: add pp boost at SCREEN (usually 0 so trigger remains honest).")

args = ap.parse_args()

# ----------------- DB helpers -----------------
def sql_connect(db_path):
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
    conn.execute("CREATE INDEX IF NOT EXISTS runs_exp_seed_steps ON runs(exp_name, seed, steps, ts);")
    conn.commit()
    return conn

def fetch_acc(conn, exp_name, seed, steps_min=1):
    cur = conn.execute("""
      SELECT mean_reward FROM runs
      WHERE exp_name = ? AND seed = ? AND steps >= ?
      ORDER BY ts DESC LIMIT 1
    """, (exp_name, seed, int(steps_min)))
    row = cur.fetchone()
    return None if row is None else float(row[0])

# ----------------- math -----------------
def hoeffding_lcb(y, delta_t):
    n = max(1, len(y))
    mu = sum(y)/n
    rad = math.sqrt((2.0/n) * math.log(1.0/max(delta_t,1e-12)))
    return mu - rad, mu

def harmonic_deltas(total_delta, B):
    H_B = sum(1.0/i for i in range(1, B+1))
    return [total_delta / ((t+1) * H_B) for t in range(B)], H_B

# ----------------- seeds -----------------
def make_seeds(base, target_n, pool_start):
    if len(base) >= target_n:
        return base[:target_n]
    extra, cur, used = [], pool_start, set(base)
    while len(base) + len(extra) < target_n:
        if cur not in used:
            extra.append(cur); used.add(cur)
        cur += 1
    return base + extra

# ----------------- launcher -----------------
def launch_train(exp_name, seed, epochs, dataset, data_dir, runs_dir, db_name,
                 lr, wd, mom, bs, cosine, label_smoothing=None, num_workers=8):
    """
    Unified launcher for your CIFAR-10/100 trainer which requires --dataset.
    Matches the CLI:
      CIFAR-10/100 trainer with label smoothing for SGM demos:
        --dataset {cifar10,cifar100}
        --label_smoothing (optional)
        --sched {cosine,none}
        --warmup_epochs
    """
    cmd = [
        "python", "train_pgm_cifar.py",           # unified script
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--runs_dir", runs_dir,
        "--db_name", db_name,
        "--data_dir", data_dir,
        "--dataset", dataset,                     # REQUIRED by your trainer
        "--batch_size", str(bs),
        "--lr", str(lr),
        "--weight_decay", str(wd),
        "--momentum", str(mom),
        "--num_workers", str(num_workers),
        "--sched", "cosine" if cosine else "none",
        "--warmup_epochs", "0",
    ]
    if label_smoothing is not None:
        cmd += ["--label_smoothing", str(label_smoothing)]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        tail = "\n".join(e.output.splitlines()[-50:]) if e.output else "<no output>"
        return e.returncode, tail

def ensure_runs(conn, exp_name, seeds, epochs, **kw):
    accs, to_run = {}, []
    for s in seeds:
        a = fetch_acc(conn, exp_name, s, steps_min=epochs)
        if a is None:
            to_run.append(s)
        else:
            accs[s] = a
    if to_run:
        for s in to_run:
            rc, out = launch_train(exp_name, s, epochs, **kw)
            if rc != 0:
                raise RuntimeError(f"train failed for {exp_name} seed={s}\n{out}")
            a = fetch_acc(conn, exp_name, s, steps_min=epochs)
            if a is None:
                raise RuntimeError(f"no row logged for {exp_name} seed={s}")
            accs[s] = a
    return accs

# ----------------- schedules -----------------
class SpendScheduler:
    def __init__(self, mode, total_delta, rounds):
        self.mode = mode
        self.total_delta = total_delta
        self.rounds = rounds
        self.deltas_harm, self.H_B = harmonic_deltas(total_delta, rounds)
        self.confirms_done = 0

    def delta_for_round(self, t, will_confirm=False):
        """Return δ_t used at confirmation in round t (1-indexed)."""
        if self.mode == "harmonic":
            return self.deltas_harm[t-1]
        if self.mode == "cths":
            if not will_confirm:
                return 0.0
            k = self.confirms_done + 1
            return self.total_delta / (k * self.H_B)
        raise ValueError(self.mode)

    def note_confirm(self):
        if self.mode == "cths":
            self.confirms_done += 1

# ----------------- main -----------------
def main():
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.runs_dir, args.db_name)
    conn = sql_connect(db_path)

    # fixed proposal knob (simple demo)
    prop_tag = {"ls01": {"label_smoothing": 0.1}}[args.proposal]

    # incumbent/proposal exp name bases
    inc_base = f"{args.exp_prefix}_{args.dataset}_inc"
    prop_base = f"{args.exp_prefix}_{args.dataset}_prop_{args.proposal}"

    # scheduler
    sched = SpendScheduler(args.schedule, args.delta, args.rounds)

    # cache incumbent (screen)
    inc_screen = f"{inc_base}_screen_e{args.epochs_screen}"
    inc_acc_screen = ensure_runs(
        conn, inc_screen, args.seeds_screen, args.epochs_screen,
        dataset=args.dataset, data_dir=args.data_dir, runs_dir=args.runs_dir, db_name=args.db_name,
        lr=args.lr, wd=args.weight_decay, mom=args.momentum, bs=args.batch_size,
        cosine=args.sched_cosine, label_smoothing=None, num_workers=args.num_workers
    )
    inc_mean_screen = sum(inc_acc_screen.values())/len(inc_acc_screen)
    print(f"[SGM] incumbent (screen) mean acc = {inc_mean_screen:.2f}%")

    accepts = 0
    spent = 0.0

    for t in range(1, args.rounds+1):
        print(f"\n[SGM] Round {t}/{args.rounds}")

        # --- SCREEN ---
        prop_screen = f"{prop_base}_screen_t{t}_e{args.epochs_screen}"
        prop_acc_screen = ensure_runs(
            conn, prop_screen, args.seeds_screen, args.epochs_screen,
            dataset=args.dataset, data_dir=args.data_dir, runs_dir=args.runs_dir, db_name=args.db_name,
            lr=args.lr, wd=args.weight_decay, mom=args.momentum, bs=args.batch_size,
            cosine=args.sched_cosine,
            label_smoothing=(prop_tag["label_smoothing"] if args.dataset=="cifar10" else None),
            num_workers=args.num_workers
        )
        # Optional synthetic lift at SCREEN (usually 0)
        if args.pp_boost_screen != 0.0:
            for s in prop_acc_screen:
                prop_acc_screen[s] += args.pp_boost_screen

        deltas_pp = [prop_acc_screen[s] - inc_acc_screen[s] for s in args.seeds_screen]
        mean_pp = sum(deltas_pp)/len(deltas_pp)
        print(f"[SGM][screen] meanΔ={mean_pp:+.2f}pp (trigger≥{args.mean_pp_threshold}pp?)")

        trigger = (mean_pp >= args.mean_pp_threshold)
        if not trigger:
            print("[SGM] No confirm (below threshold).")
            continue

        # --- CONFIRM ---
        seeds_conf = make_seeds(args.seeds_screen, args.confirm_n, args.seed_pool_start)

        inc_confirm = f"{inc_base}_confirm_e{args.epochs_confirm}"
        inc_acc_confirm = ensure_runs(
            conn, inc_confirm, seeds_conf, args.epochs_confirm,
            dataset=args.dataset, data_dir=args.data_dir, runs_dir=args.runs_dir, db_name=args.db_name,
            lr=args.lr, wd=args.weight_decay, mom=args.momentum, bs=args.batch_size,
            cosine=args.sched_cosine, label_smoothing=None, num_workers=args.num_workers
        )

        prop_confirm = f"{prop_base}_confirm_t{t}_e{args.epochs_confirm}"
        prop_acc_confirm = ensure_runs(
            conn, prop_confirm, seeds_conf, args.epochs_confirm,
            dataset=args.dataset, data_dir=args.data_dir, runs_dir=args.runs_dir, db_name=args.db_name,
            lr=args.lr, wd=args.weight_decay, mom=args.momentum, bs=args.batch_size,
            cosine=args.sched_cosine,
            label_smoothing=(prop_tag["label_smoothing"] if args.dataset=="cifar10" else None),
            num_workers=args.num_workers
        )
        # Synthetic true gain at CONFIRM only (demonstrates power)
        if args.pp_boost_confirm != 0.0:
            for s in prop_acc_confirm:
                prop_acc_confirm[s] += args.pp_boost_confirm

        # spend δ_t depending on schedule
        delta_t = sched.delta_for_round(t, will_confirm=True)
        spent += delta_t

        # Hoeffding LCB on normalized improvements (pp / rmax_pp)
        y = [max(-1.0, min(1.0, (prop_acc_confirm[s] - inc_acc_confirm[s]) / args.rmax_pp)) for s in seeds_conf]
        lcb, mu = hoeffding_lcb(y, delta_t if delta_t>0 else 1.0)  # guard if δ_t=0
        inc_m = sum(inc_acc_confirm.values())/len(seeds_conf)
        prop_m = sum(prop_acc_confirm.values())/len(seeds_conf)
        decision = "ACCEPT" if lcb > 0.0 else "REJECT"

        print(f"[SGM][confirm] schedule={args.schedule}  δ_t={delta_t:.5f}  spent≈{spent:.5f}")
        print(f"               μ={mu:.4f}  LCB={lcb:.4f}  inc={inc_m:.2f}%  prop={prop_m:.2f}%  => {decision}")

        # Count confirmations (spend index) regardless of outcome
        sched.note_confirm()
        if decision == "ACCEPT":
            accepts += 1
            # Optional: promote proposal to incumbent for next screen
            inc_acc_screen = {s: prop_acc_screen[s] for s in args.seeds_screen}

    print("\n[SGM] SUMMARY")
    print(f"  rounds={args.rounds}, schedule={args.schedule}, δ={args.delta}")
    print(f"  accepts={accepts}, spent≈{spent:.5f}")
    print("  Note: CTHS spends δ only on rounds that trigger confirmation; harmonic spends every round.")

if __name__ == "__main__":
    main()
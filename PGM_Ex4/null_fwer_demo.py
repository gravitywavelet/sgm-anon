#!/usr/bin/env python3
import argparse, os, sqlite3, subprocess, time, math, json
from pathlib import Path

# ----------------- CLI -----------------
ap = argparse.ArgumentParser("SGM null-pipeline FWER demo (CIFAR-10)")
ap.add_argument("--data_dir", default="./data")
ap.add_argument("--runs_dir", default="../runs")
ap.add_argument("--db_name", default="pgm_cifar10.db")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--confirm_n", type=int, default=8)          # seeds per round
ap.add_argument("--rounds", type=int, default=12)             # rounds per trial
ap.add_argument("--trials", type=int, default=5)              # independent trials
ap.add_argument("--delta", type=float, default=0.10)          # global error budget
ap.add_argument("--max_procs", type=int, default=4)
ap.add_argument("--base_seed_pool_start", type=int, default=101)
ap.add_argument("--seed_pool_span", type=int, default=1000)   # offset per trial
ap.add_argument("--exp_prefix", default="null_fwer")
ap.add_argument("--normalize_R", type=float, default=1.0)     # accuracy in [0,1] => 1.0; if you pass percentages use 100
args = ap.parse_args()

# ----------------- SQL helpers -----------------
def sql_connect(db_path):
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER, exp_name TEXT, env_id TEXT, seed INTEGER, steps INTEGER,
      mean_reward REAL, std_reward REAL, lr REAL, gamma REAL, clip_range REAL,
      batch_size INTEGER, n_steps INTEGER, device TEXT, notes TEXT, extra TEXT
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS runs_exp_seed_steps ON runs(exp_name, seed, steps, ts);")
    conn.commit()
    return conn

def fetch_acc(conn, exp_name, seed, epochs_min=1):
    cur = conn.execute("""
      SELECT mean_reward FROM runs
      WHERE exp_name = ? AND seed = ? AND steps >= ?
      ORDER BY ts DESC LIMIT 1
    """, (exp_name, seed, int(epochs_min)))
    row = cur.fetchone()
    return None if row is None else float(row[0])

# ----------------- trainer wrapper -----------------
def launch_train(exp_name, seed, epochs, runs_dir, db_name, data_dir,
                 lr=0.1, wd=5e-4, ls=None, sched="cosine", warmup=0, bs=128, log_tail=50):
    cmd = [
        "python", "train_pgm_cifar10.py",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--runs_dir", runs_dir,
        "--db_name", db_name,
        "--data_dir", data_dir,
        "--lr", str(lr),
        "--weight_decay", str(wd),
        "--sched", sched,
        "--warmup_epochs", str(warmup),
        "--batch_size", str(bs),
    ]
    if ls is not None:
        cmd += ["--label_smoothing", str(ls)]

    # Fire and wait; surface failure with short log tail
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        msg = e.output.splitlines()[-log_tail:]
        return e.returncode, "\n".join(msg)

# ----------------- math: Hoeffding LCB -----------------
def hoeffding_lcb(y, delta_t):
    n = len(y)
    mu = sum(y)/max(n,1)
    rad = math.sqrt((2.0/n) * math.log(1.0/max(delta_t,1e-12)))
    return mu - rad, mu

# harmonic spending
def harmonic_deltas(delta, B):
    H_B = sum(1.0/i for i in range(1, B+1))
    return [delta / ((t+1) * H_B) for t in range(B)]

# ----------------- seeds -----------------
def make_seeds(start, n):
    # deterministic consecutive seeds: start, start+2, ...
    # (skips to avoid accidental overlap with tiny prior tests)
    return [start + 2*i for i in range(n)]

# ----------------- main -----------------
def main():
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    db_path = os.path.join(args.runs_dir, args.db_name)
    conn = sql_connect(db_path)

    # 1) Build a reusable incumbent cache ONCE per (trial) seed pool
    #    We’ll reuse incumbents across all rounds to halve runtime.
    # Trial loop: each trial uses a disjoint seed window to keep things clean.
    fwer_hits = 0
    trial_results = []

    for trial in range(1, args.trials+1):
        print(f"\n=== Trial {trial}/{args.trials} ===")
        seed_base = args.base_seed_pool_start + (trial-1)*args.seed_pool_span
        seeds = make_seeds(seed_base, args.confirm_n)
        rounds = args.rounds
        deltas = harmonic_deltas(args.delta, rounds)

        # Incumbent cache exp_name
        inc_exp = f"{args.exp_prefix}_inc_cache_trial{trial}"
        # Train incumbent (no label smoothing) for all seeds if not present
        for s in seeds:
            acc = fetch_acc(conn, inc_exp, s, epochs_min=args.epochs)
            if acc is None:
                rc, out = launch_train(inc_exp, s, args.epochs, args.runs_dir, args.db_name, args.data_dir,
                                       lr=0.1, wd=5e-4, ls=None, sched="cosine", warmup=0, bs=128)
                if rc != 0:
                    print(f"[ERR][inc seed {s}] trainer failed:\n{out}")
                    return
        # Collect incumbent accuracies
        inc_acc = {s: fetch_acc(conn, inc_exp, s, epochs_min=args.epochs) for s in seeds}

        any_accept = False
        for t in range(1, rounds+1):
            delta_t = deltas[t-1]
            prop_exp = f"{args.exp_prefix}_prop_null_t{t}_trial{trial}"

            # Proposal == Incumbent (true Δ = 0): run proposal fresh to induce independent noise
            for s in seeds:
                acc = fetch_acc(conn, prop_exp, s, epochs_min=args.epochs)
                if acc is None:
                    rc, out = launch_train(prop_exp, s, args.epochs, args.runs_dir, args.db_name, args.data_dir,
                                           lr=0.1, wd=5e-4, ls=None, sched="cosine", warmup=0, bs=128)
                    if rc != 0:
                        print(f"[ERR][prop t{t} seed {s}] trainer failed:\n{out}")
                        return

            # pull back proposal accs
            prop_acc = {s: fetch_acc(conn, prop_exp, s, epochs_min=args.epochs) for s in seeds}

            # paired deltas in normalized units (accuracy in [0,1])
            y = [ (prop_acc[s] - inc_acc[s]) / args.normalize_R for s in seeds ]  # if DB stores %, set R=100
            lcb, mu = hoeffding_lcb(y, delta_t)
            decision = (lcb > 0.0)
            print(f"[Trial {trial}][Round {t:02d}] n={len(seeds)} mu={mu:.4f} LCB={lcb:.4f} delta_t={delta_t:.5f} -> {'ACCEPT' if decision else 'REJECT'}")
            if decision:
                any_accept = True  # a false accept under the null

        trial_results.append(any_accept)
        if any_accept:
            fwer_hits += 1

    fwer_hat = fwer_hits / max(args.trials,1)
    print("\n=== Summary ===")
    print(f"trials={args.trials}, rounds_per_trial={args.rounds}, n_confirm={args.confirm_n}, epochs={args.epochs}, δ={args.delta}")
    print(f"Empirical FWER (≥1 accept in a trial): {fwer_hat:.3f}  (target δ ≈ {args.delta})")
    print(f"Per-trial accepts: {trial_results}")

if __name__ == "__main__":
    main()
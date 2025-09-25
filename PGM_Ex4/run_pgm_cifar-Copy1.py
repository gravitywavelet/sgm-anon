#!/usr/bin/env python3
# Ex4 – PGM for supervised HP tuning on CIFAR-10 (drives train_pgm_cifar.py)
import argparse, json, math, os, sqlite3, subprocess, time, random
from pathlib import Path

# ---------------- PAC bound (Empirical Bernstein) ----------------
def eb_lower_bound(y_list, delta, R=2.0):
    """Return (LB, mu) for y in [-1,1]; R = b-a = 2.0 after clipping/normalizing."""
    n = len(y_list)
    mu = sum(y_list) / n
    s2 = sum((y - mu) ** 2 for y in y_list) / n  # population variance
    term1 = math.sqrt(2.0 * s2 * math.log(3.0 / delta) / n)
    term2 = (3.0 * R * math.log(3.0 / delta)) / n
    return mu - term1 - term2, mu

# ---------------- Proposal generator (one knob at a time) ----------------
def propose(cfg):
    """Propose EXACTLY ONE change; gamma does not exist here (supervised)."""
    new = cfg.copy()
    knobs = ["lr", "weight_decay", "batch_size", "momentum"]  # 1-param proposals
    name = random.choice(knobs)

    if name == "lr":
        # multiplicative jitter on log-scale
        fac = 2.0 ** random.uniform(-0.75, 0.75)     # ~x0.6 to x1.68 (gentler than x2)
        new["lr"] = float(min(0.4, max(1e-4, cfg["lr"] * fac)))
    elif name == "weight_decay":
        fac = 2.0 ** random.uniform(-1.0, 1.0)       # x0.5..x2
        new["weight_decay"] = float(min(5e-3, max(1e-6, cfg["weight_decay"] * fac)))
    elif name == "batch_size":
        choices = [64, 96, 128, 160, 192, 256]
        if cfg["batch_size"] in choices:
            choices.remove(cfg["batch_size"])
        new["batch_size"] = int(random.choice(choices))
    elif name == "momentum":
        step = random.uniform(-0.05, 0.05)
        new["momentum"] = float(min(0.99, max(0.80, cfg["momentum"] + step)))

    return new, [name]

# ---------------- SQLite helpers ----------------
def ensure_tables(conn):
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
    );
    """)
    conn.commit()

def fetch_acc(conn, exp_name, seed, epochs_min=1):
    """Fetch latest accuracy (%) for (exp_name, seed) with steps>=epochs_min."""
    cur = conn.execute("""
      SELECT mean_reward FROM runs
      WHERE exp_name = ? AND seed = ? AND steps >= ?
      ORDER BY ts DESC LIMIT 1
    """, (exp_name, seed, int(epochs_min)))
    row = cur.fetchone()
    return None if row is None else float(row[0])  # accuracy stored in mean_reward

# ---------------- Training orchestration ----------------
def launch_one(exp_name, seed, epochs, cfg, data_dir, runs_dir, db_name):
    cmd = [
        "python", "train_pgm_cifar.py",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--momentum", str(cfg["momentum"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--augment", cfg["augment"],
        "--runs_dir", runs_dir,
        "--db_name", db_name,
        "--data_dir", data_dir,
    ]
    if cfg.get("cosine", True):
        cmd.append("--cosine")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def evaluate_config(conn, config_exp, seeds, epochs, cfg, cache_label, data_dir, runs_dir,
                    db_name, max_procs):
    """
    Make sure we have results for (config_exp, seed). If missing, train now.
    Returns: dict seed -> accuracy (%)
    """
    out, to_run = {}, []
    for s in seeds:
        acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
        if acc is None:
            to_run.append(s)
        else:
            out[s] = acc

    if to_run:
        print(f"[PGM] launching {len(to_run)} trainings for '{config_exp}' with max_procs={max_procs}")
        procs, queue = [], list(to_run)
        while queue or procs:
            # fill slots
            while queue and len(procs) < max_procs:
                s = queue.pop(0)
                p = launch_one(config_exp, s, epochs, cfg, data_dir, runs_dir, db_name)
                procs.append((s, p))
            # collect finished
            still = []
            for s, p in procs:
                if p.poll() is None:
                    still.append((s, p))
                else:
                    print(f"[PGM] done seed {s} for '{config_exp}'")
            procs = still
            time.sleep(3)

        # after all done, read accuracies
        for s in to_run:
            acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
            if acc is None:
                raise RuntimeError(f"Training produced no logged result for seed {s}.")
            out[s] = acc

    return out

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../runs/pgm.db")
    ap.add_argument("--runs_dir", default="../runs")
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--baseline_exp", default="cifar10_baseline")
    ap.add_argument("--exp_name", default="pgm_cifar10")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--rmax_pp", type=float, default=2.0, help="max plausible gain in %-points used for normalization")
    ap.add_argument("--max_procs", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--init_cfg", type=str,
                    default='{"lr":0.1,"momentum":0.9,"weight_decay":0.0005,"batch_size":128,"augment":"basic","cosine":true}')
    args = ap.parse_args()

    # env prep
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db, timeout=60.0)
    ensure_tables(conn)

    seeds = args.seeds
    n = len(seeds)
    print(f"[PGM-Ex4] seeds={seeds} budget={args.budget} delta={args.delta} rmax_pp={args.rmax_pp}")

    incumbent_cfg = json.loads(args.init_cfg)

    # 0) Baseline (cached); accuracy in percent
    inc_acc = evaluate_config(conn, args.baseline_exp, seeds, args.epochs, incumbent_cfg,
                              cache_label="incumbent (baseline cached)",
                              data_dir=args.data_dir, runs_dir=args.runs_dir,
                              db_name=os.path.basename(args.db), max_procs=args.max_procs)
    inc_mean = sum(inc_acc.values()) / n
    print(f"[PGM-Ex4] incumbent baseline mean acc = {inc_mean:.2f}%")

    # 1..T proposals
    for it in range(1, args.budget + 1):
        prop_cfg, changed = propose(incumbent_cfg)
        prop_exp = f"{args.exp_name}_iter{it}_prop"

        prop_acc = evaluate_config(conn, prop_exp, seeds, args.epochs, prop_cfg,
                                   cache_label=f"proposal iter {it}",
                                   data_dir=args.data_dir, runs_dir=args.runs_dir,
                                   db_name=os.path.basename(args.db), max_procs=args.max_procs)

        # improvement (percentage points), normalized and clipped to [-1,1]
        y = []
        for s in seeds:
            d = (prop_acc[s] - inc_acc[s]) / args.rmax_pp
            d = max(-1.0, min(1.0, d))
            y.append(d)

        lb, mu = eb_lower_bound(y, delta=args.delta, R=2.0)
        prop_mean = sum(prop_acc.values()) / n

        print(f"[iter {it:02d}] change={changed} "
              f"mu={mu:.4f}  LB(1-δ)={lb:.4f}  "
              f"inc_mean={inc_mean:.2f}%  prop_mean={prop_mean:.2f}%  "
              f"decision={'ACCEPT' if lb >= 0.0 else 'REJECT'}")

        # log
        conn.execute("""
          INSERT INTO proposals(ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
                                incumbent_cfg, proposal_cfg, notes)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (int(time.time()), args.exp_name, it, "random-1param",
              1 if lb >= 0.0 else 0, args.delta, n, args.rmax_pp,
              lb, mu, json.dumps(incumbent_cfg), json.dumps(prop_cfg),
              f"changed={changed}"))
        conn.commit()

        if lb >= 0.0:
            incumbent_cfg = prop_cfg
            inc_acc = prop_acc
            inc_mean = prop_mean

    print("[PGM-Ex4] done.")

if __name__ == "__main__":
    main()
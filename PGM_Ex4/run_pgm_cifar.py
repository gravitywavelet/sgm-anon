#!/usr/bin/env python3
# Ex4 – PGM for supervised HP tuning on CIFAR-10
import argparse, json, math, os, sqlite3, subprocess, time, random
from pathlib import Path

# ---------- PAC bound ----------
def eb_lower_bound(y_list, delta, R=2.0):
    n = len(y_list)
    mu = sum(y_list)/n
    s2 = sum((y-mu)**2 for y in y_list)/n
    term1 = math.sqrt(2.0*s2*math.log(3.0/max(delta,1e-12))/n)
    term2 = (3.0*R*math.log(3.0/max(delta,1e-12)))/n
    return mu - term1 - term2, mu

# ---------- Proposal generator ----------
def propose(cfg):
    new = cfg.copy()
    knobs = ["lr","weight_decay","batch_size","momentum"]
    k = random.choice(knobs)
    if k == "lr":
        fac = 2.0**random.uniform(-0.75, 0.75)
        new["lr"] = float(min(0.4, max(1e-4, cfg["lr"]*fac)))
    elif k == "weight_decay":
        fac = 2.0**random.uniform(-1.0, 1.0)
        new["weight_decay"] = float(min(5e-3, max(1e-6, cfg["weight_decay"]*fac)))
    elif k == "batch_size":
        choices = [64,96,128,160,192,256,512]
        if cfg["batch_size"] in choices: choices.remove(cfg["batch_size"])
        new["batch_size"] = int(random.choice(choices))
    elif k == "momentum":
        step = random.uniform(-0.05, 0.05)
        new["momentum"] = float(min(0.99, max(0.80, cfg["momentum"]+step)))
    return new, [k]

# ---------- DB helpers ----------
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
    );""")
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
        "--augment", cfg["augment"],
        "--runs_dir", runs_dir,
        "--db_name", db_name,
        "--data_dir", data_dir,
        "--num_workers", str(num_workers)
    ]
    if cfg.get("cosine", True):
        cmd.append("--cosine")

    f = open(log_path, "w", buffering=1)
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p, f, log_path

def evaluate_config(conn, config_exp, seeds, epochs, cfg, label, data_dir, runs_dir,
                    db_name, max_procs, num_workers):
    """
    Returns:
      accs: dict seed -> test accuracy
      times: dict seed -> wall-clock seconds for that seed (measured here)
    """
    accs, to_run = {}, []
    for s in seeds:
        acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
        if acc is None:
            to_run.append(s)
        else:
            accs[s] = acc

    times = {}  # seed -> seconds

    if to_run:
        print(f"[PGM] launching {len(to_run)} trainings for '{config_exp}' with max_procs={max_procs} ({label})")
        procs, queue = [], list(to_run)
        start_times = {}

        while queue or procs:
            # Fill slots
            while queue and len(procs) < max_procs:
                s = queue.pop(0)
                p, f, lp = launch_one(config_exp, s, epochs, cfg, data_dir, runs_dir, db_name, num_workers)
                procs.append({"seed": s, "proc": p, "fh": f, "log": lp})
                start_times[s] = time.time()

            # Check running
            still = []
            for item in procs:
                p = item["proc"]
                s = item["seed"]
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
                    print(f"[PGM] done seed {s} for '{config_exp}'  (time={dur/60.0:.1f} min)")
            procs = still
            time.sleep(2)

        # Pull accuracies
        for s in to_run:
            acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
            if acc is None:
                raise RuntimeError(f"Training produced no logged result for seed {s}.")
            accs[s] = acc

    # Print a compact summary of per-seed times (only for newly run seeds)
    if times:
        ordered = sorted(times.items())
        mean_t = sum(times.values())/len(times)
        p50 = sorted(times.values())[len(times)//2]
        print("[PGM] per-seed times (min): " +
              ", ".join([f"{s}:{t/60.0:.1f}" for s,t in ordered]))
        print(f"[PGM] time summary (min): mean={mean_t/60.0:.1f}, median={p50/60.0:.1f}, n={len(times)}")

    return accs, times

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../runs/pgm.db")
    ap.add_argument("--runs_dir", default="../runs")
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--dataset", type=str, default="CIFAR100", choices=["CIFAR10","CIFAR100"])
    ap.add_argument("--baseline_exp", default="cifar100_baseline")
    ap.add_argument("--exp_name", default="cifar100_baseline")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--rmax_pp", type=float, default=2.0, help="max plausible gain in %-points")
    ap.add_argument("--max_procs", type=int, default=2, help="parallel trainings")
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per proc")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--init_cfg", type=str,
        default='{"lr":0.1,"momentum":0.9,"weight_decay":0.0005,"batch_size":128,"augment":"basic","cosine":true}')
    args = ap.parse_args()

    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db, timeout=60.0)
    ensure_tables(conn)

    seeds = args.seeds
    n = len(seeds)
    print(f"[PGM-Ex4] seeds={seeds} budget={args.budget} delta={args.delta} rmax_pp={args.rmax_pp}")

    inc_cfg = json.loads(args.init_cfg)

    # Baseline (incumbent)
    inc_acc, inc_times = evaluate_config(
        conn, args.baseline_exp, seeds, args.epochs, inc_cfg,
        "incumbent (baseline cached)",
        data_dir=args.data_dir, runs_dir=args.runs_dir,
        db_name=os.path.basename(args.db),
        max_procs=args.max_procs, num_workers=args.num_workers
    )
    inc_mean = sum(inc_acc.values())/n
    print(f"[PGM-Ex4] incumbent baseline mean acc = {inc_mean:.2f}%")

    # Alpha spending constants
    B = args.budget
    H_B = sum(1.0/i for i in range(1, B+1))

    for it in range(1, args.budget+1):
        # Per-iteration alpha spend
        delta_t = args.delta * (1.0 / (it * H_B))
        print(f"[PGM-Ex4] iteration {it}/{args.budget}: alpha spend δ_t = {delta_t:.6f} (total δ = {args.delta})")

        prop_cfg, changed = propose(inc_cfg)
        prop_exp = f"{args.exp_name}_iter{it}_prop"
        prop_acc, prop_times = evaluate_config(
            conn, prop_exp, seeds, args.epochs, prop_cfg,
            f"proposal iter {it}",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers
        )

        # Normalize improvement by rmax_pp
        y = []
        for s in seeds:
            d = (prop_acc[s] - inc_acc[s]) / args.rmax_pp
            d = max(-1.0, min(1.0, d))
            y.append(d)

        lb, mu = eb_lower_bound(y, delta_t, R=2.0)
        prop_mean = sum(prop_acc.values())/n
        decision = "ACCEPT" if lb >= 0 else "REJECT"

        print(f"[iter {it:02d}] change={changed} μ={mu:.4f} LB(1-δ_t)={lb:.4f} "
              f"inc_mean={inc_mean:.2f}% prop_mean={prop_mean:.2f}% decision={decision}")

        # Log proposal (store δ_t in 'delta')
        conn.execute("""
          INSERT INTO proposals(ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
                                incumbent_cfg, proposal_cfg, notes)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (int(time.time()), args.exp_name, it, "random-1param",
              1 if decision=="ACCEPT" else 0, float(delta_t), n, float(args.rmax_pp),
              float(lb), float(mu),
              json.dumps(inc_cfg), json.dumps(prop_cfg),
              f"changed={changed}; inc_times={ {k: round(v,2) for k,v in (inc_times or {}).items()} }; "
              f"prop_times={ {k: round(v,2) for k,v in (prop_times or {}).items()} }"))
        conn.commit()

        if decision == "ACCEPT":
            inc_cfg = prop_cfg
            inc_acc = prop_acc
            inc_mean = prop_mean
            inc_times = prop_times  # carry over most recent timing set

    print("[PGM-Ex4] done.")

if __name__ == "__main__":
    main()
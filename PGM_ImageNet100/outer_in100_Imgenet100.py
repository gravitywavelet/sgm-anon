#!/usr/bin/env python3
# Ex6 – PGM outer loop for ImageNet (multi-knob proposals + two-stage PAC gate)

# outer_in100.py  (compact, balanced header)

import argparse, json, math, os, sqlite3, subprocess, time, zlib, random
from copy import deepcopy
from pathlib import Path
import torch


# ---------- PRESET PROPOSALS (exactly 6, in order) ----------
# Keys that end with '@mul' multiply the existing value instead of setting.
# We keep EMA OFF throughout presets.
_PRESET_PLAN = [
    # 1) LR up nudge + warmup
    {"lr@mul": 1.10, "warmup_epochs": 5, "ema": False},
    # 2) LR down nudge, no warmup
    {"lr@mul": 0.90, "warmup_epochs": 0, "ema": False},
    # 3) Light RandAug
    {"randaug_m": 7, "randaug_n": 2, "ema": False},
    # 4) Light regularization
    {"weight_decay": 0.04, "drop_path": 0.10, "ema": False},
    # 5) Label smoothing only
    {"label_smoothing": 0.10, "ema": False},
    # 6) Mixup only
    {"mixup": 0.10, "cutmix": 0.00, "ema": False},
]
_PLAN_IDX = 0  # advances each time propose(...) is called

def _ensure_keys(cfg):
    """Avoid KeyErrors when applying presets."""
    defaults = {
        "lr": 5e-4,
        "warmup_epochs": 10,
        "ema": False,
        "randaug_m": 7,
        "randaug_n": 1,
        "weight_decay": 0.05,
        "drop_path": 0.10,
        "label_smoothing": 0.0,
        "mixup": 0.8,
        "cutmix": 1.0,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v

def _apply_preset_changes(cfg, preset_dict):
    """
    Apply a preset change dict to cfg.
    Returns (new_cfg, changes_list) where changes_list is a list of (key, value) pairs,
    with '@mul' keys reported as ('lr@mul', factor) etc. for logging parity.
    """
    new = deepcopy(cfg)
    changes = []
    for k, v in preset_dict.items():
        if k.endswith("@mul"):
            base_k = k[:-4]
            _ensure_keys(new)
            # multiply while preserving type
            new[base_k] = type(new[base_k])(new[base_k] * v)
            changes.append((k, v))  # keep '@mul' in description for clarity
        else:
            new[k] = v
            changes.append((k, v))
    # ensure EMA off unless preset explicitly turns it on (we never do)
    if "ema" not in preset_dict:
        new["ema"] = False
    return new, changes

def detect_ema_flag():
    """Return ('custom', None) if trainer supports --ema,
       ('timm', None) if it supports --model-ema, else ('none', reason)."""
    try:
        out = subprocess.check_output(
            ["python", "train_pgm_in100.py", "-h"],
            stderr=subprocess.STDOUT, text=True
        )
    except subprocess.CalledProcessError as e:
        return "none", f"help failed: {e.output[-200:] if e.output else e}"
    if "--model-ema" in out:
        return "timm", None
    if "--ema " in out or "\n  --ema " in out:
        return "custom", None
    return "none", "no EMA flag found in trainer help"

# --- add these helpers near the top (after imports) ---
def eb_lower_bound(y_list, delta, R=1.0):
    import math
    n = len(y_list)
    mu = sum(y_list) / n
    s2 = sum((y - mu) ** 2 for y in y_list) / n
    term1 = (2.0 * s2 * math.log(3.0 / max(delta, 1e-12)) / n) ** 0.5
    term2 = (3.0 * R * math.log(3.0 / max(delta, 1e-12)) / n)
    return mu - term1 - term2, mu

def hoeffding_lower_bound(y_list, delta):
    import math
    n = len(y_list)
    mu = sum(y_list) / n
    radius = math.sqrt((2.0 / n) * math.log(1.0 / max(delta, 1e-12)))
    return mu - radius, mu

def should_confirm(deltas_pp, mean_pp_threshold=0.4, require_all_pos=False):
    if not deltas_pp:
        return False
    if require_all_pos and all(d > 0 for d in deltas_pp):
        return True
    return (sum(deltas_pp) / len(deltas_pp)) >= mean_pp_threshold

def make_seed_set(base_seeds, target_n, pool_start=1001):
    if len(base_seeds) >= target_n:
        return base_seeds[:target_n]
    extra, cur, used = [], pool_start, set(base_seeds)
    while len(base_seeds) + len(extra) < target_n:
        if cur not in used:
            extra.append(cur); used.add(cur)
        cur += 1
    return base_seeds + extra

# ---------- small utils ----------
def cfg_hash(cfg: dict) -> str:
    return format(zlib.adler32(json.dumps(cfg, sort_keys=True).encode()), "08x")

# ---------- args ----------
ap = argparse.ArgumentParser("PGM Ex6 – IN100 + DeiT-S outer loop")

ap.add_argument("--db", default="../runs/pgm_in100.db")
ap.add_argument("--runs_dir", default="../runs")
ap.add_argument("--data_dir", default="../data/imagenet100")

ap.add_argument("--exp_name", default="in100_deitS_pgm_v1")

ap.add_argument("--epochs", type=int, default=50)           # screening
ap.add_argument("--confirm_epochs", type=int, default=120)  # confirmation
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--confirm_n_seeds", type=int, default=30)

ap.add_argument("--delta", type=float, default=0.10)
ap.add_argument("--rmax_pp", type=float, default=0.5)
ap.add_argument("--confirm_mean_pp", type=float, default=0.40)
ap.add_argument("--confirm_all_pos", action="store_true")
ap.add_argument("--seed_pool_start", type=int, default=1001)

ap.add_argument("--max_procs", type=int, default=2)
ap.add_argument("--num_workers", type=int, default=8)
ap.add_argument("--budget", type=int, default=12,
                help="max outer-loop iterations (number of proposals to evaluate)")

# Keep this as a *single* json.dumps(...) call. No stray commas/brackets.
ap.add_argument(
    "--init_cfg",
    type=str,
    default=json.dumps({
        "dataset": "imagenet100",
        "model": "deit_small_patch16_224",
        "img_size": 224,
        "optimizer": "adamw",
        "lr": 5e-4,
        "weight_decay": 0.05,
        "sched": "cosine",
        "warmup_epochs": 10,
        "epochs": 50,                 # screening epochs only; confirm uses --confirm_epochs
        "batch_effective": 512,       # enforced via grad accumulation
        "drop_path": 0.1,
        "randaug_m": 7,
        "mixup": 0.8,
        "cutmix": 1.0,
        "label_smoothing": 0.1,
        "ema": False                  # proposals may toggle to True (0.9999 in trainer)
    })
)


# ----- phase overrides -----


SCREEN_OVERRIDES = {
    "model": "deit_small_patch16_224",  # same family as confirm
    "img_size": 224,
    "randaug_m": 7,                     # keep light aug if you want
    "randaug_n": 1,
    "ema": False, 
}

CONFIRM_OVERRIDES = {
    "model": "deit_small_patch16_224", # full recipe for confirm
    "img_size": 224,
    # leave randaug_m to cfg/proposal (often 9), or set explicitly:
    # "randaug_m": 9,
}

def apply_overrides(cfg, overrides):
    new = deepcopy(cfg)
    new.update(overrides)
    return new


def propose(cfg, seed=None):
    """
    Deterministic 6-step preset plan (aimed at one accept), then fallback to previous multi-knob logic.
    Returns: (prop_cfg, changes_list)

    Sets prop_cfg["_proposer_tag"] to "preset" for the first 6 calls, else "multi-knob" for fallback.
    """
    global _PLAN_IDX
    base = deepcopy(cfg)
    _ensure_keys(base)

    # --- Use the preset plan for the first 6 iterations ---
    if _PLAN_IDX < len(_PRESET_PLAN):
        prop_cfg, changes = _apply_preset_changes(base, _PRESET_PLAN[_PLAN_IDX])
        _PLAN_IDX += 1
        # Tag for DB/logging
        prop_cfg["_proposer_tag"] = "preset"
        return prop_cfg, changes

    # --- Fallback: previous multi-knob candidate recipe (keeps EMA OFF here) ---
    new = deepcopy(base); changes=[]
    moves = []
    # keep EMA off in fallback as well (given prior negative impact)
    if int(new.get("randaug_m", 7)) != 9:
        moves.append(("ra_up", lambda: (new.__setitem__("randaug_m", 9), changes.append(("randaug_m", 9)))))
    if abs(new.get("drop_path", 0.1) - 0.2) > 1e-12:
        moves.append(("dp_up", lambda: (new.__setitem__("drop_path", 0.2), changes.append(("drop_path", 0.2)))))
    if abs(new.get("weight_decay", 0.05) - 0.08) > 1e-12:
        moves.append(("wd_up", lambda: (new.__setitem__("weight_decay", 0.08), changes.append(("weight_decay", 0.08)))))

    if not moves:
        new["_proposer_tag"] = "multi-knob"
        return new, changes

    idx = (seed or 0) % len(moves)
    moves[idx][1]()
    new["_proposer_tag"] = "multi-knob"
    return new, changes

# def propose(cfg, seed=None):
#     new = deepcopy(cfg); changes=[]; moves=[]
#     # Turn EMA on as a candidate (screen override keeps it off; confirm will respect it)
#     if not new.get("ema", False):
#         moves.append(("ema_on", lambda: (new.__setitem__("ema", True), changes.append(("ema", True)))))

#     if int(new.get("randaug_m", 7)) != 9:
#         moves.append(("ra_up", lambda: (new.__setitem__("randaug_m", 9), changes.append(("randaug_m", 9)))))
#     if abs(new.get("drop_path", 0.1) - 0.2) > 1e-12:
#         moves.append(("dp_up", lambda: (new.__setitem__("drop_path", 0.2), changes.append(("drop_path", 0.2)))))
#     if abs(new.get("weight_decay", 0.05) - 0.08) > 1e-12:
#         moves.append(("wd_up", lambda: (new.__setitem__("weight_decay", 0.08), changes.append(("weight_decay", 0.08)))))

#     if not moves: return new, changes
#     idx = (seed or 0) % len(moves); moves[idx][1](); return new, changes

# ---------- DB helpers ----------
def ensure_tables(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER, exp_name TEXT, env_id TEXT, seed INTEGER, steps INTEGER,
      mean_reward REAL, std_reward REAL, lr REAL, gamma REAL, clip_range REAL,
      batch_size INTEGER, n_steps INTEGER, device TEXT, notes TEXT, extra TEXT
    );""")
    conn.execute("CREATE INDEX IF NOT EXISTS runs_exp_seed_steps ON runs(exp_name, seed, steps, ts);")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS proposals(
      id INTEGER PRIMARY KEY,
      ts INTEGER, exp_name TEXT, iter INTEGER, proposer TEXT, accepted INTEGER,
      delta REAL, n_seeds INTEGER, rmax REAL, lb REAL, mu_hat REAL,
      incumbent_cfg TEXT, proposal_cfg TEXT, notes TEXT
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

# ---------- LAUNCH (timm trainer) ----------
def launch_one(exp_name, seed, epochs, cfg, data_dir, runs_dir, db_name, num_workers, ema_mode="none"):
    logs_dir = os.path.join(runs_dir, "logs"); os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{exp_name}_seed{seed}.log")

    cmd = [
        "python", "train_pgm_in100.py",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--runs_dir", runs_dir,
        "--db_name", db_name,
        "--data_dir", data_dir,
        "--num_workers", str(num_workers),

        "--model", cfg.get("model", "deit_small_patch16_224"),
        "--img_size", str(cfg.get("img_size", 224)),
        "--optimizer", cfg.get("optimizer", "adamw"),
        "--lr", str(cfg.get("lr", 5e-4)),
        "--weight_decay", str(cfg.get("weight_decay", 0.05)),
        "--sched", cfg.get("sched", "cosine"),
        "--warmup_epochs", str(cfg.get("warmup_epochs", 10)),
        "--batch_effective", str(cfg.get("batch_effective", 512)),
        "--drop_path", str(cfg.get("drop_path", 0.1)),
        "--randaug_m", str(cfg.get("randaug_m", 7)),
        "--mixup", str(cfg.get("mixup", 0.8)),
        "--cutmix", str(cfg.get("cutmix", 1.0)),
        "--label_smoothing", str(cfg.get("label_smoothing", 0.1)),
    ]


    if cfg.get("ema", False):
        if ema_mode == "timm":
            cmd += ["--model-ema", "--model-ema-decay", "0.9999"]
        elif ema_mode == "custom":
            # your trainer expects a single --ema value
            cmd += ["--ema", "0.999"]
        else:
            print("[PGM-Ex6][warn] EMA requested but trainer has no EMA flag; ignoring.")

    f = open(log_path, "w", buffering=1)
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p, f, log_path

def evaluate_config(conn, config_exp, seeds, epochs, cfg, label, data_dir, runs_dir,
                    db_name, max_procs, num_workers, ema_mode="none"):
    accs, to_run = {}, []
    for s in seeds:
        acc = fetch_acc(conn, config_exp, s, epochs_min=epochs)
        if acc is None:
            to_run.append(s)
        else:
            accs[s] = acc

    times = {}
    if to_run:
        print(f"[PGM-Ex6] launching {len(to_run)} trainings for '{config_exp}' with max_procs={max_procs} ({label})")
        procs, queue = [], list(to_run)
        start_times = {}

        while queue or procs:
            while queue and len(procs) < max_procs:
                s = queue.pop(0)
                p, fh, lp = launch_one(config_exp, s, epochs, cfg, data_dir, runs_dir, db_name, num_workers, ema_mode=ema_mode)
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
                    print(f"[PGM-Ex6] done seed {s} for '{config_exp}'  (time={dur/60.0:.1f} min)")
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
        print("[PGM-Ex6] per-seed times (min): " +
              ", ".join([f"{s}:{t/60.0:.1f}" for s,t in ordered]))
        print(f"[PGM-Ex6] time summary (min): mean={mean_t/60.0:.1f}, median={p50/60.0:.1f}, n={len(times)}")

    return accs, times

# ---------- Main ----------
def main():
    
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

    ema_mode, ema_msg = detect_ema_flag()
    print(f"[PGM-Ex6] EMA flag mode = {ema_mode}" + (f" ({ema_msg})" if ema_msg else ""))

    seeds = args.seeds
    print(f"[PGM-Ex6] seeds={seeds} budget={args.budget} delta={args.delta} rmax_pp={args.rmax_pp}")

    # ----- Incumbent (SCREEN cache) -----
    inc_cfg = json.loads(args.init_cfg)
    screen_inc_cfg = apply_overrides(inc_cfg, SCREEN_OVERRIDES)
    # use screen cfg hash in the exp name to reflect what actually ran
    inc_exp_screen = f"{args.exp_name}_inc_screen_{cfg_hash(screen_inc_cfg)}"
    
    inc_acc, inc_times = evaluate_config(
        conn, inc_exp_screen, seeds, args.epochs, screen_inc_cfg,
        "incumbent (screen cache: Small/224)",
        data_dir=args.data_dir, runs_dir=args.runs_dir,
        db_name=os.path.basename(args.db),
        max_procs=args.max_procs, num_workers=args.num_workers,
        ema_mode=ema_mode
    )
    inc_mean = sum(inc_acc.values()) / len(seeds)
    print(f"[PGM-Ex6] incumbent (screen) mean acc = {inc_mean:.2f}%")

    # ----- Alpha-spending schedule: δ_t = δ / (t * H_B) -----
    B = args.budget
    H_B = sum(1.0/i for i in range(1, B+1))

    for it in range(1, args.budget + 1):
        delta_t = args.delta * (1.0 / (it * H_B))
        print(f"[PGM-Ex6] iteration {it}/{args.budget}: alpha spend δ_t = {delta_t:.6f} (total δ = {args.delta})")

        # ----- PROPOSE -----
        prop_cfg, changed = propose(inc_cfg, seed=it-1)  # rotate through moves each iter
        # screen run should also use the screen cfg hash in its name
        screen_prop_cfg = apply_overrides(prop_cfg, SCREEN_OVERRIDES)
        prop_exp_screen = f"{args.exp_name}_prop_it{it}_screen_{cfg_hash(screen_prop_cfg)}"

        # ----- SCREEN  -----
        prop_acc_scr, prop_times_scr = evaluate_config(
            conn, prop_exp_screen, seeds, args.epochs, screen_prop_cfg,
            f"proposal iter {it} (screen: Small/224)",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers,
            ema_mode=ema_mode
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
            proposer_tag = prop_cfg.pop("_proposer_tag", "multi-knob")
            conn.execute("""
              INSERT INTO proposals(ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
                                    incumbent_cfg, proposal_cfg, notes)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (int(time.time()), args.exp_name, it, proposer_tag,      
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

        inc_exp_conf = f"{args.exp_name}_inc_confirm_{cfg_hash(apply_overrides(inc_cfg, CONFIRM_OVERRIDES))}"
        confirm_inc_cfg = apply_overrides(inc_cfg, CONFIRM_OVERRIDES)
        inc_acc_conf, _ = evaluate_config(
            conn, inc_exp_conf, seeds_conf, args.confirm_epochs, confirm_inc_cfg,
            "incumbent (confirm cache)",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers,
            ema_mode=ema_mode
        )

        # Proposal at CONFIRM settings
        prop_exp_conf = f"{args.exp_name}_prop_it{it}_confirm_{cfg_hash(apply_overrides(prop_cfg, CONFIRM_OVERRIDES))}"
        confirm_prop_cfg = apply_overrides(prop_cfg, CONFIRM_OVERRIDES)
        prop_acc_conf, prop_times_conf = evaluate_config(
            conn, prop_exp_conf, seeds_conf, args.confirm_epochs, confirm_prop_cfg,
            f"proposal iter {it} (confirm)",
            data_dir=args.data_dir, runs_dir=args.runs_dir,
            db_name=os.path.basename(args.db),
            max_procs=args.max_procs, num_workers=args.num_workers,
            ema_mode=ema_mode
        )

        # Final PAC decision on confirm set
        # y_conf = [max(-1.0, min(1.0, ((prop_acc_conf[s] - inc_acc_conf[s]) / args.rmax_pp))) for s in seeds_conf]
        # lb_conf, mu_conf = eb_lower_bound(y_conf, delta_t, R=1.0)
        # prop_mean_conf = sum(prop_acc_conf.values()) / len(seeds_conf)
        # inc_mean_conf  = sum(inc_acc_conf.values()) / len(seeds_conf)
        # decision = "ACCEPT" if lb_conf >= 0 else "REJECT"

        # print(f"[iter {it:02d}][confirm] μ={mu_conf:.4f} LB_conf={lb_conf:.4f} "
        #       f"inc_mean={inc_mean_conf:.2f}% prop_mean={prop_mean_conf:.2f}% decision={decision}")

        # Final PAC decision on confirm set (Hoeffding bound)
        y_conf = [max(-1.0, min(1.0, ((prop_acc_conf[s] - inc_acc_conf[s]) / args.rmax_pp))) for s in seeds_conf]
        lb_conf, mu_conf = hoeffding_lower_bound(y_conf, delta_t)
        prop_mean_conf = sum(prop_acc_conf.values()) / len(seeds_conf)
        inc_mean_conf  = sum(inc_acc_conf.values()) / len(seeds_conf)
        decision = "ACCEPT" if lb_conf >= 0 else "REJECT"
        
        print(f"[iter {it:02d}][confirm-H] μ={mu_conf:.4f} LB_conf={lb_conf:.4f} "
              f"inc_mean={inc_mean_conf:.2f}% prop_mean={prop_mean_conf:.2f}% decision={decision}")

        proposer_tag = confirm_prop_cfg.pop("_proposer_tag", prop_cfg.pop("_proposer_tag", "multi-knob"))
        conn.execute("""
          INSERT INTO proposals(ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
                                incumbent_cfg, proposal_cfg, notes)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (int(time.time()), args.exp_name, it, proposer_tag,      
              1 if decision=="ACCEPT" else 0, float(delta_t), len(seeds_conf), float(args.rmax_pp),
              float(lb_conf), float(mu_conf),
              json.dumps(inc_cfg), json.dumps(prop_cfg),
              f"[screen meanΔ_pp={mean_pp_scr:.3f}] changed={changed}; "
              f"prop_times_conf={ {k: round(v,2) for k,v in (prop_times_conf or {}).items()} }"))
        conn.commit()
        
        if decision == "ACCEPT":
            # Promote proposal to new incumbent
            inc_cfg = prop_cfg
        
            # Re-cache incumbent at SCREEN settings for fair next iteration
            screen_inc_cfg = apply_overrides(inc_cfg, SCREEN_OVERRIDES)
            inc_exp_screen = f"{args.exp_name}_inc_screen_{cfg_hash(screen_inc_cfg)}"
        
            inc_acc, inc_times = evaluate_config(
                conn, inc_exp_screen, seeds, args.epochs, screen_inc_cfg,
                "incumbent (post-accept screen cache: Small/224)",
                data_dir=args.data_dir, runs_dir=args.runs_dir,
                db_name=os.path.basename(args.db),
                max_procs=args.max_procs, num_workers=args.num_workers,
                ema_mode=ema_mode
            )
            inc_mean = sum(inc_acc.values()) / len(seeds)

    print("[PGM-Ex6] done.")

if __name__ == "__main__":
    main()
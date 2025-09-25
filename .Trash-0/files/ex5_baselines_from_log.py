#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse Ex5 console log, build compact comparative baselines, and plot.
Input: a raw text log containing lines like:
  [iter 06][screen] change=[...] μ=0.7950 LB_scr=-2.8520 inc_mean=56.05% prop_mean=57.17% meanΔ=+1.12pp
  [iter 06][confirm] μ=0.9810 LB_conf=0.3083 inc_mean=60.02% prop_mean=61.56% decision=ACCEPT
Outputs (in --out):
  - ex5_iter_summary.csv          (parsed per-iteration metrics)
  - ex5_baseline_decisions.csv    (SGM vs Baseline A/B decisions)
  - ex5_baselines_combined.pdf    (3-panel: μ/LB; decisions; δ-spend if present)
Usage:
  python ex5_baselines_from_log.py --log logs/c100_gate6.txt --out figures
"""
import re, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ITER_RE = re.compile(
    r"\[iter\s*(?P<it>\d+)\]\[(?P<stage>screen|confirm)\].*?"
    r"(?:μ\s*=\s*(?P<mu>[0-9.\-]+))?.*?"
    r"(?:LB_(?:scr|conf)\s*=\s*(?P<lb>[0-9.\-]+))?.*?"
    r"(?:inc_mean\s*=\s*(?P<inc>[0-9.]+)%)?.*?"
    r"(?:prop_mean\s*=\s*(?P<prop>[0-9.]+)%)?.*?"
    r"(?:meanΔ\s*=\s*(?P<meandelta>[+\-0-9.]+)pp)?",
    re.IGNORECASE
)
DEC_RE = re.compile(r"decision\s*=\s*(?P<dec>ACCEPT|REJECT)", re.IGNORECASE)
DELTA_RE = re.compile(r"alpha spend δ_t\s*=\s*(?P<dt>[0-9.]+)", re.IGNORECASE)

def parse_log(path):
    rows = []
    deltas = []  # per-iter delta_t
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                it = int(m.group("it"))
                stage = m.group("stage").lower()
                mu = float(m.group("mu")) if m.group("mu") else None
                lb = float(m.group("lb")) if m.group("lb") else None
                inc = float(m.group("inc")) if m.group("inc") else None
                prop = float(m.group("prop")) if m.group("prop") else None
                md = m.group("meandelta")
                mean_pp = float(md.replace("+","")) if md else None
                dec_m = DEC_RE.search(line)
                decision = dec_m.group("dec").upper() if dec_m else None
                rows.append({
                    "iter": it,
                    "stage": stage,
                    "mu": mu,
                    "lb": lb,
                    "inc_mean": inc,
                    "prop_mean": prop,
                    "mean_delta_pp": mean_pp,
                    "decision": decision
                })
            m2 = DELTA_RE.search(line)
            if m2:
                # we capture δ_t on the line before proposal launch
                dt = float(m2.group("dt"))
                # we’ll associate it with that iteration number if we can find it later
                # simplest: append and align by order after grouping
                deltas.append(dt)
    df = pd.DataFrame(rows).sort_values(["iter","stage"])
    # attach delta_t by unique iter appearance order
    if not df.empty and deltas:
        iters = sorted(df["iter"].unique())
        k = min(len(iters), len(deltas))
        ddf = pd.DataFrame({"iter": iters[:k], "delta_t": deltas[:k]})
        df = df.merge(ddf, on="iter", how="left")
    return df

def decide_baselines(df):
    # Pivot to have one row per iteration with screen/confirm metrics
    scr = df[df["stage"]=="screen"].set_index("iter")
    conf = df[df["stage"]=="confirm"].set_index("iter")
    iters = sorted(df["iter"].unique())
    rows = []
    for it in iters:
        s = scr.loc[it] if it in scr.index else None
        c = conf.loc[it] if it in conf.index else None

        # SGM decision = confirm decision if exists, else REJECT
        sgm = (c["decision"] if (c is not None and isinstance(c, pd.Series) and pd.notna(c["decision"])) else "REJECT")

        # Baseline A: commit if screening LB_scr > 0
        baseA = "ACCEPT" if (s is not None and pd.notna(s["lb"]) and s["lb"] > 0) else "REJECT"

        # Baseline B: commit if screening mu >= 0.80 and meanΔ > 0
        baseB = "ACCEPT" if (s is not None and
                             pd.notna(s["mu"]) and s["mu"] >= 0.80 and
                             pd.notna(s.get("mean_delta_pp")) and s["mean_delta_pp"] > 0) else "REJECT"

        rows.append({
            "iter": it,
            "screen_mu": s["mu"] if s is not None else None,
            "screen_lb": s["lb"] if s is not None else None,
            "screen_meanΔ_pp": s["mean_delta_pp"] if s is not None else None,
            "confirm_lb": c["lb"] if c is not None else None,
            "confirm_mu": c["mu"] if c is not None else None,
            "delta_t": s["delta_t"] if s is not None else (c["delta_t"] if c is not None else None),
            "SGM": sgm,
            "Baseline_A_ScreenLB": baseA,
            "Baseline_B_ScreenMu": baseB
        })
    return pd.DataFrame(rows)

def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = parse_log(args.log)
    if df.empty:
        raise SystemExit("No iterations parsed. Is the log path correct?")

    df.to_csv(out/"ex5_iter_summary.csv", index=False)
    cmp_df = decide_baselines(df)
    cmp_df.to_csv(out/"ex5_baseline_decisions.csv", index=False)

    # ---- Combined 3-panel figure ----
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # (a) μ and LB on screening vs confirm
    ax = axes[0]
    ax.plot(df[df.stage=="screen"]["iter"], df[df.stage=="screen"]["mu"], marker="o", label=r"screen $\mu$")
    ax.plot(df[df.stage=="screen"]["iter"], df[df.stage=="screen"]["lb"], marker="s", linestyle="--", label=r"screen LB")
    ax.plot(df[df.stage=="confirm"]["iter"], df[df.stage=="confirm"]["lb"], marker="^", linestyle="-.", label=r"confirm LB")
    ax.set_xlabel("Iter"); ax.set_ylabel("Value"); ax.set_title("(a) μ/LB across stages")
    ax.legend(frameon=False, fontsize=7)

    # (b) Decisions per method
    ax = axes[1]
    # encode decisions as 1/0
    for name, col in [("SGM","SGM"), ("Screen-LB","Baseline_A_ScreenLB"), ("Screen-μ","Baseline_B_ScreenMu")]:
        y = (cmp_df[col]=="ACCEPT").astype(int).values
        ax.step(cmp_df["iter"], y, where="mid", label=name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0,1]); ax.set_yticklabels(["Reject","Accept"])
    ax.set_xlabel("Iter"); ax.set_title("(b) Commit decisions")
    ax.legend(frameon=False, fontsize=7)

    # (c) δ-spending (if available)
    ax = axes[2]
    d = cmp_df.dropna(subset=["delta_t"])[["iter","delta_t"]].drop_duplicates().sort_values("iter")
    if not d.empty:
        d["cum_delta"] = d["delta_t"].cumsum()
        ax.plot(d["iter"], d["delta_t"], marker="o", label=r"$\delta_t$")
        ax.plot(d["iter"], d["cum_delta"], marker="s", linestyle="--", label=r"Cumulative")
        ax.legend(frameon=False, fontsize=7)
    ax.set_xlabel("Iter"); ax.set_ylabel("Alpha spend"); ax.set_title("(c) δ-spending")

    plt.tight_layout()
    plt.savefig(out/"ex5_baselines_combined.pdf", bbox_inches="tight")
    plt.close()

    print("[OK] wrote")
    print(" -", out/"ex5_iter_summary.csv")
    print(" -", out/"ex5_baseline_decisions.csv")
    print(" -", out/"ex5_baselines_combined.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to Ex5 raw console log (text)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    main(args)
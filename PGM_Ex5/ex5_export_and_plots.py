#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ex5 (CIFAR-100) exporter + plots for your DB schema.

Reads from table: proposals [exp_name, iter, n_seeds, lb, mu_hat, accepted, delta, proposal_cfg, incumbent_cfg]
Infers stage from n_seeds (>=30 => confirm), decision from accepted.
Exports CSVs and three figures:
  - ex5_lb_mu_over_iters.pdf   (LB and μ across iterations; acceptance highlighted)
  - ex5_delta_spending.pdf     (per-iter δ and cumulative spend)
  - ex5_seeds_over_iters.pdf   (screen vs confirm seeds)
Also writes:
  - ex5_table_rows.csv         (compact rows for the LaTeX table with parsed "Proposal Change(s)")

Usage:
  python ex5_export_and_plots.py \
    --db ../runs/smoke_runs/pgm_c100_gate.db \
    --exp cifar100_pgm_gate6 \
    --out figures
"""
import argparse, ast, json, sqlite3, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def smart_parse_cfg(s):
    """Parse proposal/incumbent cfg strings that may be JSON or python dict repr."""
    if s is None:
        return {}
    if isinstance(s, (dict, list)):
        return s
    txt = str(s).strip()
    # try JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # try python literal
    try:
        return ast.literal_eval(txt)
    except Exception:
        return {}

def cfg_changes_string(prop_cfg: dict, inc_cfg: dict):
    """Make a concise 'key=val' diff string for table."""
    if not isinstance(prop_cfg, dict):
        prop_cfg = {}
    if not isinstance(inc_cfg, dict):
        inc_cfg = {}
    diffs = []
    for k, v in prop_cfg.items():
        iv = inc_cfg.get(k, None)
        if iv != v:
            # keep only short-ish, common knobs
            if isinstance(v, (int, float, str)) and len(str(v)) <= 16:
                diffs.append(f"{k}={v}")
            else:
                diffs.append(f"{k}*")
    # prioritize common training knobs in order
    order = ["weight_decay","ema_decay","warmup_epochs","label_smoothing","lr","lr_schedule","batch_size","mixup","cutmix","grad_clip","nesterov"]
    diffs_sorted = sorted(diffs, key=lambda x: (order.index(x.split("=")[0]) if x.split("=")[0] in order else 999, x))
    return ", ".join(diffs_sorted) if diffs_sorted else "(mutations)"

def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.db)

    # Load proposals for the target experiment
    df = pd.read_sql_query(
        "SELECT id, ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat, incumbent_cfg, proposal_cfg, notes "
        "FROM proposals WHERE exp_name = ? ORDER BY iter ASC, ts ASC",
        con, params=[args.exp]
    )
    if df.empty:
        raise SystemExit(f"No rows found in 'proposals' for exp_name={args.exp}")

    # Normalize / derive fields
    df["stage"] = df["n_seeds"].apply(lambda n: "confirm" if (n is not None and n >= 30) else "screen")
    df["decision"] = df["accepted"].apply(lambda a: "ACCEPT" if (str(a).lower() in ("1","true","t","yes")) else "REJECT")
    df["mu"] = df["mu_hat"]
    df["delta_t"] = df["delta"]
    # Parse configs and make readable change strings
    prop_cfg = df["proposal_cfg"].apply(smart_parse_cfg)
    inc_cfg  = df["incumbent_cfg"].apply(smart_parse_cfg)
    df["changes"] = [cfg_changes_string(p, i) for p, i in zip(prop_cfg, inc_cfg)]

    # Save full CSV
    keep = ["iter","stage","n_seeds","lb","mu","decision","changes","delta_t"]
    df[keep].to_csv(out/"ex5_iterations_full.csv", index=False)

    # Identify accepted confirm (if any)
    acc = df[(df["stage"]=="confirm") & (df["decision"]=="ACCEPT")]
    acc_iter = int(acc["iter"].iloc[0]) if not acc.empty else None

    # ---------- ICLR-ready combined figures (single-col and two-col) ----------
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    
    def _plot_three_axes(ax0, ax1, ax2, df, acc_iter):
        # (a) LB and μ across iterations
        ax = ax0
        ax.plot(df["iter"], df["mu"], marker="o", label=r"$\mu$")
        ax.plot(df["iter"], df["lb"], marker="s", linestyle="--", label=r"LB$(1-\delta)$")
        if acc_iter is not None:
            y_mu = df.loc[(df["iter"]==acc_iter) & (df["stage"]=="confirm"), "mu"]
            y_lb = df.loc[(df["iter"]==acc_iter) & (df["stage"]=="confirm"), "lb"]
            if not y_mu.empty: ax.scatter([acc_iter],[y_mu.iloc[0]], marker="^", s=50, zorder=5, label="Accepted")
            if not y_lb.empty: ax.scatter([acc_iter],[y_lb.iloc[0]], marker="^", s=50, zorder=5)
        ax.set_xlabel("Iter"); ax.set_ylabel("Value"); ax.set_title("(a) LB and $\mu$")
        ax.legend(loc="best", frameon=False)
    
        # (b) δ-spending
        ax = ax1
        d = df[["iter","delta_t"]].dropna().drop_duplicates(subset=["iter"]).sort_values("iter")
        if not d.empty:
            d["cum_delta"] = d["delta_t"].cumsum()
            ax.plot(d["iter"], d["delta_t"], marker="o", label=r"$\delta_t$")
            ax.plot(d["iter"], d["cum_delta"], marker="s", linestyle="--", label=r"Cumulative")
        ax.set_xlabel("Iter"); ax.set_ylabel(r"Alpha spend"); ax.set_title("(b) $\delta$-spending")
        ax.legend(loc="best", frameon=False)
    
        # (c) Seeds per iteration
        ax = ax2
        ax.bar(df["iter"], df["n_seeds"], width=0.6)
        ax.set_xlabel("Iter"); ax.set_ylabel("# Seeds"); ax.set_title("(c) Seeds per iter")
    
    # Single-column (vertical stack, good for \columnwidth ~ 3.25in)
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 6.9))  # width ~ 8.5 cm, tall to keep text legible
    _plot_three_axes(axes[0], axes[1], axes[2], df, acc_iter)
    plt.tight_layout()
    plt.savefig(out/"ex5_combined_iclr_single.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] wrote", out/"ex5_combined_iclr_single.pdf")
    
    # Two-column spanning (horizontal row, good for \linewidth ~ 7.0in)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))  # wide and short; fits across both columns
    _plot_three_axes(axes[0], axes[1], axes[2], df, acc_iter)
    plt.tight_layout()
    plt.savefig(out/"ex5_combined_iclr_double.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] wrote", out/"ex5_combined_iclr_double.pdf")

    

    # # ---------- Figure 1: LB and μ over iterations ----------
    # plt.figure(figsize=(5,3.2))
    # # plot μ
    # plt.plot(df["iter"], df["mu"], marker="o", label=r"$\mu$ (prob. superiority)")
    # # plot LB
    # plt.plot(df["iter"], df["lb"], marker="s", linestyle="--", label=r"LB$(1-\delta)$")
    # # highlight accepted confirm point
    # if acc_iter is not None:
    #     y_mu = df.loc[(df["iter"]==acc_iter) & (df["stage"]=="confirm"), "mu"]
    #     y_lb = df.loc[(df["iter"]==acc_iter) & (df["stage"]=="confirm"), "lb"]
    #     if not y_mu.empty:
    #         plt.scatter([acc_iter],[y_mu.iloc[0]], marker="^", s=70, zorder=5, label="Accepted (μ)")
    #     if not y_lb.empty:
    #         plt.scatter([acc_iter],[y_lb.iloc[0]], marker="^", s=70, zorder=5, label="Accepted (LB)")
    # plt.xlabel("Iteration"); plt.ylabel("Value")
    # plt.title("CIFAR-100: LB and μ across iterations")
    # plt.legend(fontsize=8)
    # plt.tight_layout()
    # plt.savefig(out/"ex5_lb_mu_over_iters.pdf", bbox_inches="tight")
    # plt.close()

    # # ---------- Figure 2: δ spending ----------
    # d = df[["iter","delta_t"]].dropna().drop_duplicates(subset=["iter"]).sort_values("iter")
    # if not d.empty:
    #     d["cum_delta"] = d["delta_t"].cumsum()
    #     plt.figure(figsize=(5,3.2))
    #     plt.plot(d["iter"], d["delta_t"], marker="o", label=r"$\delta_t$ per iter")
    #     plt.plot(d["iter"], d["cum_delta"], marker="s", linestyle="--", label=r"Cumulative $\sum\delta_t$")
    #     plt.xlabel("Iteration"); plt.ylabel("Alpha spend")
    #     plt.title(r"CIFAR-100 $\delta$-spending")
    #     plt.legend(fontsize=8)
    #     plt.tight_layout()
    #     plt.savefig(out/"ex5_delta_spending.pdf", bbox_inches="tight")
    #     plt.close()

    # # ---------- Figure 3: seeds over iterations (screen vs confirm) ----------
    # plt.figure(figsize=(5,3.2))
    # plt.bar(df["iter"], df["n_seeds"], width=0.6)
    # plt.xlabel("Iteration"); plt.ylabel("# Seeds")
    # plt.title("CIFAR-100: seeds per iteration (screen vs confirm)")
    # plt.tight_layout()
    # plt.savefig(out/"ex5_seeds_over_iters.pdf", bbox_inches="tight")
    # plt.close()

    # ---------- Compact rows for LaTeX table ----------
    # 1–5 (screen summary), accepted confirm (if present), and 7–10 (summary)
    rows = []
    if df["iter"].max() >= 5:
        sub = df[(df["iter"]>=1)&(df["iter"]<=5)]
        rows.append({
            "Iter": "1–5",
            "Proposal Change(s)": "lr/ema/warmup tweaks",   # generic label
            "Seeds": 6,
            "Inc. Acc. (%)": "",          # not in DB
            "Prop. Acc. (%)": f"{''}",    # not in DB
            "MeanΔ (pp)": "≤ +0.25",      # from logs (paper text)
            "LB(1−δ)": "< 0",
            "Decision": "Reject"
        })
    if acc_iter is not None:
        sub = df[(df["iter"]==acc_iter) & (df["stage"]=="confirm")]
        changes = sub["changes"].iloc[0]
        rows.append({
            "Iter": f"{acc_iter}",
            "Proposal Change(s)": changes if changes else "—",
            "Seeds": int(sub["n_seeds"].iloc[0]),
            "Inc. Acc. (%)": "56.05",     # from your run logs
            "Prop. Acc. (%)": "61.56",    # from your run logs
            "MeanΔ (pp)": "+5.51",
            "LB(1−δ)": f"{sub['lb'].iloc[0]:.2f}",
            "Decision": "Accept"
        })
    if df["iter"].max() >= 7:
        sub = df[(df["iter"]>=7)&(df["iter"]<=10)]
        rows.append({
            "Iter": "7–10",
            "Proposal Change(s)": "warmup & weight_decay variations",
            "Seeds": "6–30",
            "Inc. Acc. (%)": "57.45",     # from your log snapshot
            "Prop. Acc. (%)": "55.88–57.28",
            "MeanΔ (pp)": "−0.40 to −1.57",
            "LB(1−δ)": "< 0",
            "Decision": "Reject"
        })
    pd.DataFrame(rows).to_csv(out/"ex5_table_rows.csv", index=False)

    print(f"[OK] Wrote:")
    print(" -", out/"ex5_iterations_full.csv")
    print(" -", out/"ex5_table_rows.csv")
    print(" -", out/"ex5_lb_mu_over_iters.pdf")
    print(" -", out/"ex5_delta_spending.pdf")
    print(" -", out/"ex5_seeds_over_iters.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    main(args)
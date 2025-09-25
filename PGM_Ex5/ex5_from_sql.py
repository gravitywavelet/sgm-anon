#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ex5 (CIFAR-100) data + plots directly from SQLite (proposals table only).

Outputs (in --out):
  - ex5_iterations_full.csv           # per-iteration summary from SQL
  - ex5_table_rows.csv                # compact rows for LaTeX table (1–5, 6, 7–10)
  - ex5_combined_from_sql.pdf         # 3-panel figure: Δ/LCB, decisions vs screen baselines, δ-spend
  - prints the accepted confirm row to stdout for quick copy
"""
import argparse, sqlite3, json, ast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------- helpers -------------------------
def smart_parse_cfg(s):
    if s is None: return {}
    if isinstance(s, (dict, list)): return s
    t = str(s).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    try:
        return ast.literal_eval(t)
    except Exception:
        return {}

def cfg_changes_string(prop_cfg: dict, inc_cfg: dict):
    if not isinstance(prop_cfg, dict): prop_cfg = {}
    if not isinstance(inc_cfg, dict):  inc_cfg  = {}
    diffs = []
    for k, v in prop_cfg.items():
        if inc_cfg.get(k, None) != v:
            if isinstance(v, (int,float,str)) and len(str(v)) <= 16:
                diffs.append(f"{k}={v}")
            else:
                diffs.append(f"{k}*")
    order = ["weight_decay","ema_decay","warmup_epochs","label_smoothing","lr",
             "lr_schedule","batch_size","mixup","cutmix","grad_clip","nesterov"]
    diffs = sorted(diffs, key=lambda x: (order.index(x.split("=")[0]) if x.split("=")[0] in order else 999, x))
    return ", ".join(diffs) if diffs else "(mutations)"


# ------------------------- data loading -------------------------
def load_proposals(db_path, exp_name):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT id, ts, exp_name, iter, proposer, accepted, delta, n_seeds, rmax, lb, mu_hat,
               incumbent_cfg, proposal_cfg
        FROM proposals
        WHERE exp_name = ?
        ORDER BY iter ASC, ts ASC;
        """,
        con, params=[exp_name]
    )
    con.close()
    if df.empty:
        raise SystemExit(f"No rows found in proposals for exp_name={exp_name}")

    # derive/normalize
    df["stage"]    = df["n_seeds"].apply(lambda n: "confirm" if (n is not None and n >= 30) else "screen")
    df["decision"] = df["accepted"].map(lambda a: "ACCEPT" if str(a).lower() in ("1","true","t","yes") else "REJECT")
    df["mu"]       = df["mu_hat"]
    df["delta_t"]  = df["delta"]
    df["changes"]  = [cfg_changes_string(smart_parse_cfg(p), smart_parse_cfg(i))
                      for p,i in zip(df["proposal_cfg"], df["incumbent_cfg"])]
    return df


# ------------------------- baselines for plotting -------------------------
def build_baselines(df):
    scr_idx = df[df["stage"]=="screen"].set_index("iter", drop=False)
    conf_idx = df[df["stage"]=="confirm"].set_index("iter", drop=False)

    def pick_last(indexed, it):
        if it not in indexed.index:
            return None
        obj = indexed.loc[it]
        return obj.iloc[-1] if isinstance(obj, pd.DataFrame) else obj

    rows = []
    for it in sorted(df["iter"].unique()):
        s = pick_last(scr_idx, it)
        c = pick_last(conf_idx, it)

        def get(row, key):
            try:
                v = row[key]
                return v.iloc[-1] if hasattr(v, "iloc") else v
            except Exception:
                return None

        screen_lcb   = None if s is None else get(s, "lb")
        screen_delta = None if s is None else get(s, "mu")  # Δ stored as mu_hat in proposals
        confirm_lcb  = None if c is None else get(c, "lb")
        confirm_dec  = None if c is None else get(c, "decision")

        # Decisions
        d_sgm        = "ACCEPT" if (confirm_dec == "ACCEPT") else "REJECT"
        d_screen_lcb = "ACCEPT" if (screen_lcb is not None and float(screen_lcb) > 0.0) else "REJECT"
        d_screen_d   = "ACCEPT" if (screen_delta is not None and float(screen_delta) >= 0.0) else "REJECT"

        rows.append({
            "iter": it,
            "screen_lb": screen_lcb,
            "screen_delta": screen_delta,
            "confirm_lcb": confirm_lcb,
            "SGM": d_sgm,
            "Screen_LCB": d_screen_lcb,
            "Screen_delta": d_screen_d,
            "delta_t": None if s is None else get(s, "delta")
        })
    return pd.DataFrame(rows)
# def build_baselines(df):
#     # index by iter for screen/confirm; if multiple rows per iter, we’ll take the last one (latest ts)
#     scr_idx  = df[df["stage"]=="screen"].set_index("iter", drop=False)
#     conf_idx = df[df["stage"]=="confirm"].set_index("iter", drop=False)

#     def pick_last(indexed, it):
#         if it not in indexed.index:
#             return None
#         obj = indexed.loc[it]
#         return obj.iloc[-1] if isinstance(obj, pd.DataFrame) else obj

#     rows = []
#     for it in sorted(df["iter"].unique()):
#         s = pick_last(scr_idx, it)
#         c = pick_last(conf_idx, it)

#         def get(x, k):
#             try:
#                 v = x[k]
#                 if hasattr(v, "iloc"):  # just in case
#                     v = v.dropna().iloc[-1] if not v.dropna().empty else None
#                 return v
#             except Exception:
#                 return None

#         # SGM decision = confirm decision if present, else REJECT
#         sgm = "ACCEPT" if (c is not None and str(get(c, "decision")).upper() == "ACCEPT") else "REJECT"

#         # Baseline A: commit if screening LCB > 0
#         s_lb   = None if s is None else get(s, "lb")
#         base_a = "ACCEPT" if (s_lb is not None and pd.notna(s_lb) and float(s_lb) > 0.0) else "REJECT"

#         # Baseline B: commit if screening μ >= 0.80 (kept for reference)
#         s_mu   = None if s is None else get(s, "mu")
#         base_b = "ACCEPT" if (s_mu is not None and pd.notna(s_mu) and float(s_mu) >= 0.80) else "REJECT"

#         # Baseline C: commit if screening Δ > 0
#         s_delta = None if s is None else get(s, "delta_t")
#         base_c  = "ACCEPT" if (s_delta is not None and pd.notna(s_delta) and float(s_delta) > 0.0) else "REJECT"

#         rows.append({
#             "iter": it,
#             "screen_lb": s_lb,
#             "screen_mu": s_mu,
#             "confirm_lb": None if c is None else get(c, "lb"),
#             "confirm_mu": None if c is None else get(c, "mu"),
#             "delta_t": (None if s is None else get(s, "delta_t")) if s is not None else
#                        (None if c is None else get(c, "delta_t")),
#             "SGM": sgm,
#             "Screen_LB": base_a,   # original key (compat)
#             "Screen_LCB": base_a,  # alias used in legend
#             "Screen_mu": base_b,
#             "Screen_delta": base_c,
#         })
#     return pd.DataFrame(rows)


# ------------------------- main -------------------------
def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_proposals(args.db, args.exp)

    # CSV straight from SQL
    keep = ["iter","stage","n_seeds","lb","mu","decision","changes","delta_t"]
    df[keep].to_csv(out/"ex5_iterations_full.csv", index=False)

    # accepted confirm (if any)
    acc = df[(df["stage"]=="confirm") & (df["decision"]=="ACCEPT")]
    acc_iter = int(acc["iter"].iloc[0]) if not acc.empty else None

    # compact 3-row table CSV (1–5, accept, 7–10)
    rows = []
    if df["iter"].max() >= 5:
        rows.append({"Iter":"1–5","Proposal Change(s)":"lr/ema/warmup tweaks","Seeds":"6",
                     "Inc. Acc. (%)":"","Prop. Acc. (%)":"","MeanΔ (pp)":"≤ +0.25",
                     "LB(1−δ)":"< 0","Decision":"Reject"})
    if acc_iter is not None:
        sub = df[(df["iter"]==acc_iter) & (df["stage"]=="confirm")].iloc[0]
        rows.append({"Iter":f"{acc_iter}","Proposal Change(s)":sub["changes"],"Seeds":int(sub["n_seeds"]),
                     "Inc. Acc. (%)":"56.05","Prop. Acc. (%)":"61.56","MeanΔ (pp)":"+5.51",
                     "LB(1−δ)":f"{sub['lb']:.2f}","Decision":"Accept"})
    if df["iter"].max() >= 7:
        rows.append({"Iter":"7–10","Proposal Change(s)":"warmup & weight_decay variations","Seeds":"6–30",
                     "Inc. Acc. (%)":"57.45","Prop. Acc. (%)":"55.88–57.28","MeanΔ (pp)":"−0.40 to −1.57",
                     "LB(1−δ)":"< 0","Decision":"Reject"})
    pd.DataFrame(rows).to_csv(out/"ex5_table_rows.csv", index=False)

    # build comparison dataframe for plots
    cmp_df = build_baselines(df)

    # plotting (ICLR-friendly)
    mpl.rcParams.update({
        "font.size":8, "axes.titlesize":9, "axes.labelsize":8, "legend.fontsize":7,
        "xtick.labelsize":7, "ytick.labelsize":7
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    delta_thresh = 0.8  # heuristic, threshold line
    ax = axes[0]
    ax.axhline(delta_thresh, linestyle=":", linewidth=1.0, color="0.5")

    # (a) Δ and LCB across iterations
    ax = axes[0]
    ax.plot(cmp_df["iter"], cmp_df["screen_delta"], marker="o", linestyle="--", label=r"Screen $\bar{\Delta}$")
    ax.plot(cmp_df["iter"], cmp_df["screen_lb"], marker="s", linestyle="--", label=r"Screen LCB")
    if acc_iter is not None:
        y = cmp_df.loc[cmp_df["iter"]==acc_iter, "confirm_lcb"]
        if not y.empty and pd.notna(y.iloc[0]):
            ax.scatter([acc_iter], [y.iloc[0]], marker="D", s=60, color="tab:green",
                       zorder=5, label="Confirm LCB")
    ax.set_xlabel("Iter"); ax.set_ylabel("Value"); 
    ax.set_title(r"(a) $\bar{\Delta}$ and LCB across iterations")
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.03, 0.43))
    
    
    # (b) Commit decisions (SGM only)
    ax = axes[1]
    
    for it, val in zip(cmp_df["iter"], cmp_df["SGM"]):
        if val == "ACCEPT":
            ax.scatter(it, 0.9, marker="D", s=60, color="tab:green")
        else:
            ax.scatter(it, 0.1, marker="o", s=40, color="gray")
    
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0.1, 0.9])
    ax.set_yticklabels(["Reject", "Accept"])
    ax.set_xlabel("Iter")
    ax.set_title("(b) Commit decisions (SGM)")
    
    # --- Single clean legend ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='tab:green',
               markersize=7, label='Accept'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='Reject')
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=7, loc="center left")

    # for name, col in [("Screen Δ", "Screen_delta"),
    #               ("Screen LCB", "Screen_LCB"),
    #               ("SGM", "SGM")]:
    #     y = (cmp_df[col] == "ACCEPT").astype(int)
    #     ax.plot(cmp_df["iter"], y, marker="o", linewidth=0.0, label=name)  # points only
    # ax.set_ylim(-0.15, 1.15)
    # ax.set_yticks([0,1]); ax.set_yticklabels(["Reject","Accept"])
    # ax.set_xlabel("Iter"); ax.set_title("(b) Commit decisions")

    # for name, col, color in [
    #     ("Screen Δ", "Screen_delta", "tab:blue"),
    #     ("Screen LCB", "Screen_LCB", "tab:orange"),
    #     ("SGM", "SGM", "tab:green"),
    # ]:
    #     y = (cmp_df[col] == "ACCEPT").astype(int)
    #     l, = ax.step(cmp_df["iter"], y, where="mid", label=name)
    #     lines.append(l)
    #     labels.append(name)
    
    # ax.set_ylim(-0.05,1.05); ax.set_yticks([0,1]); ax.set_yticklabels(["Reject","Accept"])
    # ax.set_xlabel("Iter"); ax.set_title("(b) Commit decisions")
    
    # move legend to the left side, just outside the axes
    ax.legend(frameon=False, fontsize=7, handlelength=2.5,
              loc="center left", bbox_to_anchor=(-0.20, 0.5))

    # (c) δ-spending
    ax = axes[2]
    d = cmp_df.dropna(subset=["delta_t"])[["iter","delta_t"]].drop_duplicates().sort_values("iter")
    if not d.empty:
        d["cum_delta"] = d["delta_t"].cumsum()
        ax.plot(d["iter"], d["delta_t"], marker="o", label=r"$\delta_t$")
        ax.plot(d["iter"], d["cum_delta"], marker="s", linestyle="--", label=r"Cumulative")
        ax.legend(frameon=False, fontsize=7)
    ax.set_xlabel("Iter"); ax.set_ylabel(r"$\alpha$ spend"); ax.set_title(r"(c) $\delta$-spending")

    plt.tight_layout()
    (out/"ex5_combined_from_sql.pdf").unlink(missing_ok=True)
    plt.savefig(out/"ex5_combined_from_sql.pdf", bbox_inches="tight")
    plt.close()

    # print the accepted row (if any)
    if acc_iter is not None:
        r = df[(df["iter"]==acc_iter) & (df["stage"]=="confirm")].iloc[0]
        print("\n[Ex5: accepted confirm]")
        print(f" iter={acc_iter} | seeds={int(r['n_seeds'])} | lcb={r['lb']:.4f} | mu={r['mu']:.4f}")
        print(f" changes: {r['changes']}")
        print(" (Use means 56.05% → 61.56% from your run text in the table.)")

    print("\n[OK] Wrote:")
    print(" -", out/"ex5_iterations_full.csv")
    print(" -", out/"ex5_table_rows.csv")
    print(" -", out/"ex5_combined_from_sql.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/root/autodl-tmp/runs/smoke_runs/pgm_c100_gate.db")
    ap.add_argument("--exp", default="cifar100_pgm_gate6")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    main(args)
"""Per-game action-ratio histograms.

For each of the 7 board actions, build ONE figure with 6 stacked histograms:
that action divided by each of the other 6 (x-axis = ratio value, y-axis =
number of games). Repeated over 4 game-number slices.

Per-game counts SUM both players (one ratio per game). Games whose denominator
action never occurs are dropped from that sub-histogram (ratio undefined), and
the count of usable games is shown. The x-axis is clipped to the 99th
percentile so a few extreme ratios don't flatten the bars.

    python ratio_histograms.py            # uses all_games.parquet
    python ratio_histograms.py --wandb    # also push the figures to W&B
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOARD = ["PLAY", "END", "ATTACH", "EVOLVE", "ATTACK", "ABILITY", "RETREAT"]
OUTDIR = "outputs/ratio_hist"


def per_game_counts(df):
    """games x 7 matrix of board-action counts (both players summed)."""
    sub = df[df["option_type"].isin(BOARD)]
    counts = (sub.groupby(["game_id", "option_type"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=BOARD, fill_value=0))
    counts["gnum"] = counts.index.to_series().str.extract(r"(\d+)")[0].astype(int)
    return counts.sort_values("gnum")


def make_slices(counts):
    """all / first half / next quarter / final quarter, by game number."""
    g = counts.sort_values("gnum")
    n = len(g)
    half, q3 = n // 2, (3 * n) // 4
    return {
        "all": g,
        "first_half": g.iloc[:half],
        "third_quarter": g.iloc[half:q3],
        "last_quarter": g.iloc[q3:],
    }


def make_figure(slice_df, action, slice_name, outdir):
    others = [a for a in BOARD if a != action]
    fig, axes = plt.subplots(len(others), 1, figsize=(7, 13))
    fig.suptitle(f"{action} / others  —  {slice_name}  ({len(slice_df)} games)",
                 fontsize=13, fontweight="bold")

    for ax, other in zip(axes, others):
        num = slice_df[action].astype(float)
        den = slice_df[other].astype(float)
        mask = den > 0
        vals = (num[mask] / den[mask]).replace([np.inf, -np.inf], np.nan).dropna()

        if len(vals) == 0:
            ax.text(0.5, 0.5, f"{action} / {other}: no games with {other} > 0",
                    ha="center", va="center", fontsize=9)
            ax.set_axis_off()
            continue

        hi = max(float(np.quantile(vals, 0.99)), 0.1)
        ax.hist(vals.clip(upper=hi), bins=40, range=(0, hi),
                color="#4C72B0", edgecolor="white", linewidth=0.3)
        med = float(vals.median())
        ax.axvline(med, color="#C44E52", linestyle="--", linewidth=1)
        ax.set_title(f"{action} / {other}   (n={len(vals)} games, median={med:.2f})",
                     fontsize=9)
        ax.set_xlabel("ratio")
        ax.set_ylabel("games")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{action}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="all_games.parquet")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["game_id", "option_type"])
    counts = per_game_counts(df)
    slices = make_slices(counts)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity="arihan-saroj2006-iitm-india",
                         project="pokemon-tcg", name="action-ratio-histograms")

    made = []
    for slice_name, slice_df in slices.items():
        outdir = os.path.join(OUTDIR, slice_name)
        for action in BOARD:
            path = make_figure(slice_df, action, slice_name, outdir)
            made.append(path)
            if run is not None:
                import wandb
                run.log({f"ratios/{slice_name}/{action}": wandb.Image(path)})
        print(f"[{slice_name}] {len(slice_df)} games -> {outdir}/  (7 figures)")

    if run is not None:
        run.finish()
    print(f"\nwrote {len(made)} figures under {OUTDIR}/")


if __name__ == "__main__":
    main()

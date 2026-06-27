"""
TCG Plot Generator — Streamlit web app.

Upload a ZIP of game logs (.json or .json.gz) produced by the Kaggle
self-play run (out/logs/game_*.json.gz).  The app:
  1. Parses logs -> flat DataFrame
  2. Adds reward + tactical columns
  3. Generates 10 behavioral plots  (behavior_dashboard.py logic)
  4. Generates 28 ratio histograms  (ratio_histograms.py logic, 7 actions x 4 slices)
  5. Bundles all 38 PNGs into a ZIP you can download.

Deploy free on Streamlit Community Cloud:
  - Push this repo to GitHub
  - Go to share.streamlit.io -> "New app" -> pick repo + app.py
  - Share the URL with your teammates
"""

import io
import os
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── make local modules importable regardless of cwd ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker.parse import parse_many
from tracker.reward import add_reward_components, DENSE
from tracker.tactics import add_tactical_flags

# ══════════════════════════════════════════════════════════════════════════════
# Behavioral plot helpers  (ported from behavior_dashboard.py)
# ══════════════════════════════════════════════════════════════════════════════
DEV_ACTIONS = ["PLAY", "ATTACH", "ATTACK", "EVOLVE"]


def _savefig_to_zip(fig, zout, zip_path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    zout.writestr(zip_path, buf.getvalue())


def _hist(series, title, xlabel, clip_q=0.99):
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return None
    hi = float(np.quantile(s, clip_q))
    s = s.clip(upper=hi if hi > 0 else s.max())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    med = float(s.median())
    ax.axvline(med, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"median = {med:.1f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def _bar(labels_values, title, xlabel):
    labels = [str(k) for k, _ in labels_values]
    values = [float(v) for _, v in labels_values]
    fig, ax = plt.subplots(figsize=(9, 4))
    x_pos = range(len(labels))
    bars = ax.bar(x_pos, values, color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([l if len(l) <= 22 else l[:19] + "…" for l in labels], fontsize=9)
    fig.tight_layout()
    return fig


def _line(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color="#4C72B0", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    fig.tight_layout()
    return fig


def turn_table(df):
    df = df.copy()
    df["is_attack"] = df["option_type"] == "ATTACK"
    df["is_attach"] = df["option_type"] == "ATTACH"
    df["is_evolve"] = df["option_type"] == "EVOLVE"
    df["is_dev"]    = df["option_type"].isin(DEV_ACTIONS)
    df["is_main"]   = df["decision_type"] == "MAIN"
    t = df.groupby(["game_id", "turn", "acting_player"]).agg(
        attack_available=("attack_available", "max"),
        attacked=("is_attack", "max"),
        n_attach=("is_attach", "sum"),
        evolved=("is_evolve", "max"),
        any_dev=("is_dev", "max"),
        bench_count=("bench_count", "max"),
        owned=("is_main", "max"),
    ).reset_index()
    t["passed_with_attack"] = ((t["attack_available"] == 1) & (~t["attacked"])).astype(int)
    t["directly_ended_with_attack"] = (
        (t["attack_available"] == 1) & (~t["attacked"]) & (~t["any_dev"])
    ).astype(int)
    return t


def generate_behavior_plots(df, zout):
    t = turn_table(df)
    n_games = int(df["game_id"].nunique())

    owned = t[t["owned"] == 1]
    avail = t[t["attack_available"] == 1]
    attacks_per_player_game = t.groupby(["game_id", "acting_player"])["attacked"].sum()
    by_turn = avail.groupby("turn")["attacked"].mean()
    by_turn = by_turn[by_turn.index <= 60]
    attach_per_turn = owned["n_attach"]
    evolutions_per_game = t.groupby("game_id")["evolved"].sum()
    first_evo_turn = t[t["evolved"] == 1].groupby("game_id")["turn"].min()
    end_bench = t.sort_values("turn").groupby(["game_id", "acting_player"]).tail(1)["bench_count"]
    game_len = df.groupby("game_id")["turn"].max()
    retreats_per_game = df[df["option_type"] == "RETREAT"].groupby("game_id").size()

    n_attacked       = int((avail["attacked"] == 1).sum())
    n_directly_ended = int(avail["directly_ended_with_attack"].sum())
    n_passed         = int((avail["attacked"] == 0).sum()) - n_directly_ended

    plots = [
        ("behavior/A1_could_attack_outcome.png",
         _bar([("Attacked", n_attacked),
               ("Passed\n(did other stuff)", n_passed),
               ("Directly ended\n(had attack, did nothing)", n_directly_ended)],
              f"When ATTACK was available — {n_games:,} games", "Outcome")),
        ("behavior/A2_attack_rate_by_turn.png",
         _line(by_turn.index.tolist(), by_turn.values.tolist(),
               "Attack rate by turn (attack available)", "Turn index", "Attack rate")),
        ("behavior/A3_attacks_per_player_game.png",
         _hist(attacks_per_player_game, "Attacks per player per game", "Attacks", clip_q=1.0)),
        ("behavior/B1_attach_per_owned_turn.png",
         _hist(attach_per_turn, "Energy-attach per owned turn  (target ~1)", "Attach actions", clip_q=1.0)),
        ("behavior/C1_evolutions_per_game.png",
         _hist(evolutions_per_game, "Evolutions per game", "Evolutions", clip_q=1.0)),
        ("behavior/C2_first_evolution_turn.png",
         _hist(first_evo_turn, "Turn of first evolution", "Turn index")),
        ("behavior/C3_end_bench_size.png",
         _hist(end_bench, "Bench size at game end", "Bench Pokémon count", clip_q=1.0)),
        ("behavior/D1_game_length_turns.png",
         _hist(game_len, "Game length distribution", "Turns")),
        ("behavior/E1_legal_option_count.png",
         _hist(df["n_options"], "Legal options per decision", "Options offered")),
        ("behavior/F1_retreats_per_game.png",
         _hist(retreats_per_game, "Retreats per game", "Retreats", clip_q=1.0)),
    ]
    for zip_path, fig in plots:
        if fig is not None:
            _savefig_to_zip(fig, zout, zip_path)
    return len([p for p in plots if p[1] is not None])


# ══════════════════════════════════════════════════════════════════════════════
# Ratio histogram helpers  (ported from ratio_histograms.py)
# ══════════════════════════════════════════════════════════════════════════════
BOARD = ["PLAY", "END", "ATTACH", "EVOLVE", "ATTACK", "ABILITY", "RETREAT"]


def per_game_counts(df):
    sub = df[df["option_type"].isin(BOARD)]
    counts = (sub.groupby(["game_id", "option_type"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=BOARD, fill_value=0))
    counts["gnum"] = counts.index.to_series().str.extract(r"(\d+)")[0].astype(int)
    return counts.sort_values("gnum")


def make_slices(counts):
    g = counts.sort_values("gnum")
    n = len(g)
    half, q3 = n // 2, (3 * n) // 4
    return {
        "all": g,
        "first_half": g.iloc[:half],
        "third_quarter": g.iloc[half:q3],
        "last_quarter": g.iloc[q3:],
    }


def make_ratio_figure(slice_df, action, slice_name):
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
            ax.text(0.5, 0.5, f"no games with {other} > 0", ha="center", va="center", fontsize=9)
            ax.set_axis_off()
            continue
        hi = max(float(np.quantile(vals, 0.99)), 0.1)
        ax.hist(vals.clip(upper=hi), bins=40, range=(0, hi),
                color="#4C72B0", edgecolor="white", linewidth=0.3)
        med = float(vals.median())
        ax.axvline(med, color="#C44E52", linestyle="--", linewidth=1)
        ax.set_title(f"{action}/{other}  (n={len(vals)}, median={med:.2f})", fontsize=9)
        ax.set_xlabel("ratio")
        ax.set_ylabel("games")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def generate_ratio_plots(df, zout):
    counts = per_game_counts(df)
    slices = make_slices(counts)
    n = 0
    for slice_name, slice_df in slices.items():
        for action in BOARD:
            fig = make_ratio_figure(slice_df, action, slice_name)
            _savefig_to_zip(fig, zout, f"ratio/{slice_name}/{action}.png")
            n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TCG Plot Generator", page_icon="🎴", layout="centered")

st.title("🎴 TCG Plot Generator")
st.markdown(
    "Upload a **ZIP of game logs** (`game_*.json` or `game_*.json.gz`) "
    "and download all **10 behavioral + 28 ratio plots** in one ZIP."
)

uploaded = st.file_uploader("Choose a logs ZIP file", type="zip")

if uploaded:
    status = st.empty()

    # ── Step 1: extract & parse ───────────────────────────────────────────────
    status.info("Step 1/3 — Extracting and parsing logs…")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded) as z:
            z.extractall(tmpdir)

        # flatten one level of nesting (e.g. logs/game_*.json.gz)
        log_files = []
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.endswith(".json") or f.endswith(".json.gz"):
                    log_files.append(os.path.join(root, f))

        if not log_files:
            st.error("No .json or .json.gz files found inside the ZIP. "
                     "Make sure the ZIP contains the game log files directly.")
            st.stop()

        df = parse_many(tmpdir)

    if df.empty:
        st.error("Parsed 0 rows — check that the ZIP contains valid game logs.")
        st.stop()

    df = add_reward_components(df, weights=DENSE)
    df = add_tactical_flags(df)

    n_games = int(df["game_id"].nunique())
    status.success(f"Step 1/3 done — {n_games:,} games, {len(df):,} decisions parsed.")

    # ── Step 2: generate all plots ────────────────────────────────────────────
    status.info("Step 2/3 — Generating 10 behavioral plots…")
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        n_beh = generate_behavior_plots(df, zout)
        status.info("Step 3/3 — Generating 28 ratio histograms…")
        n_rat = generate_ratio_plots(df, zout)

    total = n_beh + n_rat
    status.success(f"Done — {n_beh} behavioral + {n_rat} ratio plots generated ({total} total).")

    # ── Step 3: download button ───────────────────────────────────────────────
    st.download_button(
        label=f"⬇️  Download all {total} plots (ZIP)",
        data=out_buf.getvalue(),
        file_name="tcg_plots.zip",
        mime="application/zip",
    )

    st.markdown(
        "**ZIP layout:**\n"
        "```\n"
        "tcg_plots.zip\n"
        "├── behavior/\n"
        "│   ├── A1_could_attack_outcome.png\n"
        "│   ├── A2_attack_rate_by_turn.png\n"
        "│   └── … (10 total)\n"
        "└── ratio/\n"
        "    ├── all/PLAY.png … RETREAT.png\n"
        "    ├── first_half/…\n"
        "    ├── third_quarter/…\n"
        "    └── last_quarter/… (28 total)\n"
        "```"
    )

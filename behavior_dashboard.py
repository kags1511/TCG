"""Behavioral dashboard -> W&B and/or PNG files.

Built from the flat decision table (all_games.parquet). Headline view is the
PASSIVITY suite (section A): of the turns where an ATTACK was actually offered,
how often did the player NOT attack -- and how often did it have the attack and
just end the turn ("could attack but directly ended").

Covered here (table-derived):
  A passivity/aggression  B energy-attaches/turn  C evolutions/bench
  D game length           E option-count + choice entropy
  F retreats              H first-player win rate

Deferred (need a raw-log re-parse): win-reason/deck-out, mulligan, coin %,
whiff rate, supporters, prize-timing, status-stuck.

    python behavior_dashboard.py --wandb
    python behavior_dashboard.py --save-plots          # PNG files only
    python behavior_dashboard.py --wandb --save-plots  # both
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENTITY, PROJECT = "arihan-saroj2006-iitm-india", "pokemon-tcg"
DEV_ACTIONS = ["PLAY", "ATTACH", "ATTACK", "EVOLVE"]
PLOT_DIR = "outputs/behavior_plots"


# ── matplotlib helpers ────────────────────────────────────────────────────────

def _savefig(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def mpl_hist(series, name, title, xlabel, ylabel, clip_q=0.99, outdir=PLOT_DIR):
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
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    return _savefig(fig, name, outdir)


def mpl_bar(labels_values, name, title, xlabel, ylabel, outdir=PLOT_DIR):
    labels = [str(k) for k, _ in labels_values]
    values = [float(v) for _, v in labels_values]
    fig, ax = plt.subplots(figsize=(9, 4))
    x_pos = range(len(labels))
    bars = ax.bar(x_pos, values, color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(
        [l if len(l) <= 22 else l[:19] + "…" for l in labels],
        fontsize=9
    )
    fig.tight_layout()
    return _savefig(fig, name, outdir)


def mpl_line(x, y, name, title, xlabel, ylabel, outdir=PLOT_DIR):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color="#4C72B0", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    fig.tight_layout()
    return _savefig(fig, name, outdir)


# ── W&B helpers (unchanged) ───────────────────────────────────────────────────

def wb_hist(run, series, name, title, clip_q=0.99):
    import wandb
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return
    hi = float(np.quantile(s, clip_q))
    s = s.clip(upper=hi if hi > 0 else s.max())
    tbl = wandb.Table(dataframe=pd.DataFrame({name: s}))
    run.log({f"hist/{name}": wandb.plot.histogram(tbl, name, title=title)})


def wb_bar(run, labels_values, key, title):
    import wandb
    tbl = wandb.Table(data=[[str(k), float(v)] for k, v in labels_values],
                      columns=["label", "count"])
    run.log({key: wandb.plot.bar(tbl, "label", "count", title=title)})


def wb_line(run, x, y, key, xname, yname, title):
    import wandb
    tbl = wandb.Table(data=[[float(a), float(b)] for a, b in zip(x, y)],
                      columns=[xname, yname])
    run.log({key: wandb.plot.line(tbl, xname, yname, title=title)})


# ── core ──────────────────────────────────────────────────────────────────────

def turn_table(df):
    """Collapse decisions into one row per (game, turn, player)."""
    df = df.copy()
    df["is_attack"] = df["option_type"] == "ATTACK"
    df["is_attach"] = df["option_type"] == "ATTACH"
    df["is_evolve"] = df["option_type"] == "EVOLVE"
    df["is_dev"] = df["option_type"].isin(DEV_ACTIONS)
    df["is_main"] = df["decision_type"] == "MAIN"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="all_games.parquet")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--save-plots", action="store_true",
                    help="save all charts as PNG files to behavior_plots/")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    t = turn_table(df)

    n_games = int(df["game_id"].nunique())

    # ---------- A. passivity / aggression ----------
    owned = t[t["owned"] == 1]
    avail = t[t["attack_available"] == 1]
    passed_rate = float((avail["attacked"] == 0).mean())
    directly_ended_rate = float(avail["directly_ended_with_attack"].mean())
    attacks_per_player_game = t.groupby(["game_id", "acting_player"])["attacked"].sum()
    pure_pass_rate = float((~owned["any_dev"]).mean())
    by_turn = avail.groupby("turn")["attacked"].mean()
    by_turn = by_turn[by_turn.index <= 60]

    # ---------- B. energy (owned turns only) ----------
    attach_per_turn = owned["n_attach"]

    # ---------- C. development ----------
    evolutions_per_game = t.groupby("game_id")["evolved"].sum()
    first_evo_turn = t[t["evolved"] == 1].groupby("game_id")["turn"].min()
    end_bench = t.sort_values("turn").groupby(["game_id", "acting_player"]).tail(1)["bench_count"]

    # ---------- D. game flow ----------
    game_len = df.groupby("game_id")["turn"].max()

    # ---------- E. decisions ----------
    multi = df[df["n_options"] > 1]
    vc = multi["option_type"].value_counts(normalize=True)
    choice_entropy = float(-(vc * np.log2(vc)).sum())

    # ---------- F / H ----------
    retreats_per_game = df[df["option_type"] == "RETREAT"].groupby("game_id").size()
    g0 = df[df["acting_player"] == 0].groupby("game_id")["outcome"].first()
    p0_win_rate = float((g0 > 0).mean())

    n_attacked      = int((avail["attacked"] == 1).sum())
    n_directly_ended = int(avail["directly_ended_with_attack"].sum())
    n_passed        = int((avail["attacked"] == 0).sum()) - n_directly_ended

    print(f"=== PASSIVITY (section A) — {n_games:,} games ===")
    print(f"attack-available turns      : {len(avail):,}")
    print(f"  ...passed (didn't attack) : {passed_rate:.1%}")
    print(f"  ...had attack & just ended: {directly_ended_rate:.1%}")
    print(f"pure-pass turns (no action) : {pure_pass_rate:.1%}")
    print(f"median attacks / player-game: {attacks_per_player_game.median():.0f}")
    print(f"attach actions / turn (mean): {attach_per_turn.mean():.2f}  (target ~1)")
    print(f"choice entropy (multi-opt)  : {choice_entropy:.2f} bits")
    print(f"P0 win rate                 : {p0_win_rate:.1%}")

    # ── PNG output ─────────────────────────────────────────────────────────────
    saved = []
    if args.save_plots:
        od = PLOT_DIR

        # A1 — headline bar: could-attack outcome
        saved.append(mpl_bar(
            [("Attacked", n_attacked),
             ("Passed\n(did other stuff)", n_passed),
             ("Directly ended\n(had attack, did nothing)", n_directly_ended)],
            "A1_could_attack_outcome",
            f"When ATTACK was available, what happened?  ({n_games:,} games)",
            xlabel="Outcome", ylabel="Number of turns", outdir=od,
        ))

        # A2 — attack rate by turn (line)
        saved.append(mpl_line(
            by_turn.index.tolist(), by_turn.values.tolist(),
            "A2_attack_rate_by_turn",
            "Attack rate by turn (turns where attack was available)",
            xlabel="Engine turn index", ylabel="Attack rate (0–1)", outdir=od,
        ))

        # A3 — attacks per player-game
        saved.append(mpl_hist(
            attacks_per_player_game, "A3_attacks_per_player_game",
            "Attacks per player per game",
            xlabel="Number of attacks", ylabel="Number of player-games",
            clip_q=1.0, outdir=od,
        ))

        # B1 — attach actions per owned turn
        saved.append(mpl_hist(
            attach_per_turn, "B1_attach_per_owned_turn",
            "Energy-attach actions per owned turn  (target ~1)",
            xlabel="Attach actions in turn", ylabel="Number of turns",
            clip_q=1.0, outdir=od,
        ))

        # C1 — evolutions per game
        saved.append(mpl_hist(
            evolutions_per_game, "C1_evolutions_per_game",
            "Evolutions per game",
            xlabel="Evolutions", ylabel="Number of games",
            clip_q=1.0, outdir=od,
        ))

        # C2 — first evolution turn
        saved.append(mpl_hist(
            first_evo_turn, "C2_first_evolution_turn",
            "Turn of first evolution",
            xlabel="Turn index", ylabel="Number of games",
            outdir=od,
        ))

        # C3 — end bench size
        saved.append(mpl_hist(
            end_bench, "C3_end_bench_size",
            "Bench size at game end",
            xlabel="Bench Pokémon count", ylabel="Number of player-games",
            clip_q=1.0, outdir=od,
        ))

        # D1 — game length
        saved.append(mpl_hist(
            game_len, "D1_game_length_turns",
            "Game length distribution",
            xlabel="Turns", ylabel="Number of games",
            outdir=od,
        ))

        # E1 — legal option count
        saved.append(mpl_hist(
            df["n_options"], "E1_legal_option_count",
            "Legal options per decision",
            xlabel="Number of options offered", ylabel="Number of decisions",
            outdir=od,
        ))

        # F1 — retreats per game
        saved.append(mpl_hist(
            retreats_per_game, "F1_retreats_per_game",
            "Retreats per game",
            xlabel="Retreats", ylabel="Number of games",
            clip_q=1.0, outdir=od,
        ))

        saved = [p for p in saved if p]
        print(f"\nsaved {len(saved)} PNGs -> {od}/")
        for p in saved:
            print(f"  {p}")

    # ── W&B output ─────────────────────────────────────────────────────────────
    if not args.wandb:
        return

    import wandb
    run = wandb.init(entity=ENTITY, project=PROJECT, name="behavior-dashboard",
                     config={"games": n_games, "decisions": int(len(df))})

    run.log({
        "passivity/attack_available_turns": int(len(avail)),
        "passivity/passed_with_attack_rate": passed_rate,
        "passivity/directly_ended_with_attack_rate": directly_ended_rate,
        "passivity/pure_pass_rate": pure_pass_rate,
        "energy/attach_per_turn_mean": float(attach_per_turn.mean()),
        "decisions/choice_entropy_bits": choice_entropy,
        "sanity/p0_win_rate": p0_win_rate,
        "dev/evolutions_per_game_mean": float(evolutions_per_game.mean()),
    })

    wb_bar(run, [("attacked", n_attacked),
                 ("passed (did other stuff, no attack)", n_passed),
                 ("directly ended (had attack, just passed)", n_directly_ended)],
           "passivity/could_attack_outcome",
           "When ATTACK was available, what happened?")

    wb_hist(run, attacks_per_player_game, "attacks_per_player_game",
            "Attacks per player per game", clip_q=1.0)
    wb_line(run, by_turn.index.tolist(), by_turn.values.tolist(),
            "passivity/attack_rate_by_turn", "turn", "attack_rate",
            "Attack rate by turn (where attack was available)")
    wb_hist(run, attach_per_turn, "attach_per_turn",
            "Energy-attach actions per turn", clip_q=1.0)
    wb_hist(run, evolutions_per_game, "evolutions_per_game",
            "Evolutions per game", clip_q=1.0)
    wb_hist(run, first_evo_turn, "first_evolution_turn",
            "Turn of first evolution")
    wb_hist(run, end_bench, "end_bench_size",
            "Bench size at game end", clip_q=1.0)
    wb_hist(run, game_len, "game_length_turns", "Game length (turns)")
    wb_hist(run, df["n_options"], "legal_option_count",
            "Legal options per decision")
    wb_hist(run, retreats_per_game, "retreats_per_game",
            "Retreats per game", clip_q=1.0)

    run.finish()
    print("\ndone -- open the behavior-dashboard run on wandb.ai")


if __name__ == "__main__":
    main()

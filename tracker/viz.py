"""Stage 3: stream the tracked table to Weights & Biases.

W&B is not where code runs -- it is a website you push numbers to. The three
log calls map to the three views in CONTEXT.md section 6:
  - reward curve  (WHEN)  -> per-game scalar log()
  - action histogram (WHAT) -> bar chart of chosen option types
  - raw rows               -> wandb.Table for drill-down

wandb is imported lazily so the rest of the tracker works without it installed.
"""

ENTITY = "arihan-saroj2006-iitm-india"  # from the W&B quickstart
PROJECT = "pokemon-tcg"


def init_run(entity=ENTITY, project=PROJECT, name=None, config=None):
    import wandb
    return wandb.init(entity=entity, project=project, name=name, config=config or {})


def _action_bar(run, game_df, title="Action distribution"):
    import wandb
    counts = game_df["option_type"].dropna().value_counts()
    table = wandb.Table(
        data=[[str(k), int(v)] for k, v in counts.items()],
        columns=["action", "count"],
    )
    run.log({"action_dist": wandb.plot.bar(table, "action", "count", title=title)})


def log_game(run, game_df):
    """Log one game's summary: reward curve point + action histogram."""
    # "final" = the last CHRONOLOGICAL decision (the table may be sorted by
    # player elsewhere), so prize_diff_final / win reflect the actual endgame.
    final = game_df.sort_values("step_index").iloc[-1]
    run.log({
        "reward_total_mean": float(game_df["reward_total"].mean()),
        "prize_diff_final": float(final["prize_diff"]),
        "hp_ratio_mean": float(game_df["hp_ratio"].mean()),
        "damage_dealt_total": float(game_df["damage_dealt"].sum()),
        "win": int(final["outcome"]),
        "turns": int(game_df["turn"].max() or 0),
    })
    _action_bar(run, game_df)


def log_retreat_quality(run, df):
    """Third view: how tactically sound the bot's retreats are."""
    import wandb
    from .tactics import add_tactical_flags, retreat_quality_summary

    if "retreat_judgeable" not in df.columns:
        df = add_tactical_flags(df)
    s = retreat_quality_summary(df)
    run.log({f"retreat/{k}": v for k, v in s.items()})

    j = df[df["retreat_judgeable"] == True]  # noqa: E712
    if len(j):
        good = int(j["retreat_good"].sum())
        table = wandb.Table(
            data=[["good", good], ["bad", len(j) - good]],
            columns=["retreat", "count"],
        )
        run.log({"retreat_quality": wandb.plot.bar(
            table, "retreat", "count", title="Retreat quality (good vs bad)")})
    return df


def log_overview(run, df):
    """Aggregate dashboard for a BATCH of games: one per-game summary table +
    headline scalars + a histogram of per-game reward. One handful of log()
    calls instead of one per game."""
    import pandas as pd
    import wandb
    from .tactics import add_tactical_flags, retreat_quality_summary

    if "retreat_judgeable" not in df.columns:
        df = add_tactical_flags(df)

    rows = []
    for gid, g in df.groupby("game_id"):
        final = g.sort_values("step_index").iloc[-1]
        g0 = g[g["acting_player"] == 0]
        p0_win = int(g0["outcome"].iloc[0]) if len(g0) else 0
        rq = retreat_quality_summary(g)
        rows.append({
            "game_id": gid,
            "turns": int(g["turn"].max() or 0),
            "n_decisions": int(len(g)),
            "reward_mean": float(g["reward_total"].mean()),
            "prize_diff_final": int(final["prize_diff"]),
            "p0_win": p0_win,
            "retreats": int(rq["retreats_total"]),
            "good_retreat_rate": rq["good_retreat_rate"],
        })
    summ = pd.DataFrame(rows)

    run.log({"per_game": wandb.Table(dataframe=summ)})
    run.log({"reward_hist": wandb.plot.histogram(
        wandb.Table(dataframe=summ[["reward_mean"]]), "reward_mean",
        title="Per-game mean reward")})
    run.log({
        "overview/games": int(len(summ)),
        "overview/decisions": int(len(df)),
        "overview/p0_win_rate": float((summ["p0_win"] > 0).mean()),
        "overview/mean_game_reward": float(summ["reward_mean"].mean()),
        "overview/mean_turns": float(summ["turns"].mean()),
        "overview/median_decisions_per_game": float(summ["n_decisions"].median()),
    })
    return summ


def log_dataset(run, df, max_rows=10000):
    """Push the raw flat table to W&B for drill-down (capped for upload size)."""
    import wandb
    sample = df if len(df) <= max_rows else df.sample(max_rows, random_state=0)
    run.log({"rows": wandb.Table(dataframe=sample)})


def stream(df, entity=ENTITY, project=PROJECT, name=None, per_game_limit=60):
    """Feed the whole tracked table, get a full dashboard.

    Few games  -> one reward-curve point per game (a real curve).
    Many games  -> an aggregate overview table + headline scalars instead, so we
    don't fire thousands of slow per-game log() calls.
    All cases get the action histogram and the retreat-quality view.
    """
    from .tactics import add_tactical_flags
    if "retreat_judgeable" not in df.columns:
        df = add_tactical_flags(df)

    n_games = df["game_id"].nunique()
    run = init_run(entity=entity, project=project, name=name,
                   config={"games": int(n_games), "decisions": int(len(df))})

    if n_games <= per_game_limit:
        for game_id, game_df in df.groupby("game_id"):
            log_game(run, game_df)                   # view 2: per-game reward curve
    else:
        log_overview(run, df)                        # view 2 (batch): aggregate

    _action_bar(run, df, title="Action distribution (all games)")  # view 1: histogram
    log_retreat_quality(run, df)                     # view 3: retreat quality
    log_dataset(run, df)
    run.finish()
    return run

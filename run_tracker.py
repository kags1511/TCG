"""CLI: feed it logs, it tracks everything, optionally streams to W&B.

Examples
--------
# Stage 1+2 on the synthetic fixture (no real replays needed yet):
    python run_tracker.py --demo

# Real replays once you have them (a file, folder, or glob):
    python run_tracker.py path/to/replays/ --out table.parquet
    python run_tracker.py "replays/*.json.gz" --out table.parquet

# Also stream the tracked table to the W&B dashboard:
    python run_tracker.py path/to/replays/ --wandb

# Use the late-game sparse reward profile:
    python run_tracker.py path/to/replays/ --sparse
"""

import argparse

import pandas as pd

from tracker.parse import parse_many
from tracker.reward import add_reward_components, DENSE, SPARSE
from tracker.tactics import add_tactical_flags, retreat_quality_summary


def summarize(df):
    print("=" * 70)
    print(f"rows: {len(df)}   games: {df['game_id'].nunique()}")
    print("-" * 70)
    cols = ["game_id", "turn", "acting_player", "decision_type", "option_type",
            "prize_diff", "hp_ratio", "damage_dealt", "reward_total", "outcome"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("HEAD:")
        print(df[cols].head(6).to_string(index=False))
        print("\nTAIL:")
        print(df[cols].tail(6).to_string(index=False))
        print("\nAction histogram (option_type):")
        print(df["option_type"].value_counts().to_string())
        print("\nPer-game reward_total mean:")
        print(df.groupby("game_id")["reward_total"].mean().to_string())
        print("\nRetreat quality (view 3):")
        for k, v in retreat_quality_summary(df).items():
            print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser(description="Pokemon TCG replay tracker")
    ap.add_argument("path", nargs="?", help="replay file, folder, or glob")
    ap.add_argument("--demo", action="store_true",
                    help="build + parse the synthetic fixture")
    ap.add_argument("--in-parquet",
                    help="load an already-built table instead of parsing replays")
    ap.add_argument("--sparse", action="store_true",
                    help="use the SPARSE late-game reward profile")
    ap.add_argument("--out", help="write the tracked table to this .parquet")
    ap.add_argument("--wandb", action="store_true",
                    help="stream the tracked table to W&B")
    args = ap.parse_args()

    if args.in_parquet:
        # Table already built (and reward/tactics columns present) -> just load.
        df = pd.read_parquet(args.in_parquet)
        print(f"[loaded] {args.in_parquet}: {len(df)} rows, "
              f"{df['game_id'].nunique()} games")
        if args.wandb:
            from tracker.viz import stream
            print("streaming to W&B ...")
            stream(df)
            print("done -- open the run on wandb.ai")
        else:
            summarize(df)
        return

    if args.demo:
        from tracker.make_fixture import write_fixture
        path = write_fixture()
        print(f"[demo] wrote fixture: {path}")
    elif args.path:
        path = args.path
    else:
        ap.error("give a replay path, --in-parquet, or use --demo")

    # Stage 1: logs -> flat table
    df = parse_many(path)
    if df.empty:
        print("no rows parsed -- check the path / replay format")
        return

    # Stage 2: reward components
    weights = SPARSE if args.sparse else DENSE
    df = add_reward_components(df, weights=weights)

    # Third view: tactical (retreat) quality flags
    df = add_tactical_flags(df)

    summarize(df)

    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")

    # Stage 3: W&B
    if args.wandb:
        from tracker.viz import stream
        print("\nstreaming to W&B ...")
        stream(df)
        print("done -- open the run on wandb.ai")


if __name__ == "__main__":
    main()

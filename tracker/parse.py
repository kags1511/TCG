"""Stage 1: raw replay -> flat row table.

A replay is {info, rewards, steps}. Each step is a list of 2 agent-views;
exactly one is ACTIVE (non-null observation + the action it took). We emit one
row per ACTIVE view. See CONTEXT.md sections 4-5.
"""

import gzip
import json
import os
import glob as _glob

import pandas as pd

from .schema import SELECT_TYPE, OPT_TYPE, SEL_CONTEXT, STRAT_COLS, COLUMNS


def load_replay(path):
    """Read a replay from .json or .json.gz."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---- helpers ----
def _sum_hp(mons):
    return sum((m.get("hp") or 0) for m in mons if m)


def _sum_energy(mons):
    # energies[] holds the attached energy of a Pokemon (CONTEXT.md section 4).
    return sum(len(m.get("energies") or []) for m in mons if m)


def summarize_player(p):
    """Collapse a PlayerState into the scalar features we track. Called twice
    per row (me and opponent) so the extraction logic lives in one place."""
    active = p.get("active") or []
    bench = p.get("bench") or []
    allmons = active + bench
    return {
        "active_hp": _sum_hp(active),
        "total_hp": _sum_hp(allmons),
        "bench_count": sum(1 for m in bench if m),
        # own hand is a list; opponent's is exposed only as a count.
        "hand_count": p.get("handCount", len(p.get("hand") or [])),
        "deck_count": p.get("deckCount", 0),
        # prize[] holds the remaining face-down prizes (None entries).
        "prizes_remaining": len(p.get("prize") or []),
        "energy_total": _sum_energy(allmons),
    }


def active_view(step):
    """Return (agent_index, view) for the ACTIVE agent in a step, or (None, None)."""
    for ai, view in enumerate(step):
        if view is not None and view.get("observation") is not None:
            return ai, view
    return None, None


def extract_row(step, game_id, step_index, final_rewards):
    """Turn one replay step into one flat row dict (or None to skip)."""
    ai, view = active_view(step)
    if view is None:
        return None

    obs = view["observation"]
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    action = view.get("action") or []

    players = cur.get("players") or []
    if len(players) < 2:
        return None  # malformed / terminal bookkeeping step

    me_idx = cur.get("yourIndex", ai)
    me = summarize_player(players[me_idx])
    opp = summarize_player(players[1 - me_idx])

    opts = sel.get("option") or []
    # OptionType of the chosen option = "what did the agent actually do".
    chosen_opt_type = None
    if action and isinstance(action[0], int) and 0 <= action[0] < len(opts):
        chosen_opt_type = opts[action[0]].get("type")
    # What was OFFERED this decision (13 ATTACK, 8 ATTACH, 12 RETREAT).
    opt_types = {o.get("type") for o in opts}

    prizes_me = me["prizes_remaining"]
    prizes_opp = opp["prizes_remaining"]

    # Outcome from the perspective of THIS row's acting player.
    rew = final_rewards[me_idx] if me_idx < len(final_rewards) else 0
    outcome = 1 if rew > 0 else (-1 if rew < 0 else 0)

    row = {
        "game_id": game_id,
        "step_index": step_index,
        "turn": cur.get("turn"),
        "acting_player": me_idx,
        "decision_type": SELECT_TYPE.get(sel.get("type"), sel.get("type")),
        "option_type": OPT_TYPE.get(chosen_opt_type, chosen_opt_type),
        "context": SEL_CONTEXT.get(sel.get("context"), sel.get("context")),
        "action_chosen": json.dumps(action),  # keep parquet-friendly
        "n_options": len(opts),
        "prizes_me": prizes_me,
        "prizes_opp": prizes_opp,
        # prize_diff: positive = the acting player is AHEAD (my prizes taken
        # minus opponent's). Sign flipped from the CONTEXT.md section 6 formula
        # so "more = winning" reads naturally on the dashboard.
        "prize_diff": (6 - prizes_me) - (6 - prizes_opp),
        "my_active_hp": me["active_hp"],
        "opp_active_hp": opp["active_hp"],
        "my_total_hp": me["total_hp"],
        "opp_total_hp": opp["total_hp"],
        "hp_ratio": me["total_hp"] / max(opp["total_hp"], 1),
        "bench_count": me["bench_count"],
        "opp_bench_count": opp["bench_count"],
        "hand_count": me["hand_count"],
        "opp_hand_count": opp["hand_count"],
        "my_deck_count": me["deck_count"],
        "opp_deck_count": opp["deck_count"],
        "energy_attached_this_turn": int(bool(cur.get("energyAttached"))),
        "my_energy_total": me["energy_total"],
        "opp_energy_total": opp["energy_total"],
        "energy_superiority": me["energy_total"] / max(opp["energy_total"], 1),
        "attack_available": int(13 in opt_types),
        "attach_available": int(8 in opt_types),
        "retreat_available": int(12 in opt_types),
        "outcome": outcome,
    }
    for c in STRAT_COLS:
        row[c] = 0.0  # strategy-detection module fills these later
    return row


def parse_replay(path):
    """One replay file -> DataFrame of decision rows."""
    data = load_replay(path)
    game_id = os.path.basename(path).split(".")[0]
    rewards = data.get("rewards") or [0, 0]
    rows = []
    for si, step in enumerate(data.get("steps") or []):
        r = extract_row(step, game_id, si, rewards)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_many(path_or_glob):
    """A file, a directory, or a glob -> one concatenated DataFrame.

    Feed it the folder of downloaded leaderboard replays and it tracks them all.
    """
    if os.path.isdir(path_or_glob):
        paths = sorted(
            _glob.glob(os.path.join(path_or_glob, "*.json"))
            + _glob.glob(os.path.join(path_or_glob, "*.json.gz"))
        )
    elif any(ch in path_or_glob for ch in "*?["):
        paths = sorted(_glob.glob(path_or_glob))
    else:
        paths = [path_or_glob]

    frames = [parse_replay(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)

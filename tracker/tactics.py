"""Third view: tactical quality of retreats.

The action histogram only counts HOW OFTEN the bot retreats; it can't say
whether a given retreat was smart. A retreat is judged by what happens over the
next few of that player's own decisions:

  dodged_ko       opponent took NO prize in the window  (you didn't get KO'd)
  saved_energy    you had energy invested and kept it    (CONTEXT s7 #13:
                  "never die with energy attached")
  reward_up       summed reward over the window was positive

A retreat is GOOD if it dodged a KO and EITHER saved the energy investment or
led to positive reward. Retreats too close to the end of a player's sequence to
look `window` decisions ahead are left unjudged (NaN), not counted as bad.
"""

import numpy as np
import pandas as pd

WINDOW = 3        # how many of the player's own later decisions to look ahead
ENERGY_FLOOR = 1  # a retreat only "saves energy" if there was energy to save


def add_tactical_flags(df, window=WINDOW, energy_floor=ENERGY_FLOOR):
    """Annotate each RETREAT row with forward-looking quality flags."""
    df = df.sort_values(["game_id", "acting_player", "step_index"]).reset_index(drop=True)
    g = df.groupby(["game_id", "acting_player"], sort=False)

    # Future board state = value `window` of this player's OWN decisions later.
    fut_prizes_opp = g["prizes_opp"].shift(-window)
    fut_my_energy = g["my_energy_total"].shift(-window)

    # Reward summed over the next `window` decisions (NaN only if none exist).
    shifts = pd.concat([g["reward_total"].shift(-k) for k in range(1, window + 1)], axis=1)
    fut_reward = shifts.sum(axis=1, min_count=1)

    is_retreat = df["option_type"] == "RETREAT"
    judgeable = is_retreat & fut_prizes_opp.notna()

    dodged_ko = fut_prizes_opp >= df["prizes_opp"]                 # opp took no prize
    saved_energy = (df["my_energy_total"] >= energy_floor) & (fut_my_energy >= df["my_energy_total"])
    reward_up = fut_reward > 0
    good = dodged_ko & (saved_energy | reward_up)

    df["is_retreat"] = is_retreat
    df["retreat_judgeable"] = judgeable
    # NaN where not judgeable so they're ignored by .mean().
    df["retreat_dodged_ko"] = np.where(judgeable, dodged_ko, np.nan)
    df["retreat_saved_energy"] = np.where(judgeable, saved_energy, np.nan)
    df["retreat_reward_up"] = np.where(judgeable, reward_up, np.nan)
    df["retreat_good"] = np.where(judgeable, good, np.nan)
    return df


def retreat_quality_summary(df):
    """Aggregate the flags into dashboard scalars."""
    if "retreat_judgeable" not in df.columns:
        df = add_tactical_flags(df)
    j = df[df["retreat_judgeable"] == True]  # noqa: E712
    if len(j) == 0:
        return {
            "retreats_total": int(df.get("is_retreat", pd.Series(dtype=bool)).sum()),
            "retreats_judged": 0,
            "good_retreat_rate": float("nan"),
            "dodged_ko_rate": float("nan"),
            "saved_energy_rate": float("nan"),
            "reward_up_rate": float("nan"),
        }
    return {
        "retreats_total": int(df["is_retreat"].sum()),
        "retreats_judged": int(len(j)),
        "good_retreat_rate": float(j["retreat_good"].mean()),
        "dodged_ko_rate": float(j["retreat_dodged_ko"].mean()),
        "saved_energy_rate": float(j["retreat_saved_energy"].mean()),
        "reward_up_rate": float(j["retreat_reward_up"].mean()),
    }

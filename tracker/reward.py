"""Stage 2: decomposed reward components on top of the flat table.

These are also diagnostic "sensors" (CONTEXT.md section 6) -- that is why they
stay decomposed instead of collapsing to one scalar. DENSE early (reward small
good actions so losing games still teach), SPARSE later (mostly win/loss +
prize efficiency) so the agent stops farming proxies. Switching profile is a
config change, not a rewrite.
"""

# Component -> weight. The reward_total is the weighted sum.
DENSE = {
    "prize_advantage": 1.0,
    "hp_advantage": 0.5,
    "energy_advantage": 0.3,
    "damage_dealt": 0.4,
    "prize_gained": 1.0,
    "active_blocker": 0.2,
    "win": 2.0,
}

SPARSE = {
    "prize_advantage": 0.3,
    "hp_advantage": 0.0,
    "energy_advantage": 0.0,
    "damage_dealt": 0.0,
    "prize_gained": 0.5,
    "active_blocker": 0.0,
    "win": 3.0,
}

DAMAGE_NORM = 300.0  # ~one big attack, used to scale damage_dealt into ~[0,1]


def add_reward_components(df, weights=DENSE):
    """Add normalized reward-component columns + reward_total to the table."""
    df = df.copy()
    if df.empty:
        for k in DENSE:
            df[k] = []
        df["reward_total"] = []
        return df

    # --- snapshot advantages (no time axis) ---
    df["prize_advantage"] = df["prize_diff"] / 6.0
    tot_hp = (df["my_total_hp"] + df["opp_total_hp"]).replace(0, 1)
    df["hp_advantage"] = df["my_total_hp"] / tot_hp
    tot_e = (df["my_energy_total"] + df["opp_energy_total"]).replace(0, 1)
    df["energy_advantage"] = df["my_energy_total"] / tot_e
    # Stub until EN_Card_Data.csv lets us identify wall/lock cards by name.
    df["active_blocker"] = 0.0
    df["win"] = df["outcome"]

    # --- temporal deltas, kept per (game, acting_player) so "me" stays the
    # same perspective from one of that player's decisions to the next ---
    df = df.sort_values(["game_id", "step_index"]).reset_index(drop=True)
    grp = df.groupby(["game_id", "acting_player"], sort=False)

    prizes_taken_me = 6 - df["prizes_me"]
    df["prize_gained"] = (
        prizes_taken_me.groupby([df["game_id"], df["acting_player"]])
        .diff().fillna(0).clip(lower=0)
    )
    opp_hp_delta = grp["opp_total_hp"].diff().fillna(0)
    df["damage_dealt"] = (-opp_hp_delta).clip(lower=0) / DAMAGE_NORM

    # --- weighted sum ---
    df["reward_total"] = sum(weights.get(k, 0.0) * df[k] for k in weights)
    return df

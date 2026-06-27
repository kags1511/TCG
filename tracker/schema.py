"""Single source of truth for column names and the cabt enum maps.

The enum maps mirror decode_log.py / CONTEXT.md section 4 so the tracker is
self-contained. The flat-row schema (COLUMNS) is defined here and nowhere else.
"""

# ---- cabt enum maps (CONTEXT.md section 4) ----
SELECT_TYPE = {
    0: "MAIN", 1: "CARD", 2: "ATTACHED_CARD", 3: "CARD_OR_ATTACHED",
    4: "ENERGY", 5: "SKILL", 6: "ATTACK", 7: "EVOLVE", 8: "COUNT",
    9: "YES_NO", 10: "SPECIAL_COND",
}

OPT_TYPE = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD",
    6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
    12: "RETREAT", 13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_COND",
}

SEL_CONTEXT = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH", 4: "TO_ACTIVE",
    5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD", 9: "TO_DECK",
    10: "TO_DECK_BOTTOM", 11: "TO_PRIZE", 12: "NOT_MOVE", 13: "DMG_COUNTER",
    14: "DMG_COUNTER_ANY", 15: "DAMAGE", 16: "REMOVE_DMG", 17: "HEAL",
    18: "EVOLVES_FROM", 19: "EVOLVES_TO", 20: "DEVOLVE", 21: "ATTACH_FROM",
    22: "ATTACH_TO", 23: "DETACH_FROM", 24: "LOOK", 25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD", 28: "SWITCH_ENERGY_CARD",
    29: "DISCARD_C_OR_AC", 30: "DISCARD_ENERGY", 31: "TO_HAND_ENERGY",
    32: "TO_DECK_ENERGY", 33: "SWITCH_ENERGY", 34: "SKILL_ORDER", 35: "ATTACK",
    36: "DISABLE_ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT", 39: "DMG_COUNT",
    40: "REMOVE_DMG_COUNT", 41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE",
    44: "FIRST_EFFECT", 45: "MORE_DEVOLVE", 46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_COND", 48: "RECOVER_SPECIAL_COND",
}

# The 18 opponent-strategy probabilities (CONTEXT.md section 7).
# Filled with 0.0 for now; the strategy-detection module populates them later.
STRAT_NAMES = [
    "energy_acceleration", "hand_disruption", "prize_manipulation", "bench_spread",
    "status_amplification", "energy_denial", "damage_snipe", "underdog_setup",
    "damage_wall", "item_lock", "ability_lock", "endurance_healing",
    "active_trap", "pivot_strategy", "recharge_attacker", "hand_scaling_damage",
    "mill_deckout", "deck_thinning_shield",
]
STRAT_COLS = [f"strat_{name}" for name in STRAT_NAMES]

# ---- The flat-row schema. One row per agent decision. ----
IDENTITY_COLS = [
    "game_id", "step_index", "turn", "acting_player",
    "decision_type", "option_type", "context", "action_chosen", "n_options",
]
PRIZE_COLS = ["prizes_me", "prizes_opp", "prize_diff"]
HP_COLS = ["my_active_hp", "opp_active_hp", "my_total_hp", "opp_total_hp", "hp_ratio"]
BOARD_COLS = [
    "bench_count", "opp_bench_count", "hand_count", "opp_hand_count",
    "my_deck_count", "opp_deck_count",
]
ENERGY_COLS = [
    "energy_attached_this_turn", "my_energy_total", "opp_energy_total",
    "energy_superiority",
]
# What the engine OFFERED this decision (not just what was chosen) -- lets us
# ask "could it have attacked but didn't?".
MENU_COLS = ["attack_available", "attach_available", "retreat_available"]
LABEL_COLS = ["outcome"]

COLUMNS = (
    IDENTITY_COLS + PRIZE_COLS + HP_COLS + BOARD_COLS + ENERGY_COLS
    + MENU_COLS + STRAT_COLS + LABEL_COLS
)

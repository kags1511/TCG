"""Synthesize a raw replay in the documented cabt schema.

We don't have a raw game_XXXX.json.gz locally yet (only the decoded text log),
so this builds a schema-correct fake game to exercise the parser end to end.
Narrative: P0 grinds P1 down -- P0's prize pile counts 6 -> 0 while P1's
Pokemon lose HP -- so prize_diff, damage_dealt and the reward curve all move.

Swap this out for real Kaggle replays later; the parser is unchanged.
"""

import gzip
import json
import os


def _mon(card_id, hp, max_hp, n_energy):
    return {
        "id": card_id, "serial": card_id * 10, "hp": hp, "maxHp": max_hp,
        "appearThisTurn": False, "energies": list(range(n_energy)),
        "energyCards": [], "tools": [], "preEvolution": [],
    }


def _player(active_hp, bench_hps, prizes_remaining, hand_n, deck_n, active_energy):
    return {
        "active": [_mon(304, active_hp, 200, active_energy)],
        "bench": [_mon(800 + i, hp, 120, 1) for i, hp in enumerate(bench_hps)],
        "benchMax": 5,
        "hand": [{"id": 1} for _ in range(hand_n)],
        "handCount": hand_n,
        "deckCount": deck_n,
        "discard": [],
        "prize": [None] * prizes_remaining,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


# type ids: PLAY 7, ATTACH 8, EVOLVE 9, RETREAT 12, ATTACK 13, END 14
_MENU_TYPES = [7, 8, 9, 12, 13, 14]
_NAME2ID = {"PLAY": 7, "ATTACH": 8, "EVOLVE": 9, "RETREAT": 12, "ATTACK": 13, "END": 14}


def _main_select(chosen_name):
    # A MAIN menu containing the full move spread; the bot picks `chosen_name`.
    option = [{"type": t, "index": i} for i, t in enumerate(_MENU_TYPES)]
    chosen_idx = _MENU_TYPES.index(_NAME2ID[chosen_name])
    return {"type": 0, "context": 0, "minCount": 1, "maxCount": 1, "option": option}, [chosen_idx]


def build_replay():
    # Ground-truth timeline: (turn, acting, p0_state, p1_state, chosen_action).
    # P0 wins (prizes 6->0, P1 HP grinds down). Includes 3 judgeable retreats:
    # two GOOD P0 retreats (keep energy, opp takes no prize) and one BAD P1
    # retreat (loses the energy race while P0 keeps taking prizes).
    timeline = [
        (1, 0, _player(150, [120], 6, 5, 40, 3), _player(200, [110], 6, 5, 40, 2), "RETREAT"),
        (1, 1, _player(150, [120], 6, 5, 39, 3), _player(200, [110], 6, 5, 39, 2), "ATTACH"),
        (2, 0, _player(150, [120], 6, 5, 38, 3), _player(200, [110], 6, 5, 38, 2), "ATTACK"),
        (2, 1, _player(150, [120], 6, 5, 37, 3), _player(110, [110], 6, 5, 37, 2), "ATTACH"),
        (3, 0, _player(150, [120], 5, 5, 36, 3), _player(110, [110], 6, 5, 36, 3), "RETREAT"),
        (3, 1, _player(150, [120], 5, 5, 35, 3), _player(110, [110], 6, 4, 35, 1), "RETREAT"),
        (4, 0, _player(150, [120], 4, 5, 34, 3), _player(20, [110], 6, 4, 34, 3), "ATTACK"),
        (4, 1, _player(150, [120], 4, 5, 33, 3), _player(110, [60], 6, 4, 33, 1), "ATTACH"),
        (5, 0, _player(150, [120], 3, 5, 32, 3), _player(110, [60], 6, 4, 32, 3), "ATTACK"),
        (5, 1, _player(140, [120], 3, 5, 31, 3), _player(40, [60], 6, 3, 31, 1), "ATTACH"),
        (6, 0, _player(150, [120], 2, 5, 30, 3), _player(40, [60], 6, 3, 30, 3), "ATTACK"),
        (6, 1, _player(150, [120], 1, 5, 29, 3), _player(110, [20], 6, 3, 29, 1), "ATTACH"),
        (7, 0, _player(150, [120], 0, 5, 28, 3), _player(0, [20], 6, 3, 28, 3), "ATTACK"),
    ]

    steps = []
    for i, (turn, acting, p0, p1, chosen) in enumerate(timeline):
        players = [p0, p1]
        sel, action = _main_select(chosen)
        result = 0 if i == len(timeline) - 1 else -1  # last step: P0 win
        obs = {
            "current": {
                "turn": turn, "yourIndex": acting, "result": result,
                "energyAttached": (i % 2 == 0), "players": players,
            },
            "select": sel,
            "logs": [],
        }
        active_view = {"observation": obs, "action": action,
                       "status": "ACTIVE", "reward": 0.0}
        inactive_view = {"observation": None, "action": None,
                         "status": "INACTIVE", "reward": 0.0}
        step = [active_view, inactive_view] if acting == 0 else [inactive_view, active_view]
        steps.append(step)

    return {
        "info": {"TeamNames": ["fixture-P0", "fixture-P1"]},
        "rewards": [1.0, -1.0],
        "steps": steps,
    }


def write_fixture(path="fixtures/game_fixture.json.gz"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    replay = build_replay()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(replay, f)
    return path


if __name__ == "__main__":
    p = write_fixture()
    print(f"wrote {p}")

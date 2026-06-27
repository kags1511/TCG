# hops_reinforce.py -- VANILLA REINFORCE baseline on the cabt self-play env.
#
# Reuses the friend's EXACT encoder + MyModel + cg environment (copied verbatim
# from hops_selfplay.py), but:
#   * replaces MCTS with direct sampling from the policy softmax, and
#   * replaces the AlphaZero policy+value loss with a REINFORCE policy gradient.
#
# Run on Kaggle the same way as the AlphaZero notebook -- attach the same inputs
# (cg-lib + rr_model_bc_hops.pth + hops_pilot_deck.json). Cannot run off-Kaggle
# (needs the cg/cabt library).
#
# Reward: win +1 / loss -1 + potential-based prize shaping whose weight (alpha)
# decays LINEARLY 0.3 -> 0 over the first PRIZE_ANNEAL_FRAC of training, then
# pure win/loss -- matching the friend's AlphaZero schedule. gamma = 1.0.
# Baseline = batch-mean return. Exploration = sampling the policy.

import os, sys, json, time, math, random, collections, glob, gzip
if os.environ.get("HOPS_FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import numpy as np
import multiprocessing, queue

for _p in (glob.glob('/kaggle/input/**/cg-lib', recursive=True)
           + glob.glob('/kaggle_simulations/**/cg-lib', recursive=True)
           + ["_cgtest"]
           + glob.glob('**/cg-lib', recursive=True)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
from cg.api import (
    AreaType, OptionType, SelectContext, all_attack, all_card_data,
    search_begin, search_end, search_step, to_observation_class,
)
from cg.game import battle_start, battle_select, battle_finish

def _find(filename, required=True):
    cands = ([filename]
             + glob.glob(f'/kaggle/input/**/{filename}', recursive=True)
             + glob.glob(f'/kaggle/working/**/{filename}', recursive=True)
             + glob.glob(f'**/{filename}', recursive=True))
    for c in cands:
        if os.path.exists(c): return c
    if required:
        raise FileNotFoundError(f"{filename} not found; searched {cands[:6]}")
    return None

try:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            raise RuntimeError(f"GPU sm_{major}{minor} not supported by this PyTorch (need sm_70+)")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
except Exception as _e:
    print(f"CUDA unavailable ({_e}), falling back to cpu")
    device = torch.device("cpu")
print("device:", device)


PILOT_DECK = [11, 11, 11, 11, 12, 19, 19, 19, 19, 65, 65, 65, 65, 66, 66, 66, 304, 304, 878, 878, 878, 878, 879, 879, 1086, 1086, 1086, 1086, 1097, 1097, 1097, 1115, 1115, 1115, 1122, 1122, 1122, 1122, 1152, 1152, 1152, 1152, 1171, 1171, 1171, 1171, 1182, 1182, 1194, 1194, 1210, 1210, 1227, 1227, 1227, 1227, 1255, 1255, 1255, 1255]
assert len(PILOT_DECK) == 60
POOL_DECKS = [PILOT_DECK]

all_card   = all_card_data()
card_table = {c.cardId: c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1
num_words_encoder    = 24
encoder_size         = 22000
decoder_main_feature = 8
decoder_attack_offset = 14
decoder_card_offset   = decoder_attack_offset + attack_count
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count

class SparseVector:
    def __init__(self):
        self.index = []; self.value = []; self.offset = []; self.pos = 0
    def add(self, index, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index); self.value.append(value)
    def add_pos(self, pos): self.pos += pos
    def add_single(self, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos); self.value.append(value)
        self.pos += 1
    def word_start(self): self.offset.append(len(self.index))

def add_card(sv, card):
    if card is not None: sv.add(card.id, 1)
    sv.add_pos(card_count)

def add_cards(sv, cards, value):
    if cards is not None:
        for card in cards: sv.add(card.id, value)
    sv.add_pos(card_count)

def add_pokemon(sv, poke):
    if poke is None:
        sv.add_single(1); sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0); sv.add_single(poke.hp / 400)
        add_card(sv, poke); add_cards(sv, poke.tools, 1.0); add_cards(sv, poke.energyCards, 0.5)

def add_player(sv, ps):
    sv.add_single(ps.deckCount / 60); sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8); sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1); sv.add_pos(7)
    sv.add_single(ps.poisoned); sv.add_single(ps.burned); sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed); sv.add_single(ps.confused)
    add_cards(sv, ps.discard, 0.25)

def get_encoder_input(obs, your_deck):
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start(); pos = sv.pos
            if j < len(ps.bench): add_pokemon(sv, ps.bench[j])
            else: add_pokemon(sv, None)
            if j != 7: sv.pos = pos
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active): add_pokemon(sv, ps.active[0])
        else: add_pokemon(sv, None)
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start(); add_player(sv, ps)
    sv.word_start(); add_cards(sv, state.players[your_index].hand, 0.25)
    sv.word_start()
    for id in your_deck: sv.add(id, 0.25)
    sv.add_pos(card_count)
    sv.word_start(); add_cards(sv, state.stadium, 1.0)
    sv.word_start(); sv.add_single(1); sv.add_single(state.turn / 10); sv.add_single(state.firstPlayer == your_index)
    return sv

def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK: return obs.select.deck[index]
        case AreaType.HAND: return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE: return ps.active[index]
        case AreaType.BENCH: return ps.bench[index]
        case AreaType.PRIZE: return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _: return None

def decoder_main(sv, feature_index, card):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)

def decoder_card_id(sv, context, card_id):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)

def decoder_card(sv, context, card):
    if card is not None: decoder_card_id(sv, context, card.id)

def get_decoder_input(obs, actions):
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1); continue
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END: sv.add(1, 1)
                case OptionType.YES: sv.add(2, 1)
                case OptionType.NO: sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION: sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER: sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK: sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY: decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY: decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD: decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT: decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD: decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL: decoder_card_id(sv, context, o.cardId)
    return sv


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
    def forward(self, x, encoder_out):
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = torch.nn.functional.relu(self.fc1(res))
        y = self.fc2(y)
        return self.norm2(res + y)

class MyModel(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers_encoder, num_layers_decoder):
        super().__init__()
        self.d_model = d_model
        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        enc = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(enc, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList(DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder))
        self.decoder_fc = torch.nn.Linear(d_model, 1)
    def forward(self, ie, ve, oe, idc, vdc, odc):
        v = self.encoder_bag(ie, oe, ve).reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch = v.size(1)
        enc = self.encoder(v)
        value = torch.tanh(self.encoder_fc(enc).mean(0))
        p = self.decoder_bag(idc, odc, vdc).reshape(batch, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, enc)
        p = torch.tanh(self.decoder_fc(p).transpose(0, 1).view(batch, -1))
        return value, p

def eval_nn(sv_enc, sv_dec, model):
    dev = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=dev),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=dev),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=dev),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=dev),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=dev),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=dev))
    return value.tolist()[0][0], policy.tolist()[0]



# ============================================================================
# REINFORCE config
# ============================================================================
os.makedirs("out", exist_ok=True)

LOG_GAMES = bool(int(os.environ.get("LOG_GAMES", "1")))
LOG_DIR   = os.environ.get("LOG_DIR", "logs_out")
if LOG_GAMES:
    os.makedirs(LOG_DIR, exist_ok=True)
_game_counter = 0

WIN_REWARD, LOSS_REWARD = 1.0, -1.0
GAMMA      = 1.0
N_PRIZES   = 6

# Prize-shaping decay — exact friend's formula: PRIZE_ALPHA0 -> 0 over the
# first PRIZE_ANNEAL_FRAC of wall-clock training time, then pure win/loss.
PRIZE_ALPHA0      = float(os.environ.get("PRIZE_ALPHA0", 0.3))   # full 6-prize swing; per prize = alpha/6
PRIZE_ANNEAL_FRAC = float(os.environ.get("PRIZE_ANNEAL_FRAC", 0.7))

LR               = float(os.environ.get("LR", 1e-4))
GRAD_CLIP        = 1.0
GAMES_PER_ROUND  = int(os.environ.get("GAMES_PER_ROUND", 40))
MAX_HOURS        = float(os.environ.get("MAX_HOURS", 8.0))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", 5))
EVAL_EVERY          = int(os.environ.get("EVAL_EVERY", 5))
EVAL_GAMES          = int(os.environ.get("EVAL_GAMES", 20))

BC_WEIGHTS = _find("rr_model_bc_hops.pth", required=False)


def prize_alpha_for(elapsed_secs):
    """Friend's exact formula: alpha decays 0.3 -> 0 over first 70% of wall-clock time."""
    frac = min(1.0, elapsed_secs / (MAX_HOURS * 3600))
    return PRIZE_ALPHA0 * max(0.0, 1.0 - frac / PRIZE_ANNEAL_FRAC)


# ============================================================================
# Action enumeration (extracted verbatim from create_node) + the REINFORCE agent
# ============================================================================
def enumerate_actions(obs):
    """All legal option-combinations (maxCount picks), up to 64 -- the same set
    the AlphaZero code scored."""
    actions = []
    indices = list(range(obs.select.maxCount))
    for _ in range(64):
        actions.append(indices.copy())
        for i in range(len(indices)):
            idx = len(indices) - i - 1
            if indices[idx] < len(obs.select.option) - i - 1:
                indices[idx] += 1
                for j in range(idx + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


def policy_scores(sv_enc, sv_dec, model):
    """One decision -> (per-action score tensor, value). Keeps grad."""
    dev = next(model.parameters()).device
    value, p = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=dev),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=dev),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=dev),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=dev),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=dev),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=dev))
    return p[0], value[0][0]


class RSample:
    __slots__ = ("sv_enc", "sv_dec", "action", "phi", "ret")
    def __init__(self, sv_enc, sv_dec, action):
        self.sv_enc = sv_enc; self.sv_dec = sv_dec; self.action = action
        self.phi = 0.0; self.ret = 0.0


def reinforce_agent(obs_dict, your_deck, model, greedy=False):
    """Pure policy: softmax over the option-combination scores, then SAMPLE
    (that sampling is the exploration). greedy=True -> argmax, for eval."""
    obs = to_observation_class(obs_dict)
    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    with torch.no_grad():
        scores, _ = policy_scores(sv_enc, sv_dec, model)
        probs = torch.softmax(scores, dim=0)
    a = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1))
    return actions[a], RSample(sv_enc, sv_dec, a)


def random_agent(obs_dict):
    obs = to_observation_class(obs_dict)
    n, k = len(obs.select.option), obs.select.maxCount
    return random.sample(range(n), min(k, n)) if n > 0 else []


# ============================================================================
# Self-play game -> REINFORCE returns (win/loss + decaying prize shaping, gamma=1)
# ============================================================================
def play_one_game(model, sample_list, prize_alpha=0.0, log=False):
    global _game_counter
    obs, start = battle_start(PILOT_DECK, PILOT_DECK)
    if start.errorPlayer >= 0:
        return None
    samples = [[], []]
    steps = [] if log else None
    while obs["current"]["result"] < 0:
        side = obs["current"]["yourIndex"]
        selected, s = reinforce_agent(obs, PILOT_DECK, model)
        if log:
            step = [None, None]
            step[side] = {"observation": obs, "action": selected}
            steps.append(step)
        my_rem  = len(obs["current"]["players"][side]["prize"])
        opp_rem = len(obs["current"]["players"][1 - side]["prize"])
        s.phi = prize_alpha * (opp_rem - my_rem) / N_PRIZES
        samples[side].append(s)
        obs = battle_select(selected)
    battle_finish()

    res = obs["current"]["result"]
    final_rem = [len(obs["current"]["players"][k]["prize"]) for k in range(2)]

    if log and steps:
        _game_counter += 1
        game_rewards = [0.0, 0.0]
        if res != 2:
            game_rewards[res] = WIN_REWARD
            game_rewards[1 - res] = LOSS_REWARD
        record = {"info": {"TeamNames": ["agent", "agent"]},
                  "rewards": game_rewards, "steps": steps}
        path = os.path.join(LOG_DIR, f"game_{_game_counter}.json.gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(record, f)

    for i in range(2):
        seq = samples[i]
        T = len(seq)
        if T == 0:
            continue
        terminal = 0.0 if res == 2 else (WIN_REWARD if i == res else LOSS_REWARD)
        phi_terminal = prize_alpha * (final_rem[1 - i] - final_rem[i]) / N_PRIZES
        rewards = []
        for t in range(T):
            phi_now = seq[t].phi
            phi_next = seq[t + 1].phi if t + 1 < T else phi_terminal
            rewards.append(phi_next - phi_now)
        rewards[-1] += terminal
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + GAMMA * G
            seq[t].ret = G
            sample_list.append(seq[t])
    return res


# ============================================================================
# REINFORCE update: loss = -(log pi(a) * advantage), advantage = G - mean(G)
# ============================================================================
def reinforce_update(model, optimizer, samples):
    if not samples:
        return {}
    returns = torch.tensor([s.ret for s in samples], dtype=torch.float32, device=device)
    adv = returns - returns.mean()
    if len(adv) > 1:
        adv = adv / (returns.std() + 1e-8)

    optimizer.zero_grad()
    n = len(samples)
    total_loss = 0.0
    ent_sum = 0.0
    for i, s in enumerate(samples):
        scores, _ = policy_scores(s.sv_enc, s.sv_dec, model)     # recompute WITH grad
        logp = torch.log_softmax(scores, dim=0)[s.action]
        loss = -(logp * adv[i]) / n                              # mean over the batch
        loss.backward()                                         # accumulate grads (memory-light)
        total_loss += float(loss.item())
        with torch.no_grad():
            p = torch.softmax(scores, dim=0)
            ent_sum += float(-(p * torch.log(p.clamp_min(1e-9))).sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return {"loss": total_loss, "entropy": ent_sum / n, "mean_return": float(returns.mean())}


# ============================================================================
# Eval: greedy REINFORCE policy vs the random agent (absolute strength yardstick)
# ============================================================================
def evaluate_vs_random(model, n_games=EVAL_GAMES):
    wins = losses = draws = 0
    with torch.inference_mode():
        for i in range(n_games):
            obs, start = battle_start(PILOT_DECK, PILOT_DECK)
            if start.errorPlayer >= 0:
                continue
            your_index = i % 2
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = reinforce_agent(obs, PILOT_DECK, model, greedy=True)
                else:
                    selected = random_agent(obs)
                obs = battle_select(selected)
            battle_finish()
            r = obs["current"]["result"]
            if r == 2: draws += 1
            elif r == your_index: wins += 1
            else: losses += 1
    total = wins + losses + draws
    return (wins / total if total else 0.0), wins, losses, draws


# ============================================================================
# Main loop
# ============================================================================
def main():
    model = MyModel(128, 2, 256, 1, 1).to(device)    # CONFIRMED dims
    if BC_WEIGHTS:
        model.load_state_dict(torch.load(BC_WEIGHTS, map_location=device))
        print("warm-started from BC:", BC_WEIGHTS)
    else:
        print("WARNING: rr_model_bc_hops.pth not found -- training from random init")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    # subtract 10-min buffer so the final save always completes before the hard limit
    deadline = start_time + MAX_HOURS * 3600 - 600
    rnd = 0
    while time.time() < deadline:
        elapsed = time.time() - start_time
        alpha = prize_alpha_for(elapsed)
        samples = []
        for _ in range(GAMES_PER_ROUND):
            play_one_game(model, samples, prize_alpha=alpha, log=LOG_GAMES)
        m = reinforce_update(model, optimizer, samples)
        elapsed_h = elapsed / 3600
        print(f"round {rnd:4d} | {elapsed_h:.2f}h | alpha {alpha:.4f} | samples {len(samples):5d} | "
              f"loss {m.get('loss',0):+.4f} | entropy {m.get('entropy',0):.3f} | "
              f"meanG {m.get('mean_return',0):+.3f}", flush=True)
        if (rnd + 1) % CHECKPOINT_EVERY == 0:
            path = f"out/rr_model_reinforce_{rnd+1}.pth"
            torch.save(model.state_dict(), path)
            print("  saved", path, flush=True)
        if (rnd + 1) % EVAL_EVERY == 0:
            wr, w, l, d = evaluate_vs_random(model)
            print(f"  eval vs random: win {wr:.1%}  ({w}-{l}-{d})", flush=True)
        rnd += 1

    torch.save(model.state_dict(), "out/rr_model_reinforce_final.pth")
    print("done. final: out/rr_model_reinforce_final.pth")


if __name__ == "__main__":
    main()

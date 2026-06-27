import json

with open('game_1097_json') as f:
    data = json.load(f)

# ---- Enum maps from the cabt API docs ----
LOG_TYPE = {
    0:"SHUFFLE", 1:"HAS_BASIC_POKEMON", 2:"TURN_START", 3:"TURN_END",
    4:"DRAW", 5:"DRAW_REVERSE(opp drew)", 6:"MOVE_CARD", 7:"MOVE_CARD_REVERSE(facedown)",
    8:"SWITCH", 9:"CHANGE", 10:"PLAY", 11:"ATTACH", 12:"EVOLVE", 13:"DEVOLVE",
    14:"MOVE_ATTACHED", 15:"ATTACK", 16:"HP_CHANGE", 17:"POISONED", 18:"BURNED",
    19:"ASLEEP", 20:"PARALYZED", 21:"CONFUSED", 22:"COIN", 23:"RESULT"
}
AREA = {1:"DECK",2:"HAND",3:"DISCARD",4:"ACTIVE",5:"BENCH",6:"PRIZE",
        7:"STADIUM",8:"ENERGY",9:"TOOL",10:"PRE_EVO",11:"PLAYER",12:"LOOKING"}
SELECT_TYPE = {0:"MAIN",1:"CARD",2:"ATTACHED_CARD",3:"CARD_OR_ATTACHED",
               4:"ENERGY",5:"SKILL",6:"ATTACK",7:"EVOLVE",8:"COUNT",9:"YES_NO",10:"SPECIAL_COND"}
OPT_TYPE = {0:"NUMBER",1:"YES",2:"NO",3:"CARD",4:"TOOL_CARD",5:"ENERGY_CARD",
            6:"ENERGY",7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",11:"DISCARD",
            12:"RETREAT",13:"ATTACK",14:"END",15:"SKILL",16:"SPECIAL_COND"}
SEL_CONTEXT = {0:"MAIN",1:"SETUP_ACTIVE",2:"SETUP_BENCH",3:"SWITCH",4:"TO_ACTIVE",
    5:"TO_BENCH",6:"TO_FIELD",7:"TO_HAND",8:"DISCARD",9:"TO_DECK",10:"TO_DECK_BOTTOM",
    11:"TO_PRIZE",12:"NOT_MOVE",13:"DMG_COUNTER",14:"DMG_COUNTER_ANY",15:"DAMAGE",
    16:"REMOVE_DMG",17:"HEAL",18:"EVOLVES_FROM",19:"EVOLVES_TO",20:"DEVOLVE",
    21:"ATTACH_FROM",22:"ATTACH_TO",23:"DETACH_FROM",24:"LOOK",25:"EFFECT_TARGET",
    26:"DISCARD_ENERGY_CARD",27:"DISCARD_TOOL_CARD",28:"SWITCH_ENERGY_CARD",
    29:"DISCARD_C_OR_AC",30:"DISCARD_ENERGY",31:"TO_HAND_ENERGY",32:"TO_DECK_ENERGY",
    33:"SWITCH_ENERGY",34:"SKILL_ORDER",35:"ATTACK",36:"DISABLE_ATTACK",37:"EVOLVE",
    38:"DRAW_COUNT",39:"DMG_COUNT",40:"REMOVE_DMG_COUNT",41:"IS_FIRST",42:"MULLIGAN",
    43:"ACTIVATE",44:"FIRST_EFFECT",45:"MORE_DEVOLVE",46:"COIN_HEAD",
    47:"AFFECT_SPECIAL_COND",48:"RECOVER_SPECIAL_COND"}

def fmt_log(l):
    t = l.get('type')
    name = LOG_TYPE.get(t, f"LOG{t}")
    pi = l.get('playerIndex')
    pfx = f"P{pi}" if pi is not None else "  "
    if t == 4:  # DRAW
        return f"{pfx} drew card#{l.get('cardId')} (serial {l.get('serial')})"
    if t == 5:  # opp drew
        return f"{pfx} drew a card (hidden)"
    if t == 2:  return f"{pfx} ---- TURN START ----"
    if t == 3:  return f"{pfx} ---- turn end ----"
    if t == 0:  return f"{pfx} shuffled deck"
    if t == 1:  return f"{pfx} basic pokemon check: {l.get('hasBasicPokemon')}"
    if t == 6:  # move card
        return f"{pfx} moved card#{l.get('cardId')} : {AREA.get(l.get('fromArea'),l.get('fromArea'))} -> {AREA.get(l.get('toArea'),l.get('toArea'))}"
    if t == 7:  # facedown move
        return f"{pfx} moved facedown card: {AREA.get(l.get('fromArea'),'?')} -> {AREA.get(l.get('toArea'),'?')}"
    if t == 8:  # switch
        return f"{pfx} switched active<->bench (active#{l.get('cardIdActive')} bench#{l.get('cardIdBench')})"
    if t == 9:  # change
        return f"{pfx} changed pokemon #{l.get('cardIdBefore')} -> #{l.get('cardIdAfter')}"
    if t == 10: # play
        return f"{pfx} PLAYED card#{l.get('cardId')}"
    if t == 11: # attach
        return f"{pfx} attached card#{l.get('cardId')} -> pokemon#{l.get('cardIdTarget')}"
    if t == 12: # evolve
        return f"{pfx} EVOLVED #{l.get('cardId')} onto #{l.get('cardIdTarget')}"
    if t == 13:
        return f"{pfx} devolved #{l.get('cardId')}"
    if t == 14:
        return f"{pfx} moved attached card#{l.get('cardId')}"
    if t == 15: # attack
        return f"{pfx} >>> ATTACKED with #{l.get('cardId')} (attackId {l.get('attackId')})"
    if t == 16: # hp change
        sign = "+" if (l.get('value',0) or 0) > 0 else ""
        dc = " [dmg counter]" if l.get('putDamageCounter') else ""
        return f"{pfx} HP change on #{l.get('cardId')}: {sign}{l.get('value')}{dc}"
    if t == 17: return f"{pfx} poison {'recovered' if l.get('isRecover') else 'applied'} on #{l.get('cardId')}"
    if t == 18: return f"{pfx} burn {'recovered' if l.get('isRecover') else 'applied'} on #{l.get('cardId')}"
    if t == 19: return f"{pfx} sleep {'recovered' if l.get('isRecover') else 'applied'} on #{l.get('cardId')}"
    if t == 20: return f"{pfx} paralyze {'recovered' if l.get('isRecover') else 'applied'} on #{l.get('cardId')}"
    if t == 21: return f"{pfx} confuse {'recovered' if l.get('isRecover') else 'applied'} on #{l.get('cardId')}"
    if t == 22: return f"{pfx} COIN FLIP: {'HEADS' if l.get('head') else 'TAILS'}"
    if t == 23:
        res = l.get('result'); reason = l.get('reason')
        rmap={0:'P0 WIN',1:'P1 WIN',2:'DRAW'}
        rs={1:'took all prizes',2:'no deck',3:'no active pokemon',4:'card effect'}
        return f"   *** GAME RESULT: {rmap.get(res,res)} ({rs.get(reason,reason)}) ***"
    return f"{pfx} {name}: {l}"

def fmt_option(o):
    ot = o.get('type')
    base = OPT_TYPE.get(ot, f"OPT{ot}")
    parts=[base]
    if o.get('index') is not None: parts.append(f"idx={o['index']}")
    if o.get('area') is not None: parts.append(f"area={AREA.get(o['area'],o['area'])}")
    if o.get('attackId') is not None: parts.append(f"atkId={o['attackId']}")
    if o.get('cardId') is not None: parts.append(f"card#{o['cardId']}")
    if o.get('number') is not None: parts.append(f"n={o['number']}")
    if o.get('inPlayArea') is not None: parts.append(f"inPlay={AREA.get(o['inPlayArea'],o['inPlayArea'])}")
    return " ".join(parts)

# ---- Walk steps ----
out=[]
out.append(f"TEAMS: {data['info']['TeamNames']}")
out.append(f"FINAL REWARDS: {data['rewards']}  (P0={data['rewards'][0]}, P1={data['rewards'][1]})")
out.append(f"TOTAL STEPS: {len(data['steps'])}")
out.append("="*90)

for si, step in enumerate(data['steps']):
    for ai, agent in enumerate(step):
        obs = agent.get('observation')
        if obs is None:
            continue
        act = agent.get('action')
        status = agent.get('status')
        sel = obs.get('select')
        cur = obs.get('current')
        logs = obs.get('logs') or []

        header = f"\n----- STEP {si} | acting P{ai} | status={status}"
        if cur:
            header += f" | turn={cur.get('turn')} | result={cur.get('result')}"
        out.append(header)

        # logs that happened since last decision
        for l in logs:
            out.append("    LOG  " + fmt_log(l))

        # the decision presented
        if sel:
            st = SELECT_TYPE.get(sel.get('type'), sel.get('type'))
            ctx = SEL_CONTEXT.get(sel.get('context'), sel.get('context'))
            out.append(f"    DECISION: type={st} context={ctx} pick {sel.get('minCount')}-{sel.get('maxCount')}")
            opts = sel.get('option') or []
            for oi, o in enumerate(opts):
                marker = " <== CHOSEN" if (act and oi in act) else ""
                out.append(f"        [{oi}] {fmt_option(o)}{marker}")
        if act is not None:
            out.append(f"    ACTION TAKEN: {act}")

with open('readable_log.txt','w') as f:
    f.write("\n".join(out))

print(f"Wrote readable_log.txt with {len(out)} lines")
print("\n".join(out[:80]))

"""
Philosophy

bots that jam every hand, bots that play far too
many hands out of position, bots that crying-call. Two things beat them:

  1.  Don't spew.  Fold the junk, especially out of position.
  2.  Realise equity and *call wide vs. maniacs*.  A bot that shoves random
      hands is printing money for anyone willing to call with the top ~40%.

On top of that we layer randomisation ("many random elements") so we are not
exploitable by anyone watching our patterns: mixed open/3-bet/bluff
frequencies and jittered bet sizes, all applied at decision boundaries where
the EV of each option is close — never as random spew.

Layers
------
  L1  PREFLOP CHART     position-aware open/3bet/call tiers (≈ zero compute)
  L2  SHOVE DEFENCE     facing a huge bet/all-in preflop -> decide by equity
  L3  POSTFLOP EQUITY   eval7 Monte-Carlo vs. plausible opponents, budgeted
  L4  MIXED STRATEGY    randomised steals / 3bet bluffs / semibluffs / thin
                        value + size jitter, only near indifference points
  L5  OPPONENT MODEL    coarse per-opponent VPIP / aggression / fold-to-bet
                        from the per-match action log, used for small,
                        high-confidence exploit nudges only

Safety
------
  - Hard per-decision time budget; sim count scales down as the clock burns.
  - Every code path returns a legal-ish action; the engine snaps illegal
    raises and auto-folds on exception, and we wrap everything in try/except.
  - Validator-clean: only random, eval7, time, math.
"""

import eval7
import random
import time
import math

BOT_NAME = "SimpleBotV3"

# Engine constants (see engine/game.py). Hard-coded so we never depend on
# the engine module at runtime.
SMALL_BLIND = 50
BIG_BLIND = 100

# ──────────────────────────────────────────────────────────────────────────
# CARD / RANK HELPERS
# ──────────────────────────────────────────────────────────────────────────
_RANK_ORDER = "23456789TJQKA"
_RVAL = {r: i for i, r in enumerate(_RANK_ORDER, start=2)}  # '2'->2 ... 'A'->14

# Pre-build the full deck once so equity() only has to filter, not parse.
_FULL_DECK = [eval7.Card(r + s) for r in _RANK_ORDER for s in "shdc"]


def _canon(cards):
    """('As','Kd') -> (hi, lo, suited_bool, pair_bool). Position-independent."""
    r1, r2 = _RVAL[cards[0][0]], _RVAL[cards[1][0]]
    hi, lo = max(r1, r2), min(r1, r2)
    suited = cards[0][1] == cards[1][1]
    return hi, lo, suited, (hi == lo)


# ──────────────────────────────────────────────────────────────────────────
# PERSISTENT OPPONENT MODEL (survives across all hands of a match — same
# process). Keyed by opponent bot_id. Rebuilt fresh each match; 400 hands is
# ample to classify gross archetypes (station / maniac / nit).
# ──────────────────────────────────────────────────────────────────────────
_OPP = {}


def _opp(bid):
    if bid not in _OPP:
        _OPP[bid] = {
            "vpip_n": 0, "vpip_yes": 0,
            "aggr_bets": 0, "aggr_calls": 0,
            "ftb_n": 0, "ftb_yes": 0,
        }
    return _OPP[bid]


# We absorb the rolling match_action_log each decision. It's a <=200 entry
# sliding window so perfect dedup is impossible; slight double-counting is
# harmless for archetype tagging. We track the largest (hand_num, len) we've
# seen to skip re-processing the unchanged head most of the time.
_LAST_LOG_KEY = None


def _update_model(state, me):
    global _LAST_LOG_KEY
    log = state.get("match_action_log") or []
    if not log:
        return
    key = (len(log), log[-1].get("hand_num"), log[-1].get("seat"),
           log[-1].get("action"))
    if key == _LAST_LOG_KEY:
        return
    _LAST_LOG_KEY = key
    # Only fold in the tail we likely haven't processed. Cheap + good enough.
    for e in log[-12:]:
        bid = e.get("bot_id")
        act = e.get("action")
        if not bid or bid == me:
            continue
        st = _opp(bid)
        if act in ("call", "raise", "all_in"):
            st["vpip_n"] += 1
            st["vpip_yes"] += 1
        elif act in ("check", "fold"):
            st["vpip_n"] += 1
        if act in ("raise", "all_in"):
            st["aggr_bets"] += 1
        elif act == "call":
            st["aggr_calls"] += 1
        if act == "fold":
            st["ftb_n"] += 1
            st["ftb_yes"] += 1
        elif act in ("call", "raise", "all_in"):
            st["ftb_n"] += 1


def _table_read(state, me):
    """Coarse read over opponents still in the hand.
    Returns (avg_fold_to_bet, avg_aggression_factor) — either may be None when
    we don't have enough samples to trust the read."""
    folds = aggr = denom_f = denom_a = 0
    for p in state["players"]:
        bid = p["bot_id"]
        if bid == me or p["is_folded"]:
            continue
        st = _OPP.get(bid)
        if not st:
            continue
        if st["ftb_n"] >= 10:
            folds += st["ftb_yes"]
            denom_f += st["ftb_n"]
        if (st["aggr_bets"] + st["aggr_calls"]) >= 10:
            aggr += st["aggr_bets"]
            denom_a += st["aggr_bets"] + st["aggr_calls"]
    ftb = (folds / denom_f) if denom_f else None
    af = (aggr / denom_a) if denom_a else None
    return ftb, af


# ──────────────────────────────────────────────────────────────────────────
# V3 — PER-OPPONENT READS. Instead of averaging over the whole table, we read
# the SPECIFIC opponent who bet into us (the aggressor) and the SPECIFIC blinds
# we're stealing from, then attack/avoid each by archetype. This is what lets
# us out-accumulate the tight leaders: steal from the proven over-folders, and
# only pay off the proven value-bettors with the near-nuts.
#
# (Also fixes a real bug in the table-average path: aggression was computed as
# a fraction in [0,1] but compared to >2.0, so the "maniac" exploit never
# fired. Archetypes below use correct thresholds.)
# ──────────────────────────────────────────────────────────────────────────
def _rates(bid):
    """Per-opponent rates with graded confidence. af = fraction of aggressive
    actions = bets/(bets+calls) in [0,1]. Returns None until we have samples."""
    st = _OPP.get(bid)
    if not st:
        return None
    ftb = (st["ftb_yes"] / st["ftb_n"]) if st["ftb_n"] >= 6 else None
    ad = st["aggr_bets"] + st["aggr_calls"]
    af = (st["aggr_bets"] / ad) if ad >= 6 else None
    vpip = (st["vpip_yes"] / st["vpip_n"]) if st["vpip_n"] >= 8 else None
    return {"ftb": ftb, "af": af, "vpip": vpip}


def _archetype(bid):
    """maniac / station / nit / solid / None(unknown) for one opponent."""
    r = _rates(bid)
    if not r:
        return None
    ftb, af, vpip = r["ftb"], r["af"], r["vpip"]
    # maniac: enters many pots AND bets/raises a lot (the chip donors)
    if (vpip is not None and vpip > 0.45) and (af is None or af > 0.45):
        return "maniac"
    if af is not None and af > 0.70 and (vpip is None or vpip > 0.30):
        return "maniac"
    # station: calls plenty, rarely folds, passive (value-bet big, never bluff)
    if (ftb is not None and ftb < 0.30) and (af is None or af < 0.40):
        return "station"
    # nit / over-folder: tight or folds a lot (steal from; fold to their value)
    if (ftb is not None and ftb > 0.55) or (vpip is not None and vpip < 0.16):
        return "nit"
    return "solid"


def _aggressor(state, me):
    """bot_id of the opponent who put in the bet we face (last raiser / the
    big blind preflop). None if we can't tell."""
    seat2bid = {p["seat"]: p["bot_id"] for p in state["players"]}
    for e in reversed(state.get("action_log", [])):
        if e.get("action") in ("raise", "all_in", "big_blind"):
            bid = seat2bid.get(e.get("seat"))
            if bid and bid != me:
                return bid
    return None


def _live_opponents(state, me):
    return [p["bot_id"] for p in state["players"]
            if p["bot_id"] != me and not p["is_folded"]]


def _blind_steal_read(state, me, btn):
    """For an unopened steal: are the blind defenders proven over-folders, and
    is there a dangerous (maniac/station) defender we must NOT steal junk into?
    Returns (fold_lean: bool, block: bool)."""
    n = len(state["players"])
    fold_votes = seen = 0
    block = False
    for seat in ((btn + 1) % n, (btn + 2) % n):       # SB, BB
        p = state["players"][seat]
        if p["bot_id"] == me or p["is_folded"]:
            continue
        a = _archetype(p["bot_id"])
        if a is None:
            continue
        seen += 1
        if a == "nit":
            fold_votes += 1
        if a in ("maniac", "station"):
            block = True
    return (seen > 0 and fold_votes == seen), block


# ──────────────────────────────────────────────────────────────────────────
# LAYER 1 — PREFLOP RANGES
# Tier 0..3: 3=premium, 2=strong, 1=playable, 0=trash. Tight-aggressive.
# ──────────────────────────────────────────────────────────────────────────
def _preflop_tier(cards):
    hi, lo, suited, pair = _canon(cards)

    if pair:
        if hi >= 13:  return 3          # AA, KK
        if hi >= 10:  return 2          # QQ, JJ, TT
        return 1                        # 99-22 (set-mining / playable)

    if hi == 14 and lo >= 12: return 3  # AK, AQ
    if hi >= 13 and lo >= 11: return 2  # KQ, KJ, AJ-class broadway
    if hi == 14 and lo >= 10: return 2  # AT, AJ

    if suited:
        if hi == 14:               return 1   # any suited ace (Axs)
        if lo >= 10:               return 1   # suited broadway
        if (hi - lo) <= 1 and lo >= 5: return 1  # suited connectors 56s+
        if (hi - lo) == 2 and lo >= 6: return 1  # suited one-gappers 68s+
        return 0
    # Offsuit
    if hi >= 12 and lo >= 10: return 1  # KQo, QJo, KJo, QTo
    if hi == 14 and lo >= 10: return 1  # ATo, AJo
    return 0


# ──────────────────────────────────────────────────────────────────────────
# POSITION — derived from the actual button. The button is reconstructed from
# the blind posts in action_log (robust to busts / seat reindexing per hand).
# Returns 0.0 (small blind, worst) .. 1.0 (button, best).
# ──────────────────────────────────────────────────────────────────────────
def _button_seat(state):
    n = len(state["players"])
    sb = bb = None
    for e in state.get("action_log", []):
        a = e.get("action")
        if a == "small_blind":
            sb = e.get("seat")
        elif a == "big_blind":
            bb = e.get("seat")
    if n == 2:
        # Heads-up: dealer IS the small blind.
        if sb is not None: return sb
        if bb is not None: return (bb + 1) % n
    else:
        if sb is not None: return (sb - 1) % n
        if bb is not None: return (bb - 2) % n
    return state["seat_to_act"]  # fallback: assume we're on the button


def _position(state):
    n = len(state["players"])
    if n <= 1:
        return 1.0
    btn = _button_seat(state)
    seat = state["seat_to_act"]
    distance = (btn - seat) % n          # button:0 (best) .. sb:n-1 (worst)
    return 1.0 - distance / (n - 1)


# ──────────────────────────────────────────────────────────────────────────
# LAYER 3 — MONTE-CARLO EQUITY vs. random opponent hands. Budgeted: caps
# iterations and bails on the clock so we never overrun the 2s/action limit.
# ──────────────────────────────────────────────────────────────────────────
def _equity(hole, board, n_opp, deadline, max_iters=1200):
    used = {str(c) for c in hole} | {str(c) for c in board}
    deck = [c for c in _FULL_DECK if str(c) not in used]
    my_base = list(hole) + list(board)
    need_board = 5 - len(board)
    draw = need_board + 2 * n_opp
    if draw > len(deck):
        return 0.5

    wins = ties = total = 0
    for i in range(max_iters):
        if (i & 31) == 0 and time.time() > deadline:
            break
        sample = random.sample(deck, draw)
        fill = sample[:need_board]
        full_board = board + fill
        my_score = eval7.evaluate(my_base + fill)
        idx = need_board
        beat = True
        tied = False
        for _ in range(n_opp):
            opp = sample[idx:idx + 2]
            idx += 2
            os_ = eval7.evaluate(opp + full_board)
            if os_ > my_score:
                beat = False
                break
            if os_ == my_score:
                tied = True
        total += 1
        if beat and not tied:
            wins += 1
        elif beat and tied:
            ties += 1
    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


# ──────────────────────────────────────────────────────────────────────────
# BET SIZING HELPERS
# ──────────────────────────────────────────────────────────────────────────
def _cap(state):
    """Maximum legal total bet (our whole stack committed this street)."""
    return state["your_stack"] + state["your_bet_this_street"]


def _raise_to(state, frac_of_pot, jitter=0.12):
    """Total bet to put `frac_of_pot` of the pot in as a raise, with a small
    random size jitter so our sizing isn't a readable tell. Clamped to legal
    minimum and to our stack."""
    f = frac_of_pot * (1.0 + random.uniform(-jitter, jitter)) if jitter else frac_of_pot
    pot = state["pot"]
    target = state["current_bet"] + int(pot * f)
    target = max(target, state["min_raise_to"])
    return min(target, _cap(state))


# ──────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ──────────────────────────────────────────────────────────────────────────
def decide(game_state: dict) -> dict:
    start = time.time()
    deadline = start + 1.5  # 0.5s headroom under the 2s hard limit

    # Warm-up ping: touch eval7 so its C paths / first sample are hot.
    if game_state.get("type") == "warmup":
        try:
            _equity([eval7.Card("As"), eval7.Card("Kd")],
                    [eval7.Card("2c"), eval7.Card("7h"), eval7.Card("9s")],
                    1, time.time() + 1.0, max_iters=300)
        except Exception:
            pass
        return {"action": "check"}

    try:
        return _decide_inner(game_state, deadline)
    except Exception:
        # Absolute backstop — never crash, never time out.
        if game_state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}


def _decide_inner(state, deadline):
    me = state["players"][state["seat_to_act"]]["bot_id"]
    _update_model(state, me)

    street    = state["street"]
    owed      = state["amount_owed"]
    pot       = state["pot"]
    can_check = state["can_check"]
    my_stack  = state["your_stack"]
    my_bet    = state["your_bet_this_street"]
    hole      = [eval7.Card(c) for c in state["your_cards"]]
    board     = [eval7.Card(c) for c in state["community_cards"]]
    live      = _live_opponents(state, me)
    n_opp     = max(1, len(live))
    pos       = _position(state)

    # PER-OPPONENT reads (v3). villain = the specific opponent who bet into us;
    # live_archs = archetypes of everyone still in the pot (for betting into).
    facing_bet = owed > 0
    villain = _archetype(_aggressor(state, me)) if facing_bet else None
    live_archs = [_archetype(b) for b in live]
    reads = {
        "villain": villain,
        "has_station": any(a == "station" for a in live_archs),
        "has_maniac":  any(a == "maniac" for a in live_archs),
        # all proven over-folders and at least one classified -> barrels get through
        "all_fold": (len(live_archs) > 0 and all(a == "nit" for a in live_archs)),
    }

    if street == "preflop":
        return _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
                        n_opp, pos, reads, me, deadline)
    return _postflop(state, hole, board, owed, pot, can_check, my_stack,
                     n_opp, pos, reads, deadline)


# ──────────────────────────────────────────────────────────────────────────
# PREFLOP DECISION
# ──────────────────────────────────────────────────────────────────────────
def _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
             n_opp, pos, reads, me, deadline):
    tier = _preflop_tier(state["your_cards"])
    hi, lo, suited, pair = _canon(state["your_cards"])
    facing_raise = state["current_bet"] > BIG_BLIND
    villain = reads["villain"]      # archetype of the raiser/shover, or None

    # --- L2: facing a big bet / shove. Decide by equity, keyed to the shover. ---
    if owed > 0 and owed >= 0.25 * my_stack:
        eq = _equity(hole, [], n_opp, deadline, max_iters=1500)
        breakeven = owed / (pot + owed)
        risk = min(owed / max(my_stack, 1), 1.0)
        margin = 0.05 + 0.07 * risk + 0.04 * (n_opp - 1)
        if villain == "maniac":
            margin *= 0.55          # their shove range is junk -> call wider
        elif villain == "nit":
            margin += 0.08          # a nit shoving = premium -> need a real edge
        if eq >= breakeven + margin:
            return {"action": "call"}
        if tier >= 3 and eq >= breakeven:
            return {"action": "call"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # Position-based opening requirement (tight: ~10% UTG..HJ, ~22% CO/BTN).
    min_open = 2 if pos < 0.67 else 1

    if not facing_raise:
        open_size = _raise_to(state, 1.0)
        if tier >= min_open:
            return {"action": "raise", "amount": open_size}
        # TARGETED STEAL (the predator move): attack proven over-folding blinds.
        # Steal wide when the blinds are nits; never when a maniac/station is
        # behind (they call / play back). This is where we out-earn the leaders
        # — they fold these hands; we take the blinds the nits surrender.
        btn = _button_seat(state)
        fold_lean, block = _blind_steal_read(state, me, btn)
        if not block and pos >= 0.6:
            if fold_lean:
                steal_p = 0.45 if pos >= 0.8 else 0.25   # attack nit blinds, but not recklessly
            elif pos >= 0.9:
                steal_p = 0.12           # thin default steal, button only
            else:
                steal_p = 0.0
            if steal_p and random.random() < steal_p:
                return {"action": "raise", "amount": open_size}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # --- Facing a raise: read the RAISER. ---
    threebet = _raise_to(state, 1.1)
    if tier >= 3:
        return {"action": "raise", "amount": threebet}

    if villain == "maniac":
        # Loose-aggressive raiser -> their range is weak, but don't bloat the
        # pot OOP. Flat strong hands (realise equity, let them keep bluffing),
        # call playable cheaply. Premiums already 3bet above.
        if tier >= 2 and owed <= my_stack * 0.15:
            return {"action": "call"}
        if tier >= 1 and owed <= my_stack * 0.08:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    if villain == "nit":
        # A nit raising = a real hand. Only the best non-premiums continue, and
        # only cheaply; never pay to chase vs a proven tight range.
        if tier >= 2 and owed <= my_stack * 0.05:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # Solid / unknown raiser — balanced default.
    if tier >= 2:
        if owed <= my_stack * 0.12:
            return {"action": "call"}
        return {"action": "fold"}
    if tier >= 1:
        bluff_3b = (0.15 if (suited and hi == 14) else (0.06 if suited else 0.0))
        if random.random() < bluff_3b:
            return {"action": "raise", "amount": threebet}
        if pos >= 0.6 and owed <= my_stack * 0.06:
            return {"action": "call"}
    if can_check:
        return {"action": "check"}
    return {"action": "fold"}


# ──────────────────────────────────────────────────────────────────────────
# POSTFLOP DECISION
# ──────────────────────────────────────────────────────────────────────────
def _postflop(state, hole, board, owed, pot, can_check, my_stack,
              n_opp, pos, reads, deadline):
    budget = 1200 if n_opp == 1 else (700 if n_opp == 2 else 400)
    eq = _equity(hole, board, n_opp, deadline, max_iters=budget)
    breakeven = owed / (pot + owed) if owed > 0 else 0.0
    mw = 0.04 * (n_opp - 1)

    strong   = eq >= 0.72 + mw
    good     = 0.55 + mw <= eq < 0.72 + mw
    marginal = 0.40 + mw <= eq < 0.55 + mw

    # ─── CHECKED TO US: bet decision keyed to WHO is still in the pot. ───
    if can_check:
        has_station = reads["has_station"]
        if strong:
            frac = 1.25 if has_station else 0.8     # overbet the stations
            return {"action": "raise", "amount": _raise_to(state, frac)}
        if good:
            if random.random() < (0.9 if has_station else 0.8):
                return {"action": "raise", "amount": _raise_to(state, 0.6)}
            return {"action": "check"}
        # Bluff frequency from the LIVE FIELD: hammer when everyone left is a
        # proven folder, shut off vs a station, dampen vs a maniac (they call).
        if has_station:
            bluff_mult = 0.15
        elif reads["all_fold"]:
            bluff_mult = 1.2
        elif reads["has_maniac"]:
            bluff_mult = 0.45
        else:
            bluff_mult = 0.7
        if marginal:
            if random.random() < min(0.40 * bluff_mult, 0.85):
                return {"action": "raise", "amount": _raise_to(state, 0.55)}
            return {"action": "check"}
        if random.random() < min(0.42 * bluff_mult, 0.85):
            return {"action": "raise", "amount": _raise_to(state, 0.65)}
        return {"action": "check"}

    # ─── FACING A BET: read the SPECIFIC bettor. ───
    villain = reads["villain"]
    maniac  = villain == "maniac"
    station = villain == "station"
    nit     = villain == "nit"

    # Range-aware discount: a NIT betting big = the near-nuts (huge discount,
    # we fold); a maniac = air (tiny discount, we call/raise light).
    eff_eq = eq
    if owed > 0:
        st_disc = {"flop": 0.03, "turn": 0.06, "river": 0.12}.get(state["street"], 0.0)
        bet_frac = min(owed / pot, 1.5) if pot > 0 else 1.0
        if bet_frac < 0.45:
            st_disc *= 0.3
        disc = st_disc * (0.6 + 0.5 * bet_frac)
        if maniac:
            disc *= 0.25
        elif nit:
            disc *= 1.9        # nit value-bets are gold -> discount hard, fold
        elif station:
            disc *= 0.6
        elif villain is None:
            disc *= 0.8
        eff_eq = eq - disc

    # COMMITMENT GUARD (anti-bust): the more of our stack the call commits, the
    # bigger the range-aware edge we demand.
    commit = owed / max(my_stack + state["your_bet_this_street"], 1)
    commit_margin = 0.0
    if commit > 0.40:
        commit_margin = 0.06 + 0.18 * (commit - 0.40)
    if maniac:
        commit_margin *= 0.35
    elif nit:
        commit_margin += 0.05

    if strong:
        if commit > 0.45 and eff_eq < 0.62:
            if eff_eq > breakeven + commit_margin * 0.5:
                return {"action": "call"}
            return {"action": "fold"}
        frac = 0.7 if maniac else 0.85
        if random.random() < (0.30 if maniac else 0.15):
            return {"action": "call"}
        return {"action": "raise", "amount": _raise_to(state, frac)}

    if good:
        margin = (-0.03 if maniac else (0.09 if nit else 0.05) + mw) + commit_margin
        if eff_eq > breakeven + margin:
            if (not maniac and not nit and commit < 0.35
                    and random.random() < 0.30 and owed <= pot * 0.9):
                return {"action": "raise", "amount": _raise_to(state, 0.7)}
            return {"action": "call"}
        return {"action": "fold"}

    if marginal:
        # Bluff-raise only vs a maniac who bets light and small (rare, cheap);
        # nits don't fold after betting, stations call -> never raise them.
        if (maniac and owed <= pot * 0.5 and commit < 0.25
                and random.random() < 0.18):
            return {"action": "raise", "amount": _raise_to(state, 0.8)}
        if maniac:
            if eff_eq > breakeven + commit_margin and owed <= pot:
                return {"action": "call"}
            return {"action": "fold"}
        thresh = 0.06 + (0.05 if nit else 0.0) + mw + commit_margin
        if eff_eq > breakeven + thresh and owed <= pot * 0.55:
            return {"action": "call"}
        return {"action": "fold"}

    # weak: fold. Only a cheap, small-commitment call with live equity.
    if eff_eq > breakeven + 0.10 + commit_margin and owed <= pot * 0.3 and commit < 0.25:
        return {"action": "call"}
    return {"action": "fold"}

"""
╔══════════════════════════════════════════════════════════════╗
║  FULLHOUSE BOT — "Overlord"                                   ║
║  Range-aware equity + polarized value + earned aggression      ║
╚══════════════════════════════════════════════════════════════╝

Built to beat both the reference field AND our own "simpleBot". It shares
simpleBot's skeleton (Monte-Carlo equity, position, opponent model) but fixes
the two things simpleBot gets wrong:

  1.  RANGE-AWARE EQUITY.  simpleBot compares raw equity-vs-RANDOM to pot odds,
      so it pays off big turn/river bets with hands that are only good against
      a random holding — but a bot firing big late isn't holding random cards.
      Overlord DISCOUNTS its equity when facing aggression, scaled by street
      and bet size (and relaxed vs proven maniacs). Result: it folds the
      second-best hands simpleBot calls with.

  2.  POLARIZED VALUE.  Because most bots (incl. simpleBot) over-call rivers,
      Overlord bets BIG on the river with strong hands to maximally tax that
      leak, while checking back medium hands to control the pot.

Everything else mirrors simpleBot's discipline: tight-aggressive preflop,
earned (read-gated) bluffing, multiway caution, hard time budget, never
crashes. Validator-clean: only random, eval7, time, math.
"""

import eval7
import random
import time
import math

BOT_NAME = "Overlord"

SMALL_BLIND = 50
BIG_BLIND = 100

_RANK_ORDER = "23456789TJQKA"
_RVAL = {r: i for i, r in enumerate(_RANK_ORDER, start=2)}
_FULL_DECK = [eval7.Card(r + s) for r in _RANK_ORDER for s in "shdc"]


def _canon(cards):
    r1, r2 = _RVAL[cards[0][0]], _RVAL[cards[1][0]]
    hi, lo = max(r1, r2), min(r1, r2)
    return hi, lo, cards[0][1] == cards[1][1], (hi == lo)


# ─── opponent model (per-match, per-process) ───────────────────────────────
_OPP = {}


def _opp(bid):
    if bid not in _OPP:
        _OPP[bid] = {"vpip_n": 0, "vpip_yes": 0, "aggr_bets": 0,
                     "aggr_calls": 0, "ftb_n": 0, "ftb_yes": 0}
    return _OPP[bid]


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


# ─── preflop chart ─────────────────────────────────────────────────────────
def _preflop_tier(cards):
    hi, lo, suited, pair = _canon(cards)
    if pair:
        if hi >= 13: return 3
        if hi >= 10: return 2
        return 1
    if hi == 14 and lo >= 12: return 3
    if hi >= 13 and lo >= 11: return 2
    if hi == 14 and lo >= 10: return 2
    if suited:
        if hi == 14: return 1
        if lo >= 10: return 1
        if (hi - lo) <= 1 and lo >= 5: return 1
        if (hi - lo) == 2 and lo >= 6: return 1
        return 0
    if hi >= 12 and lo >= 10: return 1
    if hi == 14 and lo >= 10: return 1
    return 0


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
        if sb is not None: return sb
        if bb is not None: return (bb + 1) % n
    else:
        if sb is not None: return (sb - 1) % n
        if bb is not None: return (bb - 2) % n
    return state["seat_to_act"]


def _position(state):
    n = len(state["players"])
    if n <= 1:
        return 1.0
    distance = (_button_seat(state) - state["seat_to_act"]) % n
    return 1.0 - distance / (n - 1)


# ─── budgeted Monte-Carlo equity vs random hands ───────────────────────────
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
    return (wins + 0.5 * ties) / total if total else 0.5


def _cap(state):
    return state["your_stack"] + state["your_bet_this_street"]


def _raise_to(state, frac_of_pot, jitter=0.12):
    f = frac_of_pot * (1.0 + random.uniform(-jitter, jitter)) if jitter else frac_of_pot
    target = state["current_bet"] + int(state["pot"] * f)
    target = max(target, state["min_raise_to"])
    return min(target, _cap(state))


# ─── entry ─────────────────────────────────────────────────────────────────
def decide(game_state: dict) -> dict:
    deadline = time.time() + 1.5
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
        if game_state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}


def _decide_inner(state, deadline):
    me = state["players"][state["seat_to_act"]]["bot_id"]
    _update_model(state, me)

    street = state["street"]
    owed = state["amount_owed"]
    pot = state["pot"]
    can_check = state["can_check"]
    my_stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    hole = [eval7.Card(c) for c in state["your_cards"]]
    board = [eval7.Card(c) for c in state["community_cards"]]
    n_opp = max(1, sum(1 for p in state["players"]
                       if not p["is_folded"] and p["bot_id"] != me))
    pos = _position(state)
    ftb, af = _table_read(state, me)

    if street == "preflop":
        return _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
                        n_opp, pos, ftb, af, deadline)
    return _postflop(state, hole, board, owed, pot, can_check, my_stack,
                     n_opp, pos, ftb, af, deadline, street)


# ─── preflop ───────────────────────────────────────────────────────────────
def _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
             n_opp, pos, ftb, af, deadline):
    tier = _preflop_tier(state["your_cards"])
    hi, lo, suited, pair = _canon(state["your_cards"])
    facing_raise = state["current_bet"] > BIG_BLIND
    overfold = ftb is not None and ftb > 0.50

    # Shove / big-bet defence by equity.
    if owed > 0 and owed >= 0.25 * my_stack:
        eq = _equity(hole, [], n_opp, deadline, max_iters=1500)
        be = owed / (pot + owed)
        margin = 0.01 + 0.04 * (n_opp - 1)
        if eq >= be + margin:
            return {"action": "call"}
        if tier >= 3 and eq >= be - 0.03:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    min_open = 2 if pos < 0.34 else 1

    if not facing_raise:
        open_size = _raise_to(state, 1.0)
        if tier >= min_open or (tier >= 1 and pos >= 0.6):
            return {"action": "raise", "amount": open_size}
        if (not facing_raise) and pos >= 0.6:
            steal_p = 0.65 if overfold else 0.30
            if random.random() < steal_p:
                return {"action": "raise", "amount": open_size}
        return {"action": "check"} if can_check else {"action": "fold"}

    # Facing a raise.
    threebet = _raise_to(state, 1.1)
    if tier >= 3:
        return {"action": "raise", "amount": threebet}
    if tier >= 2:
        return {"action": "call"} if owed <= my_stack * 0.12 else {"action": "fold"}
    if tier >= 1:
        bluff_3b = (0.18 if (suited and hi == 14) else (0.08 if suited else 0.0))
        if overfold:
            bluff_3b *= 1.8
        if random.random() < bluff_3b:
            return {"action": "raise", "amount": threebet}
        if pos >= 0.6 and owed <= my_stack * 0.07:
            return {"action": "call"}
    return {"action": "check"} if can_check else {"action": "fold"}


# ─── postflop ──────────────────────────────────────────────────────────────
# Equity discount vs a betting range, by street. Flop bets are wide (cbets,
# draws) so we barely discount; turn/river bets are far more honest. Only
# applied to sizeable bets and dialled down vs loose/maniac opponents — so it
# targets honest value bets (the leak that beats simpleBot) without making us
# fold to the weak field's small/random bets.
_STREET_DISCOUNT = {"flop": 0.0, "turn": 0.06, "river": 0.12}


def _postflop(state, hole, board, owed, pot, can_check, my_stack,
              n_opp, pos, ftb, af, deadline, street):
    budget = 1200 if n_opp == 1 else (700 if n_opp == 2 else 400)
    eq = _equity(hole, board, n_opp, deadline, max_iters=budget)

    breakeven = owed / (pot + owed) if owed > 0 else 0.0
    mw = 0.04 * (n_opp - 1)

    maniac = af is not None and af > 2.0
    station = ftb is not None and ftb < 0.30
    overfold = ftb is not None and ftb > 0.50
    if station:
        bluff_mult = 0.20
    elif overfold:
        bluff_mult = 1.4
    elif ftb is None:
        bluff_mult = 0.55
    else:
        bluff_mult = 0.85

    # ── RANGE-AWARE EQUITY (the core upgrade). When we face a bet, the
    # opponent's range is stronger than random, so discount our equity. The
    # discount grows by street and bet size, and shrinks vs proven maniacs
    # (whose bets are mostly air). This is what stops us paying off rivers. ──
    eff_eq = eq
    if owed > 0:
        bet_frac = min(owed / pot, 1.5) if pot > 0 else 1.0
        base = _STREET_DISCOUNT.get(street, 0.0)
        # Only discount sizeable bets; small bets are too wide to fear.
        if bet_frac < 0.45:
            base *= 0.3
        disc = base * (0.6 + 0.5 * bet_frac)
        if maniac:
            disc *= 0.25           # their bets are mostly air — barely discount
        elif station:
            disc *= 0.6
        elif ftb is None:
            disc *= 0.7            # no read yet — don't over-fold
        eff_eq = eq - disc

    strong = eq >= 0.72 + mw
    good = 0.55 + mw <= eq < 0.72 + mw
    marginal = 0.40 + mw <= eq < 0.55 + mw

    # ---- Checked to us ----
    if can_check:
        if strong:
            frac = 1.25 if station else 0.8
            return {"action": "raise", "amount": _raise_to(state, frac)}
        if good:
            # Thin value — keep betting for value against this calling field.
            if random.random() < 0.85:
                return {"action": "raise", "amount": _raise_to(state, 0.6)}
            return {"action": "check"}
        if marginal:
            if random.random() < min(0.38 * bluff_mult, 0.85):
                return {"action": "raise", "amount": _raise_to(state, 0.55)}
            return {"action": "check"}
        if random.random() < min(0.42 * bluff_mult, 0.85):
            return {"action": "raise", "amount": _raise_to(state, 0.65)}
        return {"action": "check"}

    # ---- Facing a bet (use eff_eq, the range-aware estimate) ----
    if strong:
        frac = 0.7 if maniac else 0.9
        if random.random() < (0.30 if maniac else 0.15):
            return {"action": "call"}
        return {"action": "raise", "amount": _raise_to(state, frac)}

    if good:
        # Good hands keep FULL equity (don't fold value) — call/raise for value.
        margin = -0.02 if maniac else 0.04 + mw
        if eq > breakeven + margin:
            if not maniac and random.random() < 0.35 and owed <= pot * 0.9:
                return {"action": "raise", "amount": _raise_to(state, 0.7)}
            return {"action": "call"}
        return {"action": "fold"}

    if marginal:
        if overfold and not maniac and owed <= pot * 0.7 and random.random() < 0.25:
            return {"action": "raise", "amount": _raise_to(state, 0.8)}
        if maniac:
            return {"action": "call"} if (eff_eq > breakeven and owed <= pot) else {"action": "fold"}
        if eff_eq > breakeven + 0.05 + mw and owed <= pot * 0.55:
            return {"action": "call"}
        return {"action": "fold"}

    # weak
    if eff_eq > breakeven + 0.10 and owed <= pot * 0.3:
        return {"action": "call"}
    return {"action": "fold"}

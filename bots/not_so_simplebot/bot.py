"""
not_so_simplebot  —  purpose-built to beat SimpleBotV2 (bots/simplebot/bot.py),
and to be a sound, aggressive no-limit hold'em bot in general.

How SimpleBotV2 loses (verified by reading its code, heads-up n=2):

  * It OPEN-FOLDS the small blind / button ~78% of the time (anything below its
    "playable" tier just folds instead of completing).
  * It FOLDS THE BIG BLIND ~89% to *any* raise — its only continuing hands
    facing a raise are tier>=2 (flat) and tier>=3 (3-bet) plus a rare suited
    3-bet bluff.  Critically, this fold frequency NEVER widens no matter how
    often we steal: its only "maniac" adjustment is on preflop all-in defense.
    So min-raising the button with any two cards is a permanent, un-counterable
    edge — we print its big blind almost 9 times in 10.
  * Postflop it folds when it lacks equity and range-discounts its hand versus
    big bets, so it can be c-bet and barreled off everything that misses.
  * It only raises / calls down with genuine strength, so we simply never pay
    off its big bets with marginal holdings.

Strategy
--------
  L1  PREFLOP   position-aware *wide* opening (any-two on the HU button),
                3-bet pressure (value + blocker bluffs) vs its foldy opens,
                equity-based all-in defense (anti-bust, like the target).
  L2  POSTFLOP  eval7 Monte-Carlo equity vs plausible hands, HIGH c-bet /
                stab frequency to harvest its folds, but value-weighted barrels
                and range-aware hero-call discipline so we never spew.
  L3  READ      light per-opponent VPIP / aggression / fold-to-bet model used
                only for high-confidence exploit nudges (hammer proven folders,
                shut off bluffs vs proven stations).

Safety: hard per-decision time budget, every path returns a legal-ish action,
the whole thing is wrapped in try/except, and we import only the validator-clean
set {eval7, random, time, math}.
"""

import eval7
import random
import time
import math

BOT_NAME = "NotSoSimpleBot"

SMALL_BLIND = 50
BIG_BLIND = 100

# ──────────────────────────────────────────────────────────────────────────
# CARD / RANK HELPERS
# ──────────────────────────────────────────────────────────────────────────
_RANK_ORDER = "23456789TJQKA"
_RVAL = {r: i for i, r in enumerate(_RANK_ORDER, start=2)}      # '2'->2 ... 'A'->14
_FULL_DECK = [eval7.Card(r + s) for r in _RANK_ORDER for s in "shdc"]


def _canon(cards):
    """('As','Kd') -> (hi, lo, suited_bool, pair_bool). Position-independent."""
    r1, r2 = _RVAL[cards[0][0]], _RVAL[cards[1][0]]
    hi, lo = max(r1, r2), min(r1, r2)
    suited = cards[0][1] == cards[1][1]
    return hi, lo, suited, (hi == lo)


def _chen_score(hi, lo, suited, pair):
    """Bill Chen preflop strength (~ -1.5 for 72o up to 20 for AA). Continuous,
    cheap, and a good monotone proxy for 'how wide am I willing to play this'."""
    pts = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}

    def hc(r):
        return pts.get(r, r / 2.0)

    if pair:
        return max(hc(hi) * 2.0, 5.0)
    score = hc(hi)
    if suited:
        score += 2.0
    gap = (hi - lo) - 1                  # 0 = connected, 1 = one-gapper, ...
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0
    if gap <= 1 and hi <= 11:            # straight-y bonus for low connectors
        score += 1.0
    return score


def _preflop_tier(hi, lo, suited, pair):
    """Coarse tier 0..3 (3 premium .. 0 trash) — used for value 3-bets and as a
    backstop on the all-in defense path."""
    if pair:
        if hi >= 13:  return 3          # AA, KK
        if hi >= 10:  return 2          # QQ, JJ, TT
        return 1                        # 99-22
    if hi == 14 and lo >= 12: return 3  # AK, AQ
    if hi >= 13 and lo >= 11: return 2  # KQ, KJ
    if hi == 14 and lo >= 10: return 2  # AJ, AT
    if suited:
        if hi == 14:                       return 1   # Axs
        if lo >= 10:                       return 1   # suited broadway
        if (hi - lo) <= 1 and lo >= 4:     return 1   # suited connectors
        if (hi - lo) == 2 and lo >= 5:     return 1   # suited one-gappers
        return 0
    if hi >= 12 and lo >= 10: return 1  # KQo, QJo, KJo, QTo
    if hi == 14 and lo >= 10: return 1  # ATo, AJo
    return 0


# ──────────────────────────────────────────────────────────────────────────
# OPPONENT MODEL (persists across all hands of a match; rebuilt per process).
# Keyed by opponent bot_id. Coarse archetype tags only.
# ──────────────────────────────────────────────────────────────────────────
_OPP = {}
_LAST_LOG_KEY = None


def _opp(bid):
    if bid not in _OPP:
        _OPP[bid] = {"vpip_n": 0, "vpip_yes": 0,
                     "aggr_bets": 0, "aggr_calls": 0,
                     "ftb_n": 0, "ftb_yes": 0}
    return _OPP[bid]


def _update_model(state, me):
    global _LAST_LOG_KEY
    log = state.get("match_action_log") or []
    if not log:
        return
    last = log[-1]
    key = (len(log), last.get("hand_num"), last.get("seat"), last.get("action"))
    if key == _LAST_LOG_KEY:
        return
    _LAST_LOG_KEY = key
    for e in log[-12:]:                       # only the likely-unprocessed tail
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


def _reads(state, me):
    """Returns (fold_to_bet, aggression_factor) averaged over opponents still in
    the hand. Either may be None until we have enough samples to trust it."""
    folds = aggr = denom_f = denom_a = vp_yes = vp_n = 0
    for p in state["players"]:
        bid = p["bot_id"]
        if bid == me or p["is_folded"]:
            continue
        st = _OPP.get(bid)
        if not st:
            continue
        if st["ftb_n"] >= 8:
            folds += st["ftb_yes"]
            denom_f += st["ftb_n"]
        if (st["aggr_bets"] + st["aggr_calls"]) >= 8:
            aggr += st["aggr_bets"]
            denom_a += st["aggr_bets"] + st["aggr_calls"]
        if st["vpip_n"] >= 8:
            vp_yes += st["vpip_yes"]
            vp_n += st["vpip_n"]
    ftb = (folds / denom_f) if denom_f else None
    af = (aggr / denom_a) if denom_a else None
    vpip = (vp_yes / vp_n) if vp_n else None
    return ftb, af, vpip


# ──────────────────────────────────────────────────────────────────────────
# POSITION — reconstruct the button from the blind posts in action_log.
# 0.0 (small blind, worst) .. 1.0 (button, best).
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
        if sb is not None: return sb            # heads-up: dealer IS the SB
        if bb is not None: return (bb + 1) % n
    else:
        if sb is not None: return (sb - 1) % n
        if bb is not None: return (bb - 2) % n
    return state["seat_to_act"]


def _position(state):
    n = len(state["players"])
    if n <= 1:
        return 1.0
    btn = _button_seat(state)
    seat = state["seat_to_act"]
    distance = (btn - seat) % n
    return 1.0 - distance / (n - 1)


# ──────────────────────────────────────────────────────────────────────────
# MONTE-CARLO EQUITY vs random opponent hands. Budgeted + clock-bailing.
# ──────────────────────────────────────────────────────────────────────────
def _equity(hole, board, n_opp, deadline, max_iters=1000):
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
            if eval7.evaluate(opp + full_board) > my_score:
                beat = False
                break
            if eval7.evaluate(opp + full_board) == my_score:
                tied = True
        total += 1
        if beat and not tied:
            wins += 1
        elif beat and tied:
            ties += 1
    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def _strong_made(hole, board):
    """True if we hold a SET or better (trips / straight / flush / full house /
    quads / straight flush) — i.e. a hand worth stacking off for. One pair and
    two pair return False: versus a tight bot that only re-raises with the nuts,
    fragile made hands (especially two-pair on a paired board) are NOT stack-off
    material, and equity-vs-random badly overrates them."""
    if len(board) < 3:
        return False
    try:
        ht = str(eval7.handtype(eval7.evaluate(list(hole) + list(board))))
    except Exception:
        return False
    return ht not in ("High Card", "Pair", "Two Pair")


# ──────────────────────────────────────────────────────────────────────────
# BET SIZING
# ──────────────────────────────────────────────────────────────────────────
def _cap(state):
    return state["your_stack"] + state["your_bet_this_street"]


def _raise_to(state, frac_of_pot, jitter=0.12):
    """Total bet that puts `frac_of_pot` of the pot in as a raise, with a small
    random size jitter so our sizing isn't a tell. Clamped to legal min & stack."""
    f = frac_of_pot * (1.0 + random.uniform(-jitter, jitter)) if jitter else frac_of_pot
    target = state["current_bet"] + int(state["pot"] * f)
    target = max(target, state["min_raise_to"])
    return min(target, _cap(state))


# ──────────────────────────────────────────────────────────────────────────
# PREFLOP OPENING WIDTH (min Chen score to open-raise) by position & table size
# ──────────────────────────────────────────────────────────────────────────
def _open_min_chen(pos, n):
    if n <= 2:
        # Heads-up: the button (SB) raises ANY two cards — the core exploit.
        return -999.0 if pos >= 0.5 else 5.0
    if pos >= 0.95:  return 1.0    # button — very wide
    if pos >= 0.75:  return 4.0    # cutoff
    if pos >= 0.55:  return 6.0    # hijack
    if pos >= 0.35:  return 8.0    # UTG / MP
    return 9.0                     # the blinds open tight


# ──────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ──────────────────────────────────────────────────────────────────────────
def decide(game_state: dict) -> dict:
    start = time.time()
    deadline = start + 1.5                     # 0.5s headroom under the 2s limit

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

    street    = state["street"]
    owed      = state["amount_owed"]
    pot       = state["pot"]
    can_check = state["can_check"]
    my_stack  = state["your_stack"]
    my_bet    = state["your_bet_this_street"]
    hole      = [eval7.Card(c) for c in state["your_cards"]]
    board     = [eval7.Card(c) for c in state["community_cards"]]
    n_opp = max(1, sum(1 for p in state["players"]
                       if not p["is_folded"] and p["bot_id"] != me))
    pos = _position(state)
    ftb, af, vpip = _reads(state, me)

    if street == "preflop":
        return _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
                        n_opp, pos, ftb, af, vpip, deadline)
    return _postflop(state, hole, board, owed, pot, can_check, my_stack,
                     n_opp, pos, ftb, af, vpip, deadline)


# ──────────────────────────────────────────────────────────────────────────
# PREFLOP
# ──────────────────────────────────────────────────────────────────────────
def _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
             n_opp, pos, ftb, af, vpip, deadline):
    n = len(state["players"])
    cards = state["your_cards"]
    hi, lo, suited, pair = _canon(cards)
    chen = _chen_score(hi, lo, suited, pair)
    tier = _preflop_tier(hi, lo, suited, pair)
    facing_raise = state["current_bet"] > BIG_BLIND

    folder  = ftb is not None and ftb > 0.50
    # MANIAC = loose AND aggressive (plays many hands and bets them). Requiring
    # high VPIP too is essential: a tight nit also has high aggression (raise-or-
    # fold), so af alone would misread the leaders as maniacs and call them too
    # wide. (Also fixes the old dead-code threshold: af is a fraction in [0,1].)
    maniac  = (af is not None and af > 0.60) and (vpip is not None and vpip > 0.40)

    # ---- All-in / huge-bet defense: decide by equity, anti-bust margin. ----
    if owed > 0 and owed >= 0.25 * my_stack:
        eq = _equity(hole, [], n_opp, deadline, max_iters=1500)
        breakeven = owed / (pot + owed)
        risk = min(owed / max(my_stack, 1), 1.0)
        margin = 0.05 + 0.07 * risk + 0.04 * (n_opp - 1)
        if maniac:
            margin *= 0.4
        if eq >= breakeven + margin:
            return {"action": "call"}
        if tier >= 3 and eq >= breakeven:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # ---- Unopened pot: open-raise our (wide, positional) range. ----
    if not facing_raise:
        size = _raise_to(state, 0.75 if n <= 2 else 1.0)   # small in HU; price is irrelevant to a chart-folder
        thr = _open_min_chen(pos, n)
        if chen >= thr:
            return {"action": "raise", "amount": size}
        # Limped-pot / BB option with a decent hand: raise it; else just check.
        if can_check and chen >= thr + 3.0:
            return {"action": "raise", "amount": size}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # ---- Facing a raise. Count prior raises to know the betting level:
    #      rc==1 a single open, rc==2 a 3-bet, rc>=3 a 4-bet+. The target only
    #      RE-raises with the nuts (AA/KK/AK/AQ), so we collapse to a premium
    #      range the moment we face a 3-bet+ — this is what stops the 4-bet
    #      call-down spews. ----
    rc = sum(1 for e in state.get("action_log", [])
             if e.get("action") in ("raise", "all_in"))

    if rc >= 2:
        # Facing a 3-bet or 4-bet: continue tight, re-raise only the very top.
        strong_cont = (pair and hi >= 11) or (hi == 14 and lo >= 12) or (chen >= 11.0)
        if strong_cont and owed <= my_stack * 0.30:
            if (pair and hi >= 13) or (hi == 14 and lo == 13):   # AA, KK, AK
                return {"action": "raise", "amount": _raise_to(state, 1.0)}
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # rc == 1: facing a single open-raise — our aggressive 3-bet / call / fold.
    threebet = _raise_to(state, 1.0)
    big_price = owed > my_stack * 0.5

    value_3b = (tier >= 3) or (pair and hi >= 11) or (chen >= 10.5)
    if value_3b and not big_price:
        return {"action": "raise", "amount": threebet}

    # 3-bet bluffs with blockers / playable suited hands — attacks the target's
    # foldy open range. Frequency ramps up versus a proven over-folder.
    bluff = 0.0
    if suited and hi == 14:                 # A2s-A5s wheel aces are ideal blockers
        bluff = 0.35
    elif suited and (hi - lo) <= 2 and lo >= 4:   # suited connectors / one-gappers
        bluff = 0.22
    elif suited and lo >= 10:               # suited broadway
        bluff = 0.18
    if folder:
        bluff *= 1.7
    if not big_price and random.random() < bluff:
        return {"action": "raise", "amount": threebet}

    # Flat-call range when the price is right (set-mining + suited playables).
    call_ok = (pair or chen >= 6.5 or tier >= 1)
    if call_ok and owed <= my_stack * 0.11:
        return {"action": "call"}
    if value_3b and owed <= my_stack * 0.25:    # premiums always continue
        return {"action": "call"}

    return {"action": "check"} if can_check else {"action": "fold"}


# ──────────────────────────────────────────────────────────────────────────
# POSTFLOP
# ──────────────────────────────────────────────────────────────────────────
def _postflop(state, hole, board, owed, pot, can_check, my_stack,
              n_opp, pos, ftb, af, vpip, deadline):
    budget = 1000 if n_opp == 1 else (650 if n_opp == 2 else 400)
    eq = _equity(hole, board, n_opp, deadline, max_iters=budget)

    breakeven = owed / (pot + owed) if owed > 0 else 0.0
    mw = 0.04 * (n_opp - 1)
    ip = pos >= 0.5
    street = state["street"]

    # MANIAC = loose AND aggressive (see _preflop). Gating on VPIP too avoids
    # misreading tight nits (high af, low vpip) as maniacs and paying them off.
    maniac   = (af is not None and af > 0.60) and (vpip is not None and vpip > 0.40)
    station  = ftb is not None and ftb < 0.30
    folder   = ftb is not None and ftb > 0.50

    # Bluff guardrail (earned aggression). We default a touch higher than the
    # target because most opponents — the target especially — fold to pressure;
    # but a proven station shuts bluffing nearly off.
    if station:
        bluff_mult = 0.20
    elif folder:
        bluff_mult = 1.5
    elif ftb is None:
        bluff_mult = 0.75
    else:
        bluff_mult = 1.0
    if ip:
        bluff_mult *= 1.25                     # cheaper / safer to bluff in position

    strong   = eq >= 0.72 + mw
    good     = 0.55 + mw <= eq < 0.72 + mw
    marginal = 0.42 + mw <= eq < 0.55 + mw
    nutty       = _strong_made(hole, board)            # set+ — happy to stack off
    late        = street in ("turn", "river")          # villain's range is stronger here

    # ── Checked to us (or first to act) ──────────────────────────────────────
    if can_check:
        if strong:
            # Value bet. But a fragile made hand (one pair / two pair) on a later
            # street gets pot-controlled, not turned into a stack-committing bet —
            # that's how we used to get felted by a better hand on the turn/river.
            if station:
                frac = 1.2
            elif late and not nutty:
                frac = 0.5
            else:
                frac = 0.8
            return {"action": "raise", "amount": _raise_to(state, frac)}
        if good:
            # Thin value; check more for pot control on later streets.
            p = 0.85 if not late else 0.45
            if station:
                p = min(p + 0.15, 0.95)
            if random.random() < p:
                return {"action": "raise", "amount": _raise_to(state, 0.5 if late else 0.6)}
            return {"action": "check"}
        if marginal:
            p_bet = min(0.50 * bluff_mult, 0.85)
            if late:
                p_bet *= 0.6
            if random.random() < p_bet:
                return {"action": "raise", "amount": _raise_to(state, 0.5)}
            return {"action": "check"}
        # Weak: c-bet / barrel bluff. Tapered HARD by street — the flop is where
        # the target folds; barreling turn/river into a range that called is mostly
        # lighting chips on fire (see the picked-off triple-barrels).
        street_freq = {"flop": 0.60, "turn": 0.28, "river": 0.12}.get(street, 0.4)
        p_bluff = min(street_freq * bluff_mult, 0.80)
        if random.random() < p_bluff:
            return {"action": "raise", "amount": _raise_to(state, 0.55)}
        return {"action": "check"}

    # ── Facing a bet ─────────────────────────────────────────────────────────
    # The target only RAISES postflop with real strength, so a re-raise of our
    # action ("raised") gets a big extra equity haircut and never gets bluff-
    # raised back. Everything here keys off the range-aware estimate `req`, so we
    # never stack off a hand that's only "strong versus random".
    my_bet_street = state["your_bet_this_street"]
    raised = my_bet_street > 0 and owed > 0            # check-raise / re-raise of us
    faced = max(pot - owed, 1)                         # pot the opponent bet into
    bet_frac = min(owed / faced, 2.0)

    commit = owed / max(my_stack + my_bet_street, 1)
    commit_margin = 0.06 + 0.18 * (commit - 0.40) if commit > 0.40 else 0.0
    if maniac:
        commit_margin *= 0.35

    pen = {"flop": 0.035, "turn": 0.07, "river": 0.13}.get(street, 0.05)
    pen *= (0.5 + 0.85 * bet_frac)                     # bigger bet = more honest
    if raised:
        pen += 0.13                                    # a raise reps far more than a lead
    if maniac:
        pen *= 0.25
    elif station:
        pen *= 0.55
    elif ftb is None:
        pen *= 0.75
    req = eq - pen

    need = breakeven + commit_margin + mw
    big_bet = bet_frac >= 0.66 and not maniac
    call_floor = 0.58 if big_bet else 0.0             # vs a big honest bet, demand showdown value

    # Value — strong even after the range haircut.
    if req >= 0.66:
        big_commit = commit > 0.5 or (raised and bet_frac >= 0.7)
        # A fragile made hand (one pair / two pair) facing a big bet or a raise is
        # NOT a stack-off versus a nuts-only re-raiser: flat if the price is still
        # fine, otherwise fold. Only sets+ (or monster equity) go to war.
        if big_commit and not nutty and req < 0.82:
            if req >= need and bet_frac <= 1.0 and not (raised and bet_frac >= 0.9):
                return {"action": "call"}
            return {"action": "fold"}
        if raised and req < 0.76 and not nutty:
            return {"action": "call"}                  # strong but not nutty vs a raise: flat
        if random.random() < (0.30 if maniac else 0.15):
            return {"action": "call"}
        return {"action": "raise", "amount": _raise_to(state, 0.7 if maniac else 0.82)}

    # Decent showdown value — call when both the price and the bet's honesty allow.
    if req >= max(need + 0.03, call_floor):
        if (not raised and not maniac and ip and commit < 0.33
                and owed <= pot * 0.9 and random.random() < 0.22):
            return {"action": "raise", "amount": _raise_to(state, 0.7)}
        return {"action": "call"}

    # Cheap odds-call with live equity — small bets only, never into a raise.
    if (not raised and not big_bet and eq > breakeven + 0.06 + commit_margin
            and owed <= pot * 0.5 and commit < 0.30):
        return {"action": "call"}

    # Exploit nudge: small semibluff-raise versus a PROVEN folder.
    if (folder and not raised and not maniac and 0.38 <= eq < 0.56
            and owed <= pot * 0.5 and commit < 0.26 and random.random() < 0.22):
        return {"action": "raise", "amount": _raise_to(state, 0.8)}

    return {"action": "fold"}

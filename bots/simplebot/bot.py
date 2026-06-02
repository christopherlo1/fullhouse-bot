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

BOT_NAME = "simpleBot"

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
    n_opp = sum(1 for p in state["players"]
                if not p["is_folded"] and p["bot_id"] != me)
    n_opp = max(1, n_opp)
    pos = _position(state)
    ftb, af = _table_read(state, me)

    # Effective stack vs the pot governs how committed we are.
    total_committed = my_bet + (pot - owed)  # rough; not used directly

    # ===================== PREFLOP =====================
    if street == "preflop":
        return _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
                        n_opp, pos, ftb, af, deadline)

    # ===================== POSTFLOP =====================
    return _postflop(state, hole, board, owed, pot, can_check, my_stack,
                     n_opp, pos, ftb, af, deadline)


# ──────────────────────────────────────────────────────────────────────────
# PREFLOP DECISION
# ──────────────────────────────────────────────────────────────────────────
def _preflop(state, hole, owed, pot, can_check, my_stack, my_bet,
             n_opp, pos, ftb, af, deadline):
    tier = _preflop_tier(state["your_cards"])
    hi, lo, suited, pair = _canon(state["your_cards"])
    facing_raise = state["current_bet"] > BIG_BLIND
    cap = my_stack + my_bet

    # --- L2: facing a big bet / shove. Decide by equity, not by chart. This
    # is the maniac/jam-bot exploit: call wide when the price is right. ---
    if owed > 0 and owed >= 0.25 * my_stack:
        eq = _equity(hole, [], n_opp, deadline, max_iters=1500)
        breakeven = owed / (pot + owed)
        # RISK-SCALED MARGIN (anti-wipe). The more of our stack a call commits,
        # the bigger the equity edge we demand. Stacking off at ~51% is barely
        # +EV but wildly high-variance — that's exactly how a bot gets wiped.
        # We insist on a real edge for big commitments, AND add a cushion per
        # extra opponent (more ways to lose).
        risk = min(owed / max(my_stack, 1), 1.0)        # fraction of stack at risk
        margin = 0.05 + 0.07 * risk + 0.04 * (n_opp - 1)
        # Exception: vs a PROVEN maniac, their shove range is genuinely wide,
        # so calling thin is correct — relax the margin back down.
        if af is not None and af > 1.8:
            margin *= 0.4
        if eq >= breakeven + margin:
            return {"action": "call"}
        # Premiums still call near break-even (they have the equity).
        if tier >= 3 and eq >= breakeven:
            return {"action": "call"}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # Position-based opening requirement.
    if pos < 0.34:   min_open = 2      # early: only strong+
    elif pos < 0.67: min_open = 1      # middle
    else:            min_open = 1      # CO/BTN: open playable+

    # Exploit: if the table folds a lot, widen our steals in late position.
    steal_spot = (not facing_raise) and pos >= 0.6
    overfold = ftb is not None and ftb > 0.55

    if not facing_raise:
        # --- Unopened pot: raise our range. ---
        open_size = _raise_to(state, 1.0)   # ~pot-sized (3x-ish on an open)
        if tier >= min_open:
            return {"action": "raise", "amount": open_size}
        if tier >= 1 and pos >= 0.6:
            return {"action": "raise", "amount": open_size}
        # L4 mixed steal: in the cutoff/button, raise some junk as a bluff,
        # more when the table over-folds. Pure randomisation -> unexploitable.
        if steal_spot:
            steal_p = 0.65 if overfold else 0.30
            if random.random() < steal_p:
                return {"action": "raise", "amount": open_size}
        if can_check:
            return {"action": "check"}
        return {"action": "fold"}

    # --- Facing a (normal-sized) raise: 3-bet / call / fold. ---
    threebet = _raise_to(state, 1.1)
    if tier >= 3:
        return {"action": "raise", "amount": threebet}
    if tier >= 2:
        if owed <= my_stack * 0.12:
            return {"action": "call"}
        return {"action": "fold"}
    if tier >= 1:
        # L4 mixed 3-bet bluff. Widened for pressure: any suited tier-1 hand
        # can 3bet-bluff. Suited aces do it most (the ace blocks their AA/AK),
        # boosted further when the table over-folds. The rest flat in position.
        bluff_3b = (0.18 if (suited and hi == 14) else (0.08 if suited else 0.0))
        if overfold:
            bluff_3b *= 1.8        # only really open up vs proven folders
        if random.random() < bluff_3b:
            return {"action": "raise", "amount": threebet}
        if pos >= 0.6 and owed <= my_stack * 0.07:
            return {"action": "call"}     # set-mine / suited playable cheaply
    if can_check:
        return {"action": "check"}
    return {"action": "fold"}


# ──────────────────────────────────────────────────────────────────────────
# POSTFLOP DECISION
# ──────────────────────────────────────────────────────────────────────────
def _postflop(state, hole, board, owed, pot, can_check, my_stack,
              n_opp, pos, ftb, af, deadline):
    # Scale sims down with more opponents (more eval per iter) and clock.
    budget = 1200 if n_opp == 1 else (700 if n_opp == 2 else 400)
    eq = _equity(hole, board, n_opp, deadline, max_iters=budget)

    breakeven = owed / (pot + owed) if owed > 0 else 0.0
    # Need more equity multiway: each extra opponent is another hand to beat.
    mw = 0.04 * (n_opp - 1)

    maniac = af is not None and af > 2.0
    station = ftb is not None and ftb < 0.30
    overfold = ftb is not None and ftb > 0.50

    # Bluff aggression multiplier — the guardrail. Hammer PROVEN folders, but
    # stay cautious by default: until we have a read (ftb is None) we assume an
    # opponent might be a station, because bluffing one is pure -EV and most
    # naive bots don't fold. Aggression is *earned* by the fold-to-bet read.
    if station:
        bluff_mult = 0.20          # confirmed never-folds: nearly off
    elif overfold:
        bluff_mult = 1.4           # confirmed folder: hammer
    elif ftb is None:
        bluff_mult = 0.55          # no read yet: cautious
    else:
        bluff_mult = 0.85          # normal opponent

    # RANGE-AWARE EQUITY for HERO-CALL spots only. A big turn/river bet is far
    # more honest than a random hand, so we discount our equity before deciding
    # whether to call off a MARGINAL/WEAK hand. Value hands (good/strong) keep
    # full equity so we still extract. This stops us paying off big rivers —
    # the single worst leak of an equity-vs-random bot. Discount shrinks vs
    # maniacs (their bets are air) and small bets (too wide to fear).
    eff_eq = eq
    if owed > 0:
        st_disc = {"flop": 0.03, "turn": 0.06, "river": 0.12}.get(state["street"], 0.0)
        bet_frac = min(owed / pot, 1.5) if pot > 0 else 1.0
        if bet_frac < 0.45:
            st_disc *= 0.3
        disc = st_disc * (0.6 + 0.5 * bet_frac)
        if maniac:
            disc *= 0.25
        elif station:
            disc *= 0.6
        elif ftb is None:
            disc *= 0.7
        eff_eq = eq - disc

    strong   = eq >= 0.72 + mw
    good     = 0.55 + mw <= eq < 0.72 + mw
    marginal = 0.40 + mw <= eq < 0.55 + mw
    # weak: below marginal

    # ---- Checked to us (no bet to call) ----
    if can_check:
        if strong:
            # Value bet; OVERBET vs stations who pay off anything.
            frac = 1.25 if station else 0.8
            return {"action": "raise", "amount": _raise_to(state, frac)}
        if good:
            # Thin value most of the time; occasionally check to trap.
            if random.random() < 0.85:
                return {"action": "raise", "amount": _raise_to(state, 0.6)}
            return {"action": "check"}
        if marginal:
            # Semibluff/stab — more aggressive now, throttled vs stations.
            if random.random() < min(0.40 * bluff_mult, 0.85):
                return {"action": "raise", "amount": _raise_to(state, 0.55)}
            return {"action": "check"}
        # weak: pure bluff. Cranked up, but routed through the guardrail.
        if random.random() < min(0.42 * bluff_mult, 0.85):
            return {"action": "raise", "amount": _raise_to(state, 0.65)}
        return {"action": "check"}

    # ---- Facing a bet ----
    # COMMITMENT GUARD (anti-bust). The more of our stack this call/raise
    # commits, the bigger the range-aware equity edge we demand. Stacking off
    # with a hand that only beats RANDOM is how we get to -10,000. All facing-
    # bet decisions use eff_eq (range-discounted) and add this margin. Relaxed
    # vs a proven maniac (their betting range really is junk).
    commit = owed / max(my_stack + state["your_bet_this_street"], 1)
    commit_margin = 0.0
    if commit > 0.40:
        commit_margin = 0.06 + 0.18 * (commit - 0.40)   # grows toward ~0.17 at full stack
    if maniac:
        commit_margin *= 0.35

    if strong:
        # For a big commitment, make sure RANGE-AWARE equity still backs the
        # stack-off — a hand that's only "strong vs random" can be crushed by a
        # shove range. If not, just call when priced; never raise into it.
        if commit > 0.45 and eff_eq < 0.62:
            if eff_eq > breakeven + commit_margin * 0.5:
                return {"action": "call"}
            return {"action": "fold"}
        frac = 0.7 if maniac else 0.85
        if random.random() < (0.30 if maniac else 0.15):
            return {"action": "call"}
        return {"action": "raise", "amount": _raise_to(state, frac)}

    if good:
        margin = (-0.02 if maniac else 0.05 + mw) + commit_margin
        if eff_eq > breakeven + margin:
            if (not maniac and commit < 0.35 and random.random() < 0.35
                    and owed <= pot * 0.9):
                return {"action": "raise", "amount": _raise_to(state, 0.7)}
            return {"action": "call"}
        return {"action": "fold"}

    if marginal:
        # vs a folder, occasionally semibluff-raise (small commitments only).
        if (overfold and not maniac and owed <= pot * 0.6
                and commit < 0.30 and random.random() < 0.22):
            return {"action": "raise", "amount": _raise_to(state, 0.8)}
        if maniac:
            if eff_eq > breakeven + commit_margin and owed <= pot:
                return {"action": "call"}
            return {"action": "fold"}
        if eff_eq > breakeven + 0.06 + mw + commit_margin and owed <= pot * 0.55:
            return {"action": "call"}
        return {"action": "fold"}

    # weak: fold. Only a cheap, small-commitment call with live equity.
    if eff_eq > breakeven + 0.10 + commit_margin and owed <= pot * 0.3 and commit < 0.25:
        return {"action": "call"}
    return {"action": "fold"}

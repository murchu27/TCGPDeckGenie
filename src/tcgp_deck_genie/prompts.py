"""Prompt templates for the Gemini deck-building pipeline.

Keeping prompts in one place makes it easy to:

* iterate on tone / instructions without touching client code,
* count tokens before sending,
* swap in a different model (or local model) without re-deriving prompts.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

from .models import Card

DECK_SIZE = 20
MAX_COPIES_PER_NAME = 2
MAX_ENERGY_TYPES = 3


# ---------------------------------------------------------------------------
# Rule block - injected verbatim into every Gemini call so the model can never
# "forget" the format constraints of TCG Pocket.
# ---------------------------------------------------------------------------

POCKET_RULES_BLOCK = f"""\
Pokémon TCG Pocket deck construction rules:
- A deck contains exactly {DECK_SIZE} cards.
- At most {MAX_COPIES_PER_NAME} copies of any single card name.
- A deck must include at least one Basic Pokémon (you cannot start a game without
  one in play).
- The Energy Zone uses {MAX_ENERGY_TYPES} or fewer energy types and generates
  exactly ONE energy per turn; energy cards are NOT part of the {DECK_SIZE}-card
  list. Because energy arrives one at a time, high-cost attacks take several turns
  to switch on - watch each attacker's energy curve.
- The bench holds at most 3 Pokémon, so board space is scarce.

EVOLUTION (decks are routinely INVALID here - get this right):
- A Stage 1 can only enter play by evolving its Basic; a Stage 2 only by evolving
  Basic -> Stage 1. There is no other way to put an evolution into play.
- Therefore EVERY evolution card you include MUST have its entire lower chain in
  the SAME deck. Treat an evolution line as one indivisible unit: e.g. to run
  Blastoise ex you must also run Wartortle AND Squirtle. The candidate list is
  pre-loaded with the required pre-evolutions - include them. As a rule of thumb
  run at least as many copies of each lower stage as the stage above it.
- A pre-evolution feeds EVERY card that evolves from it, so count those cards
  TOGETHER, not separately. The total copies of all forms sharing a stage can
  never exceed the copies of their shared pre-evolution available to evolve from
  (and that pre-evolution is itself capped at 2). Concretely: "Primarina ex" and
  plain "Primarina" both evolve from Brionne (<- Popplio), so with at most 2
  Popplio and 2 Brionne you can only ever field 2 Primarina TOTAL. Running, say,
  2 Primarina ex + 1 Primarina is wasteful - the 3rd copy is a dead card that can
  never be played. Pick the split you want across variant final forms, but keep
  their combined count within what the pre-evolution line can actually support.

PRIZES / WIN CONDITION:
- The game is won at 3 points. When the opponent KOs one of your Pokémon they
  score that Pokémon's `kopts` value: 3 (an instant loss) for a Pokémon with
  "Mega" in its name, 2 for any other 4-diamond Pokémon (this includes most ex),
  and 1 for everything else.
- High-`kopts` attackers are powerful and worth running, but conceding 2-3 points
  per KO loses the race fast. Only lean on them if the deck can keep them alive
  (healing) or pull them out of danger (switching / abilities).

ENERGY ECONOMY & RETREAT:
- Retreating the Active Pokémon costs discarding energy equal to its `retreat`
  value, and you only generate ~1 energy/turn, so pivoting is slow and expensive.
  Favour a main attacker with a low retreat cost, or pack switching support if the
  plan needs to reposition.

FOCUS (this matters as much as raw card power):
- Commit to ONE primary win condition built around a single main attacker line
  (occasionally two lines that share the same plan). Pokémon pulling in different
  directions waste your one-energy-per-turn tempo, clog the 3-slot bench, and crowd
  out the Trainers that make a deck consistent.
- Incidental pairwise synergy between otherwise-unrelated attackers is NOT a
  substitute for a coherent gameplan. After locking your core line, spend the
  remaining slots on Trainers (draw, search, switching, healing) and minimal
  support Pokémon - not on yet more standalone attackers.

When designing decks, optimise for:
  1. A coherent, FOCUSED gameplan that every Pokémon advances together.
  2. SYNERGY that serves that plan (ability + attack chains, supporter timing,
     energy acceleration, switching).
  3. STANDALONE UTILITY - some cards (draw/search supporters, switch tools, cheap
     reliable attackers) earn a slot in almost any deck regardless of synergy.
  4. CONSISTENCY - complete evolution lines, enough draw/search, a sane energy
     curve, and a realistic answer to the prize race.
""".strip()


# ---------------------------------------------------------------------------
# Counter block - extra guidance injected only when building against a known
# opponent deck (counter mode).
# ---------------------------------------------------------------------------

COUNTER_BLOCK = """\
You are building a deck to specifically BEAT the opponent deck described in
`opponent`. That summary is pre-analysed for you:
- `weakness_counts` shows how many of their Pokémon are weak to each type.
  Attacking with the most-punishing type means many of their Pokémon take
  bonus damage and fall a hit sooner - this is usually the strongest lever.
- `main_attackers` lists their key threats (with hp, prize value `kopts`,
  retreat, and attack costs/damage). Make sure your deck can survive or trade
  favourably with these, and ideally KO them before they set up.
- `tempo` tells you how slow they are: a deck leaning on Stage 2 lines can be
  raced by faster, lower-to-the-ground attackers; `prize_liabilities` counts
  their 2-3 prize Pokémon, which you want to target to win the prize race.
Counter strategy guidance:
- Exploit their weakness type, out-tempo slow setups, and dodge or punish their
  main attackers' damage windows. Don't concede the prize race: avoid leaning on
  fragile high-`kopts` Pokémon of your own unless you can protect them.
- In `weaknesses`, be honest about how THIS counter deck could still lose to the
  opponent (e.g. if they go fast / hit your weakness first).
""".strip()


# ---------------------------------------------------------------------------
# Shortlist prompt - the cheap "narrow it down" pass.
# ---------------------------------------------------------------------------


def shortlist_system_prompt() -> str:
    return (
        "You are a deck-building assistant for Pokémon TCG Pocket. "
        "You will receive a long list of candidate cards and a brief from the user. "
        "Your job is NOT to build a deck yet - only to pick the most promising "
        "candidates the deck-builder should reason over.\n\n"
        f"{POCKET_RULES_BLOCK}"
    )


def shortlist_user_prompt(
    *,
    user_brief: str,
    energy_type: str,
    candidates: Iterable[Card],
    shortlist_size: int,
) -> str:
    payload = {
        "energy_type": energy_type,
        "user_brief": user_brief,
        "shortlist_size": shortlist_size,
        "candidates": [c.compact_dict() for c in candidates],
    }
    return (
        "Pick the cards most worth considering for the deck described in "
        "`user_brief`. Use the rules block from the system prompt and the "
        "scoring criteria (synergy / standalone utility / consistency).\n\n"
        f"Return exactly {shortlist_size} card IDs (or fewer if the pool is "
        "smaller), choosing a mix of attackers, evolution-line support, and "
        "Trainers as appropriate.\n\n"
        f"Input:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )


# ---------------------------------------------------------------------------
# Deck-building prompt - the reasoning pass.
# ---------------------------------------------------------------------------


def deck_system_prompt(counter: bool = False) -> str:
    base = (
        "You are an expert Pokémon TCG Pocket deck designer. You think carefully "
        "about how cards interact: ability + attack combos, supporter timing, "
        "switch and recovery options, evolution consistency, and the tempo of "
        "a 20-card singleton-ish format.\n\n"
        f"{POCKET_RULES_BLOCK}\n\n"
        "When given a candidate pool, build a 20-card deck and return it in the "
        "JSON format described by the response schema. Always justify your "
        "picks in terms of (a) synergies with other cards in the deck, and "
        "(b) standalone utility - some cards earn their spot from raw power "
        "rather than combo potential, and you should call that out explicitly."
    )
    if counter:
        base += f"\n\n{COUNTER_BLOCK}"
    return base


def deck_user_prompt(
    *,
    user_brief: str,
    energy_type: str,
    candidates: Iterable[Card],
    must_include_ids: Iterable[str] | None = None,
    opponent: dict | None = None,
) -> str:
    payload = {
        "energy_type": energy_type,
        "user_brief": user_brief,
        "must_include_card_ids": list(must_include_ids or []),
        "candidates": [c.compact_dict() for c in candidates],
    }
    if opponent is not None:
        payload["opponent"] = opponent
    intro = (
        "Design a 20-card Pokémon TCG Pocket deck to COUNTER the deck in "
        "`opponent`, using ONLY the cards in `candidates`. "
        if opponent is not None
        else "Design a 20-card Pokémon TCG Pocket deck using ONLY the cards in "
        "`candidates`. "
    )
    return (
        intro + "Honour `must_include_card_ids` if any are present.\n\n"
        "Hard requirements (a deck that breaks these is wrong):\n"
        f"- The `cards` list must sum to EXACTLY {DECK_SIZE} cards. Add up every "
        "entry's `count` and confirm the total before answering; do not under- or "
        "over-fill.\n"
        "- Include the COMPLETE pre-evolution chain for every evolution Pokémon "
        "you use. The candidates already contain the needed pre-evolutions (some "
        "may be off-type Basics like Eevee); add them, with counts at least equal "
        "to the stage above.\n"
        "- Build around ONE focused gameplan. Do not pack in many unrelated "
        "attacker lines - prefer one core line plus Trainers for consistency.\n"
        "- Account for the `retreat` cost, the one-energy-per-turn tempo, and the "
        "prize race (`kopts`): make sure you can actually power up, protect, or "
        "reposition your attackers before they are knocked out.\n\n"
        "Fill `key_synergies` with concrete 1-line notes that name the cards "
        "involved (e.g. 'Misty + Articuno ex: heads-flip energy accel feeds "
        "Blizzard early'). Fill `standalone_value` with cards picked for raw "
        "power / consistency rather than combos. Fill `weaknesses` with the "
        "matchups or situations where this deck will struggle, and how an "
        "opponent might exploit them.\n\n"
        f"Input:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )

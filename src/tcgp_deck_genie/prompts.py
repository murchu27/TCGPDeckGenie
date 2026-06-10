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
- A deck must include at least one Basic Pokémon.
- The deck's Energy Zone uses {MAX_ENERGY_TYPES} or fewer energy types; energy cards
  are NOT part of the {DECK_SIZE}-card list (they're generated automatically each turn).
- Evolution chains must be supported: to play a Stage 1 you need its Basic;
  to play a Stage 2 you need its Basic + Stage 1.
- Pokémon ex give the opponent 2 points when KO'd instead of 1, so they are
  high-reward / high-risk picks.

When designing decks, optimise for:
  1. SYNERGY between cards (combos, ability + attack chains, supporter timing).
  2. STANDALONE UTILITY - some cards (e.g. draw supporters, switch tools, high-HP
     attackers with cheap costs) are valuable in almost any deck regardless of
     synergy.
  3. CONSISTENCY - too many high-cost Stage-2 lines without supporting draw or
     search is unreliable.
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


def deck_system_prompt() -> str:
    return (
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


def deck_user_prompt(
    *,
    user_brief: str,
    energy_type: str,
    candidates: Iterable[Card],
    must_include_ids: Iterable[str] | None = None,
) -> str:
    payload = {
        "energy_type": energy_type,
        "user_brief": user_brief,
        "must_include_card_ids": list(must_include_ids or []),
        "candidates": [c.compact_dict() for c in candidates],
    }
    return (
        "Design a 20-card Pokémon TCG Pocket deck using ONLY the cards in "
        "`candidates`. Honour `must_include_card_ids` if any are present.\n\n"
        "Fill `key_synergies` with concrete 1-line notes that name the cards "
        "involved (e.g. 'Misty + Articuno ex: heads-flip energy accel feeds "
        "Blizzard early'). Fill `standalone_value` with cards picked for raw "
        "power / consistency rather than combos. Fill `weaknesses` with the "
        "matchups or situations where this deck will struggle, and how an "
        "opponent might exploit them.\n\n"
        f"Input:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )

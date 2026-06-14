"""Local, deterministic filtering over the cached card corpus.

This is the cost-saving stage that runs before we touch the LLM. Every card we
exclude here is one we don't have to pay tokens to describe later.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .models import ENERGY_TYPES, Card


@dataclass
class SearchFilter:
    """Filter spec for narrowing the corpus before LLM reasoning."""

    energy_type: str | None = None
    """Limit Pokémon to this single type. Trainers are always kept."""

    set_ids: set[str] | None = None
    """Limit to cards in these specific sets."""

    keywords: list[str] = field(default_factory=list)
    """Substrings that must appear in card name, ability text, or attack text."""

    category: str | None = None
    """If set, keep only 'Pokemon' or 'Trainer' cards."""

    include_ex: bool = True
    """If False, exclude Pokémon ex (the high-prize variants)."""

    max_retreat: int | None = None
    """If set, exclude Pokémon whose retreat cost exceeds this."""

    def normalised(self) -> SearchFilter:
        et = self.energy_type
        if et is not None:
            et = et.capitalize()
            if et not in ENERGY_TYPES:
                raise ValueError(
                    f"Unknown energy type {self.energy_type!r}; allowed={list(ENERGY_TYPES)}"
                )
        return SearchFilter(
            energy_type=et,
            set_ids=set(self.set_ids) if self.set_ids else None,
            keywords=[k.lower() for k in self.keywords if k],
            category=self.category,
            include_ex=self.include_ex,
            max_retreat=self.max_retreat,
        )


def apply_filter(cards: Iterable[Card], spec: SearchFilter) -> list[Card]:
    """Return the subset of ``cards`` matching ``spec``."""
    norm = spec.normalised()
    out: list[Card] = []
    for card in cards:
        if not _match(card, norm):
            continue
        out.append(card)
    return out


def _match(card: Card, spec: SearchFilter) -> bool:
    if spec.set_ids and card.set_id not in spec.set_ids:
        return False
    if spec.category and card.category != spec.category:
        return False

    if card.is_pokemon:
        if spec.energy_type and (card.primary_type() != spec.energy_type):
            # Trainers and other-typed Pokémon fail the energy filter.
            return False
        if not spec.include_ex and card.is_ex:
            return False
        if spec.max_retreat is not None and (card.retreat or 0) > spec.max_retreat:
            return False
    elif card.is_trainer:
        # If the user asked for a specific category and we're not it, drop.
        if spec.category and spec.category != "Trainer":
            return False
    else:
        return False

    if spec.keywords:
        hay = _haystack(card)
        if not all(k in hay for k in spec.keywords):
            return False
    return True


def _haystack(card: Card) -> str:
    """Concatenated lowercase text we search keywords against."""
    parts: list[str] = [card.name.lower()]
    if card.effect:
        parts.append(card.effect.lower())
    for a in card.attacks:
        parts.append(a.name.lower())
        if a.effect:
            parts.append(a.effect.lower())
    for ab in card.abilities:
        parts.append(ab.name.lower())
        if ab.effect:
            parts.append(ab.effect.lower())
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Lightweight scoring used to rank candidates before we hand them to the LLM.
# Higher scores are kept; ties broken by id for stability.
# ---------------------------------------------------------------------------


_DRAW_PATTERNS = (
    re.compile(r"\bdraw\b", re.I),
    re.compile(r"\bsearch your deck\b", re.I),
    re.compile(r"\bput .* into your hand\b", re.I),
)

_HEAL_PATTERNS = (re.compile(r"\bheal\b", re.I),)

_SWITCH_PATTERNS = (
    re.compile(r"\bswitch\b", re.I),
    re.compile(r"\bretreat\b", re.I),
)


def candidate_score(card: Card) -> float:
    """A rough, transparent score that biases toward useful cards.

    This is intentionally simple and cheap; the LLM does the real reasoning.
    The score is only used to truncate the candidate list when it would
    otherwise blow past our prompt-size budget.
    """
    if card.is_trainer:
        # Trainers with draw, search, healing, or switch effects tend to be
        # universally useful.
        score = 5.0
        text = (card.effect or "").lower()
        score += sum(2.0 for p in _DRAW_PATTERNS if p.search(text))
        score += sum(1.5 for p in _HEAL_PATTERNS if p.search(text))
        score += sum(1.5 for p in _SWITCH_PATTERNS if p.search(text))
        return score

    # Pokémon scoring:
    #   * High-HP / ex Pokémon tend to be main attackers.
    #   * Basic Pokémon are mandatory and cheap to play.
    #   * Pokémon with abilities tend to enable synergies.
    score = (card.hp or 0) / 40.0
    if card.is_basic:
        score += 1.5
    if card.is_ex:
        score += 2.0
    if card.abilities:
        score += 1.5 * len(card.abilities)
    if card.attacks:
        best_dmg = max((a.parsed_damage for a in card.attacks), default=0)
        score += best_dmg / 50.0
    return score


def top_candidates(cards: list[Card], limit: int) -> list[Card]:
    """Return the highest-scoring ``limit`` cards, ordered for stable prompts."""
    if limit <= 0 or len(cards) <= limit:
        return sorted(cards, key=lambda c: c.id)
    scored = sorted(cards, key=lambda c: (-candidate_score(c), c.id))
    kept = scored[:limit]
    return sorted(kept, key=lambda c: c.id)

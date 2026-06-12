"""Orchestration: candidate selection → optional shortlist → reasoned deck.

The two-stage funnel is the key cost-saving design:

1. **Local filter (free):** Apply ``SearchFilter`` against the cached corpus.
   Even a permissive filter (one energy type only) typically takes the corpus
   from ~2000 cards down to ~250.
2. **Cheap shortlist (~1k tokens, Flash-Lite):** Ask a small model to pick the
   most promising N candidates. Skippable for tiny candidate pools.
3. **Reasoned synthesis (~few k tokens, Flash with thinking):** Hand the
   shortlisted cards to the reasoning model and get back a structured
   ``DeckPlan`` with explicit notes on synergy, standalone value, and
   weaknesses.

Before the reasoning step we re-inject the full pre-evolution chain of every
evolution card (``_with_evolution_support``) so the model can always build a
legal line - even when an off-type Basic (e.g. Colorless Eevee under a Water
Vaporeon) was stripped out by the energy filter.

The validator at the end enforces the format constraints we cannot rely on the
LLM to always honour (exact 20 cards, ≤2 copies, energy consistency, evolution
support).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .gemini_client import GeminiClient
from .models import ENERGY_TYPES, Card, DeckPlan
from .prompts import (
    DECK_SIZE,
    MAX_COPIES_PER_NAME,
    MAX_ENERGY_TYPES,
    deck_system_prompt,
    deck_user_prompt,
    shortlist_system_prompt,
    shortlist_user_prompt,
)
from .search import SearchFilter, apply_filter, top_candidates

logger = logging.getLogger(__name__)


# Empirically these keep prompts compact and well within the free-tier 250k
# tokens-per-minute limit.
DEFAULT_PRESHORTLIST_CAP = 120
DEFAULT_SHORTLIST_SIZE = 40


@dataclass
class BuildOptions:
    """Per-run knobs for ``DeckBuilder.build``."""

    energy_type: str
    user_brief: str = "Build a strong, fun deck."
    set_ids: set[str] | None = None
    must_include_card_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    include_ex: bool = True
    max_retreat: int | None = None
    preshortlist_cap: int = DEFAULT_PRESHORTLIST_CAP
    shortlist_size: int = DEFAULT_SHORTLIST_SIZE
    use_shortlist: bool = True
    thinking_budget: int | None = None


@dataclass
class BuildResult:
    """Full output of a build: the deck and the cards it references."""

    deck: DeckPlan
    cards_used: list[Card]
    candidate_pool_size: int
    shortlist_size: int | None
    validation_warnings: list[str] = field(default_factory=list)


class DeckBuildError(RuntimeError):
    """Raised when we cannot produce a valid deck."""


class DeckBuilder:
    """Glue between the local corpus and the LLM."""

    def __init__(self, corpus: list[Card], gemini: GeminiClient) -> None:
        self._corpus = corpus
        self._by_id = {c.id: c for c in corpus}
        self._gemini = gemini

    # -- public API ---------------------------------------------------------------

    def build(self, options: BuildOptions) -> BuildResult:
        energy = self._normalise_energy(options.energy_type)

        candidate_pool = self._build_candidate_pool(options, energy)
        if not candidate_pool:
            raise DeckBuildError(
                f"No cards survived the filter (energy={energy!r}, sets={options.set_ids}, "
                f"keywords={options.keywords}). Try relaxing the filter."
            )

        shortlisted_size: int | None = None
        candidates_for_reasoning: list[Card] = candidate_pool
        if (
            options.use_shortlist
            and len(candidate_pool) > options.shortlist_size
            and self._gemini.config.shortlist_model
        ):
            shortlisted_ids = self._shortlist(options, energy, candidate_pool)
            shortlisted_cards = self._validate_ids_are_in_pool(
                shortlisted_ids, candidate_pool
            )
            # Always include any must-include cards even if the shortlist drops them.
            shortlisted_cards = _merge_must_include(
                shortlisted_cards, options.must_include_card_ids, self._by_id
            )
            shortlisted_size = len(shortlisted_cards)
            candidates_for_reasoning = shortlisted_cards

        # Guarantee the reasoning model can build complete evolution lines: pull
        # in the full pre-evolution chain (from the whole corpus) of every
        # evolution card it will see. This is essential because off-type
        # pre-evolutions - e.g. the Colorless Basic Eevee under a Water Vaporeon -
        # are stripped out by the energy filter and would otherwise never reach
        # the model, making a valid line impossible.
        candidates_for_reasoning = _with_evolution_support(
            candidates_for_reasoning, self._corpus
        )

        deck = self._build_deck(options, energy, candidates_for_reasoning)
        used_cards = [self._by_id[e.card_id] for e in deck.cards if e.card_id in self._by_id]
        warnings = validate_deck(deck, self._by_id)

        return BuildResult(
            deck=deck,
            cards_used=used_cards,
            candidate_pool_size=len(candidate_pool),
            shortlist_size=shortlisted_size,
            validation_warnings=warnings,
        )

    # -- stages -------------------------------------------------------------------

    def _build_candidate_pool(self, options: BuildOptions, energy: str) -> list[Card]:
        spec = SearchFilter(
            energy_type=energy,
            set_ids=options.set_ids,
            keywords=options.keywords,
            include_ex=options.include_ex,
            max_retreat=options.max_retreat,
        )
        # Pokémon constrained to energy type, plus all trainers (they're typeless).
        mons = apply_filter(self._corpus, spec)
        trainers_spec = SearchFilter(
            set_ids=options.set_ids,
            keywords=options.keywords,
            category="Trainer",
        )
        trainers = apply_filter(self._corpus, trainers_spec)

        combined = _dedupe_by_id(mons + trainers)
        # Always keep must-includes regardless of filter (they may be opposite-typed
        # tech inclusions the user explicitly wants).
        combined = _merge_must_include(combined, options.must_include_card_ids, self._by_id)

        # Cap before we hand to the LLM so prompt size stays predictable.
        return top_candidates(combined, options.preshortlist_cap)

    def _shortlist(
        self, options: BuildOptions, energy: str, candidate_pool: list[Card]
    ) -> list[str]:
        logger.info(
            "Shortlisting %d candidates down to %d via %s",
            len(candidate_pool),
            options.shortlist_size,
            self._gemini.config.shortlist_model,
        )
        resp = self._gemini.shortlist(
            system_prompt=shortlist_system_prompt(),
            user_prompt=shortlist_user_prompt(
                user_brief=options.user_brief,
                energy_type=energy,
                candidates=candidate_pool,
                shortlist_size=options.shortlist_size,
            ),
        )
        return resp.card_ids

    def _build_deck(
        self,
        options: BuildOptions,
        energy: str,
        candidates: list[Card],
    ) -> DeckPlan:
        logger.info(
            "Building deck from %d candidates via %s",
            len(candidates),
            self._gemini.config.reasoning_model,
        )
        return self._gemini.build_deck(
            system_prompt=deck_system_prompt(),
            user_prompt=deck_user_prompt(
                user_brief=options.user_brief,
                energy_type=energy,
                candidates=candidates,
                must_include_ids=options.must_include_card_ids,
            ),
            thinking_budget=options.thinking_budget,
        )

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _normalise_energy(et: str) -> str:
        e = et.strip().capitalize()
        if e not in ENERGY_TYPES:
            raise DeckBuildError(
                f"Unknown energy type {et!r}; allowed: {', '.join(ENERGY_TYPES)}"
            )
        return e

    @staticmethod
    def _validate_ids_are_in_pool(ids: list[str], pool: list[Card]) -> list[Card]:
        pool_by_id = {c.id: c for c in pool}
        out: list[Card] = []
        for cid in ids:
            if cid in pool_by_id:
                out.append(pool_by_id[cid])
            else:
                logger.warning("Shortlist returned card id %s that's not in the pool; ignoring.", cid)
        return out


def _dedupe_by_id(cards: list[Card]) -> list[Card]:
    seen: dict[str, Card] = {}
    for c in cards:
        seen.setdefault(c.id, c)
    return list(seen.values())


def _merge_must_include(
    cards: list[Card], must_ids: list[str], by_id: dict[str, Card]
) -> list[Card]:
    if not must_ids:
        return cards
    have = {c.id for c in cards}
    for cid in must_ids:
        if cid in by_id and cid not in have:
            cards.append(by_id[cid])
    return cards


def _index_pokemon_by_name(corpus: list[Card]) -> dict[str, Card]:
    """Map each Pokémon name to a single canonical printing (lowest card id)."""
    by_name: dict[str, Card] = {}
    for c in corpus:
        if not c.is_pokemon:
            continue
        current = by_name.get(c.name)
        if current is None or c.id < current.id:
            by_name[c.name] = c
    return by_name


def _with_evolution_support(cards: list[Card], corpus: list[Card]) -> list[Card]:
    """Return ``cards`` plus the pre-evolution chain of every evolution card.

    For each Pokémon that evolves from something, we walk ``evolve_from`` down to
    the Basic, pulling the relevant printings out of ``corpus`` (by name) and
    appending any that are not already present. The walk continues through each
    newly-added pre-evolution so multi-stage lines (Stage 2 -> Stage 1 -> Basic)
    are fully resolved. Pre-evolutions are looked up regardless of type, so
    off-type Basics that the energy filter removed are restored here.
    """
    by_name = _index_pokemon_by_name(corpus)
    have_ids = {c.id for c in cards}
    result = list(cards)
    queue = list(cards)
    while queue:
        card = queue.pop()
        if not (card.is_pokemon and card.evolve_from):
            continue
        pre = by_name.get(card.evolve_from)
        if pre is not None and pre.id not in have_ids:
            have_ids.add(pre.id)
            result.append(pre)
            queue.append(pre)
    return result


# ---------------------------------------------------------------------------
# Deck validation - the safety net for "the LLM mostly got it right" cases.
# ---------------------------------------------------------------------------


def validate_deck(deck: DeckPlan, by_id: dict[str, Card]) -> list[str]:
    """Return a list of human-readable warnings about the deck.

    These are warnings rather than errors because Gemini occasionally produces
    a deck that's interesting but technically off-spec (e.g., 19 cards or a
    missing pre-evolution); the CLI surfaces them so the user can decide.
    """
    warnings: list[str] = []
    total = sum(e.count for e in deck.cards)
    if total != DECK_SIZE:
        warnings.append(f"Deck has {total} cards; TCG Pocket decks must have exactly {DECK_SIZE}.")

    by_name: dict[str, int] = {}
    for entry in deck.cards:
        card = by_id.get(entry.card_id)
        if card is None:
            warnings.append(f"Unknown card id in deck: {entry.card_id}")
            continue
        if entry.count > MAX_COPIES_PER_NAME:
            warnings.append(
                f"{card.name}: count={entry.count} exceeds max {MAX_COPIES_PER_NAME} per name."
            )
        by_name[card.name] = by_name.get(card.name, 0) + entry.count

    for name, cnt in by_name.items():
        if cnt > MAX_COPIES_PER_NAME:
            warnings.append(
                f"{name} appears {cnt} times across entries; max allowed is {MAX_COPIES_PER_NAME}."
            )

    if len(deck.energy_types) > MAX_ENERGY_TYPES:
        warnings.append(
            f"Deck declares {len(deck.energy_types)} energy types; TCG Pocket allows up to "
            f"{MAX_ENERGY_TYPES}."
        )

    cards_in_deck = [by_id.get(e.card_id) for e in deck.cards]
    pokemon_cards = [c for c in cards_in_deck if c and c.is_pokemon]
    if not any(c.is_basic for c in pokemon_cards):
        warnings.append("Deck has no Basic Pokémon (every deck needs at least one to start).")

    # Energy-coverage warning: any Pokémon with an attack cost that requires an
    # energy type not in the deck's declared energy pool is hard to power up.
    declared_energy = set(deck.energy_types) | {"Colorless"}
    for c in pokemon_cards:
        for atk in c.attacks:
            needed = {e for e in atk.cost if e and e != "Colorless"}
            missing = needed - declared_energy
            if missing:
                warnings.append(
                    f"{c.name}'s attack '{atk.name}' needs {sorted(missing)} energy not in "
                    f"deck's declared types {sorted(deck.energy_types)}."
                )
                break  # one warning per card is enough

    # Evolution-line warning: every Stage 1 / Stage 2 needs its pre-evolution
    # represented somewhere in the deck.
    names_in_deck = {c.name for c in cards_in_deck if c is not None}
    for c in pokemon_cards:
        if c.evolve_from and c.evolve_from not in names_in_deck:
            warnings.append(
                f"{c.name} needs its pre-evolution {c.evolve_from!r} in the deck."
            )

    # Evolution-bottleneck warning: a pre-evolution feeds EVERY form that evolves
    # from it, so those forms compete for the same limited pool. The combined
    # copies of all cards sharing a pre-evolution can never exceed the copies of
    # that pre-evolution in the deck - any surplus can never be put into play.
    # e.g. 2 "Primarina ex" + 1 "Primarina" both evolve from Brionne; with only
    # 2 Brionne, the 3rd Primarina is a dead card.
    evolvers_by_pre: dict[str, list[str]] = {}
    for c in pokemon_cards:
        if c.evolve_from:
            evolvers_by_pre.setdefault(c.evolve_from, [])
            if c.name not in evolvers_by_pre[c.evolve_from]:
                evolvers_by_pre[c.evolve_from].append(c.name)

    for pre_name, form_names in evolvers_by_pre.items():
        pre_count = by_name.get(pre_name, 0)
        if pre_count == 0:
            # Pre-evolution missing entirely - already covered by the line warning.
            continue
        demand = sum(by_name.get(form, 0) for form in form_names)
        if demand > pre_count:
            forms_desc = ", ".join(sorted(form_names))
            warnings.append(
                f"{forms_desc} total {demand} copies but only evolve from "
                f"{pre_count} {pre_name}; at most {pre_count} can ever be played "
                f"({demand - pre_count} dead). Trim the evolved forms or add more "
                f"{pre_name}."
            )

    return warnings

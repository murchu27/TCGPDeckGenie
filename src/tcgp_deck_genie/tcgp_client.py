"""Fetch and normalise Pokémon TCG Pocket cards from TCGdex.

Why this module is shaped the way it is:

* TCGdex's catalogue rarely changes (only when a new TCGP set ships), so we
  fetch once into the on-disk cache (see ``cache.py``) and never hit the API
  again during routine deck-building.
* The SDK gives us rich, language-aware objects, but they include a lot of
  fields we do not use for deck design (image URLs, illustrator, pricing,
  variant metadata). We immediately project to the slim ``Card`` model so
  prompts and caches stay small.
* Many cards exist in multiple printings (mechanically-identical rarities and
  art rares). For deck-building, all that matters is the gameplay text, so we
  deduplicate by ``(name, set_id)``, preferring the lowest-numbered printing
  (which is consistently the "base" version).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tcgdexsdk import TCGdex

from .models import Ability, Attack, Card

logger = logging.getLogger(__name__)

TCGP_SERIES_ID = "tcgp"

# Reasonable default for the TCGdex public API; it's CDN-fronted but we don't
# want to look like an abusive client.
DEFAULT_FETCH_CONCURRENCY = 12

# Rarity tiers we treat as "out of reach" for a typical TCG Pocket player.
# TCGdex returns rarity as a human-readable string like "Four Diamond", "Two
# Star", "One Shiny", "Crown Rare". We match by substring (case-insensitive),
# which is enough to catch every rare tier while leaving Diamonds, Promos, and
# unrated cards alone.
RARE_RARITY_KEYWORDS: tuple[str, ...] = ("Star", "Shiny", "Crown")


@dataclass
class FetchProgress:
    """Lightweight progress payload for callers (e.g. CLI spinners)."""

    set_id: str
    completed: int
    total: int


ProgressCallback = Callable[[FetchProgress], None] | None


class TCGPClient:
    """High-level TCGdex client constrained to the TCG Pocket series."""

    def __init__(self, language: str = "en", concurrency: int = DEFAULT_FETCH_CONCURRENCY) -> None:
        self._sdk = TCGdex(language)
        self._concurrency = max(1, concurrency)

    # -- discovery -----------------------------------------------------------------

    def list_set_ids(self) -> list[str]:
        """Return every set id in the TCG Pocket series."""
        serie = self._sdk.serie.getSync(TCGP_SERIES_ID)
        if serie is None:
            raise RuntimeError(f"TCGdex returned no series for id={TCGP_SERIES_ID!r}")
        return [s.id for s in serie.sets]

    def list_card_ids_in_set(self, set_id: str) -> list[str]:
        """Return every card id that lives in ``set_id``."""
        set_obj = self._sdk.set.getSync(set_id)
        if set_obj is None:
            raise RuntimeError(f"TCGdex returned no set for id={set_id!r}")
        return [c.id for c in set_obj.cards]

    # -- fetch ---------------------------------------------------------------------

    def fetch_cards(
        self,
        set_ids: Iterable[str] | None = None,
        progress: ProgressCallback = None,
        exclude_rares: bool = True,
    ) -> list[Card]:
        """Fetch and normalise every card in the given sets (defaults to all TCGP sets).

        When ``exclude_rares`` is true (the default), printings whose rarity
        contains "Star", "Shiny", or "Crown" are dropped before the dedupe step.
        These tiers are extremely hard to obtain in TCG Pocket, so a deck that
        relies on one would be frustrating to actually build in-game. Most of
        these rares are reprints of cards that also exist at Diamond rarity and
        would already be dropped by the dedupe pass, but some unique rares
        otherwise slip through; the explicit filter catches those too.

        Set ``exclude_rares=False`` if the caller knows they have access to the
        rares in question and wants them considered for decks.
        """
        set_ids = list(set_ids) if set_ids is not None else self.list_set_ids()

        all_cards: list[Card] = []
        for set_id in set_ids:
            card_ids = self.list_card_ids_in_set(set_id)
            fetched = self._fetch_card_details_concurrent(set_id, card_ids, progress)
            if exclude_rares:
                before = len(fetched)
                fetched = [c for c in fetched if not _is_rare(c)]
                logger.debug(
                    "Set %s: filtered out %d rare printing(s) (Star/Shiny/Crown).",
                    set_id,
                    before - len(fetched),
                )
            all_cards.extend(_dedupe(fetched))
        return all_cards

    def _fetch_card_details_concurrent(
        self,
        set_id: str,
        card_ids: list[str],
        progress: ProgressCallback,
    ) -> list[Card]:
        results: list[Card] = []
        total = len(card_ids)
        completed = 0
        if progress is not None:
            progress(FetchProgress(set_id=set_id, completed=0, total=total))

        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            future_to_id = {pool.submit(self._fetch_single, cid): cid for cid in card_ids}
            for future in as_completed(future_to_id):
                card_id = future_to_id[future]
                try:
                    card = future.result()
                except Exception as exc:  # pragma: no cover - network failures
                    logger.warning("Failed to fetch %s: %s", card_id, exc)
                    card = None
                completed += 1
                if card is not None:
                    results.append(card)
                if progress is not None:
                    progress(FetchProgress(set_id=set_id, completed=completed, total=total))
        return results

    def _fetch_single(self, card_id: str) -> Card | None:
        raw = self._sdk.card.getSync(card_id)
        if raw is None:
            return None
        return _to_card(raw)


def _to_card(raw) -> Card | None:
    """Convert a TCGdex SDK Card into our slim ``Card`` model.

    The SDK uses camelCase attributes mirroring the REST API. We pull them via
    ``getattr`` so the module also works if the SDK ever swaps in dataclasses
    vs Pydantic models in a minor release.
    """
    category = getattr(raw, "category", None)
    # The SDK omits non-Pokémon-non-Trainer categories from TCG Pocket entirely
    # (there are no basic Energy cards in TCGP), but we still guard the cast.
    if category not in {"Pokemon", "Trainer", "Energy"}:
        return None

    set_obj = getattr(raw, "set", None)
    set_id = getattr(set_obj, "id", None) or ""

    attacks = []
    for a in getattr(raw, "attacks", None) or []:
        attacks.append(
            Attack(
                name=getattr(a, "name", "") or "",
                cost=list(getattr(a, "cost", None) or []),
                damage=_to_str_or_none(getattr(a, "damage", None)),
                effect=_to_str_or_none(getattr(a, "effect", None)),
            )
        )

    abilities = []
    for ab in getattr(raw, "abilities", None) or []:
        abilities.append(
            Ability(
                name=getattr(ab, "name", "") or "",
                type=_to_str_or_none(getattr(ab, "type", None)),
                effect=_to_str_or_none(getattr(ab, "effect", None)),
            )
        )

    weaknesses_raw = getattr(raw, "weaknesses", None) or []
    weaknesses: list[dict] = []
    for w in weaknesses_raw:
        if isinstance(w, dict):
            weaknesses.append(w)
        else:
            weaknesses.append(
                {
                    "type": getattr(w, "type", None),
                    "value": getattr(w, "value", None),
                }
            )

    return Card(
        id=getattr(raw, "id", "") or "",
        name=getattr(raw, "name", "") or "",
        set_id=set_id,
        category=category,
        hp=_to_int_or_none(getattr(raw, "hp", None)),
        types=list(getattr(raw, "types", None) or []),
        stage=_to_str_or_none(getattr(raw, "stage", None)),
        evolve_from=_to_str_or_none(getattr(raw, "evolveFrom", None)),
        suffix=_to_str_or_none(getattr(raw, "suffix", None)),
        attacks=attacks,
        abilities=abilities,
        weaknesses=weaknesses,
        retreat=_to_int_or_none(getattr(raw, "retreat", None)),
        trainer_type=_to_str_or_none(getattr(raw, "trainerType", None)),
        effect=_to_str_or_none(getattr(raw, "effect", None)),
        rarity=_to_str_or_none(getattr(raw, "rarity", None)),
    )


def _to_int_or_none(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_str_or_none(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _is_rare(card: Card) -> bool:
    """True if this card's rarity contains a high-rarity keyword.

    TCG Pocket's rarity tiers are (roughly, ordered cheapest → rarest):
    One-/Two-/Three-/Four-Diamond → One-/Two-/Three-Star → Shiny → Crown Rare.
    We treat everything from Star up as "rare". Cards with no rarity string
    (typically Promos that haven't been categorised yet) are kept by default
    so we don't accidentally drop obtainable promos.
    """
    if not card.rarity:
        return False
    rarity_lower = card.rarity.lower()
    return any(kw.lower() in rarity_lower for kw in RARE_RARITY_KEYWORDS)


def _dedupe(cards: list[Card]) -> list[Card]:
    """Collapse mechanically-identical reprints within a single set.

    Two cards with the same ``(name, set_id)`` are treated as duplicates; we
    keep the one with the lowest local id (everything after a set's "official"
    count is an art rare / full art reprint with identical gameplay text).
    """
    best: dict[tuple[str, str], Card] = {}
    for c in cards:
        key = (c.set_id, c.name)
        current = best.get(key)
        if current is None or _sort_key(c) < _sort_key(current):
            best[key] = c
    return list(best.values())


def _sort_key(card: Card) -> tuple:
    """Stable sort key for picking the canonical reprint of a card."""
    local = card.id.split("-", 1)[-1] if "-" in card.id else card.id
    try:
        local_num = int(local)
    except ValueError:
        local_num = 10**9
    return (local_num, card.id)

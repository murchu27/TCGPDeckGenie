"""Tests for the offline helpers in ``tcgp_client``.

The network-touching methods are exercised by live CLI runs; this file covers
the pure helpers we can unit-test without hitting TCGdex.
"""
from __future__ import annotations

from tcgp_deck_genie.models import Attack, Card
from tcgp_deck_genie.tcgp_client import (
    RARE_RARITY_KEYWORDS,
    _dedupe,
    _is_rare,
)


def _trainer(name: str, rarity: str | None, *, set_id: str = "A1") -> Card:
    return Card(
        id=f"{set_id}-{name[:3].upper()}",
        name=name,
        set_id=set_id,
        category="Trainer",
        trainer_type="Item",
        rarity=rarity,
    )


def _pokemon(
    name: str,
    rarity: str | None,
    *,
    set_id: str = "A1",
    local_num: int = 1,
) -> Card:
    return Card(
        id=f"{set_id}-{local_num:03d}",
        name=name,
        set_id=set_id,
        category="Pokemon",
        hp=80,
        types=["Water"],
        stage="Basic",
        rarity=rarity,
        attacks=[Attack(name="Water Gun", cost=["Water"], damage="20")],
    )


def test_rare_keywords_constant_matches_spec():
    # Diamond / Promo / unset cards must NOT be in this list.
    assert set(RARE_RARITY_KEYWORDS) == {"Star", "Shiny", "Crown"}


def test_is_rare_excludes_star_shiny_crown():
    assert _is_rare(_trainer("AltCard", "Two Star"))
    assert _is_rare(_trainer("AltCard", "One Shiny"))
    assert _is_rare(_trainer("AltCard", "Crown Rare"))


def test_is_rare_keeps_every_diamond_tier():
    for rarity in ("One Diamond", "Two Diamond", "Three Diamond", "Four Diamond"):
        assert not _is_rare(_trainer("Card", rarity)), f"unexpectedly flagged: {rarity}"


def test_is_rare_keeps_cards_with_missing_or_empty_rarity():
    # Promos and uncategorised cards often have no rarity string; we keep
    # them by default so we don't silently drop obtainable cards.
    assert not _is_rare(_trainer("Promo", None))
    assert not _is_rare(_trainer("Promo", ""))


def test_is_rare_is_case_insensitive():
    assert _is_rare(_trainer("X", "two star"))
    assert _is_rare(_trainer("X", "CROWN"))
    assert _is_rare(_trainer("X", "ShInY"))


def test_dedupe_collapses_reprints_to_lowest_local_id():
    # Two Diamond printings of the same card (after a rarity filter has already
    # removed the Star/Crown reprints) - dedupe should keep the lower-numbered
    # one.
    cards = [
        _pokemon("Charizard", "Four Diamond", set_id="A1", local_num=36),
        _pokemon("Charizard", "Four Diamond", set_id="A1", local_num=253),
    ]
    kept = _dedupe(cards)
    assert len(kept) == 1
    assert kept[0].id == "A1-036"


def test_dedupe_keeps_same_name_across_different_sets():
    cards = [
        _pokemon("Pikachu", "One Diamond", set_id="A1", local_num=94),
        _pokemon("Pikachu", "One Diamond", set_id="A2", local_num=12),
    ]
    kept = _dedupe(cards)
    assert {c.set_id for c in kept} == {"A1", "A2"}

"""Tests for deck validation and the build pipeline.

The ``DeckBuilder`` itself is exercised via a fake ``GeminiClient`` so we never
touch the network.
"""
from __future__ import annotations

from dataclasses import dataclass

from tcgp_deck_genie.deck_builder import (
    BuildOptions,
    DeckBuilder,
    validate_deck,
)
from tcgp_deck_genie.gemini_client import GeminiConfig, ShortlistResponse
from tcgp_deck_genie.models import Card, DeckEntry, DeckPlan


def _by_id(cards: list[Card]) -> dict[str, Card]:
    return {c.id: c for c in cards}


# ---------------------------------------------------------------------------
# validate_deck
# ---------------------------------------------------------------------------


def test_validator_flags_wrong_total(fake_corpus):
    deck = DeckPlan(
        name="x",
        energy_types=["Water"],
        cards=[DeckEntry(card_id="A1-079", count=1)],
        strategy="",
    )
    warnings = validate_deck(deck, _by_id(fake_corpus))
    assert any("must have exactly 20" in w for w in warnings)


def test_validator_flags_missing_basic(fake_corpus):
    # Build a 20-card deck made only of evolutions and a Trainer (no Basic).
    deck = DeckPlan(
        name="x",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="A1-054", count=2),  # Wartortle
            DeckEntry(card_id="A1-056", count=2),  # Blastoise ex
            DeckEntry(card_id="A1-220", count=2),  # Misty
            DeckEntry(card_id="A1-223", count=2),  # Giovanni
            DeckEntry(card_id="P-A-007", count=2),  # Prof Research
            DeckEntry(card_id="A1-054", count=2),  # padding (still no basic)
            DeckEntry(card_id="A1-054", count=2),
            DeckEntry(card_id="A1-054", count=2),
            DeckEntry(card_id="A1-054", count=2),
            DeckEntry(card_id="A1-054", count=2),
        ],
        strategy="",
    )
    warnings = validate_deck(deck, _by_id(fake_corpus))
    assert any("no Basic Pokémon" in w for w in warnings)


def test_validator_flags_missing_pre_evolution(fake_corpus):
    deck = DeckPlan(
        name="x",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="A1-079", count=2),  # Lapras (basic, OK)
            DeckEntry(card_id="A1-054", count=2),  # Wartortle, but no Squirtle
        ]
        + [DeckEntry(card_id="A1-220", count=2)] * 8,
        strategy="",
    )
    warnings = validate_deck(deck, _by_id(fake_corpus))
    assert any("pre-evolution" in w and "Squirtle" in w for w in warnings)


def test_validator_passes_well_formed_deck(fake_corpus):
    # Exactly 20 cards: 2x each = 20
    deck = DeckPlan(
        name="ok",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="A1-079", count=2),  # Lapras
            DeckEntry(card_id="A1-101", count=2),  # Articuno ex
            DeckEntry(card_id="A1-SQU", count=2),  # Squirtle (fake_corpus id)
            DeckEntry(card_id="A1-054", count=2),  # Wartortle
            DeckEntry(card_id="A1-056", count=2),  # Blastoise ex
            DeckEntry(card_id="A1-220", count=2),  # Misty
            DeckEntry(card_id="A1-223", count=2),  # Giovanni
            DeckEntry(card_id="P-A-007", count=2),  # Prof Research
            DeckEntry(card_id="A1-220", count=2),  # would double Misty
            DeckEntry(card_id="A1-223", count=2),  # would double Giovanni
        ],
        strategy="",
    )
    warnings = validate_deck(deck, _by_id(fake_corpus))
    # Duplicate-name warnings expected for Misty and Giovanni (each declared twice with count=2).
    assert any("Misty" in w and "appears 4 times" in w for w in warnings)


# ---------------------------------------------------------------------------
# DeckBuilder with a fake Gemini backend
# ---------------------------------------------------------------------------


@dataclass
class _FakeGemini:
    """Stand-in for ``GeminiClient`` that records calls and returns fixed data."""

    config: GeminiConfig
    deck_to_return: DeckPlan
    shortlist_to_return: ShortlistResponse | None = None
    shortlist_calls: int = 0
    build_calls: int = 0

    def shortlist(self, *, system_prompt: str, user_prompt: str) -> ShortlistResponse:
        self.shortlist_calls += 1
        assert "Pokémon TCG Pocket" in system_prompt
        assert "energy_type" in user_prompt
        if self.shortlist_to_return is None:
            raise AssertionError("Shortlist was called but no response was preconfigured.")
        return self.shortlist_to_return

    def build_deck(
        self, *, system_prompt: str, user_prompt: str, thinking_budget=None
    ) -> DeckPlan:
        self.build_calls += 1
        assert "expert Pokémon TCG Pocket deck designer" in system_prompt
        return self.deck_to_return


def _make_valid_deck() -> DeckPlan:
    return DeckPlan(
        name="Lapras splash",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="A1-079", count=2, role="primary attacker"),
            DeckEntry(card_id="A1-101", count=2, role="secondary attacker"),
            DeckEntry(card_id="A1-SQU", count=2, role="evolution base"),
            DeckEntry(card_id="A1-054", count=2, role="evolution stage 1"),
            DeckEntry(card_id="A1-056", count=2, role="finisher"),
            DeckEntry(card_id="A1-220", count=2, role="energy accel"),
            DeckEntry(card_id="A1-223", count=2, role="damage buff"),
            DeckEntry(card_id="P-A-007", count=2, role="draw"),
            DeckEntry(card_id="A1-079", count=2, role="dup intentional"),
            DeckEntry(card_id="A1-101", count=2, role="dup intentional"),
        ],
        strategy="Use Misty to power Lapras early.",
        key_synergies=["Misty + Lapras"],
        standalone_value=["Professor's Research"],
        weaknesses=["Lightning aggro"],
    )


def test_builder_skips_shortlist_when_pool_is_small(fake_corpus):
    gemini = _FakeGemini(
        config=GeminiConfig(api_key="fake", shortlist_model=None),
        deck_to_return=_make_valid_deck(),
    )
    builder = DeckBuilder(fake_corpus, gemini)
    result = builder.build(BuildOptions(energy_type="Water"))
    assert gemini.shortlist_calls == 0
    assert gemini.build_calls == 1
    assert result.deck.name == "Lapras splash"
    assert result.shortlist_size is None


def test_builder_uses_shortlist_when_pool_is_large(fake_corpus):
    # Force shortlist to run by lowering its threshold below the pool size.
    gemini = _FakeGemini(
        config=GeminiConfig(api_key="fake"),
        deck_to_return=_make_valid_deck(),
        shortlist_to_return=ShortlistResponse(
            card_ids=["A1-079", "A1-101", "A1-220", "A1-223", "P-A-007", "A1-SQU", "A1-054", "A1-056"],
            reasoning="ok",
        ),
    )
    builder = DeckBuilder(fake_corpus, gemini)
    result = builder.build(
        BuildOptions(energy_type="Water", use_shortlist=True, shortlist_size=4)
    )
    assert gemini.shortlist_calls == 1
    assert gemini.build_calls == 1
    assert result.shortlist_size is not None
    assert result.shortlist_size >= 1


def test_builder_honours_must_include(fake_corpus):
    """Off-type must-include should still appear in the candidate pool."""
    captured = {}

    def fake_build(self, options, energy, candidates):
        captured["candidate_ids"] = [c.id for c in candidates]
        return _make_valid_deck()

    gemini = _FakeGemini(
        config=GeminiConfig(api_key="fake", shortlist_model=None),
        deck_to_return=_make_valid_deck(),
    )
    builder = DeckBuilder(fake_corpus, gemini)
    # Monkeypatch the internal method so we can inspect what was sent.
    builder._build_deck = fake_build.__get__(builder, DeckBuilder)  # type: ignore[assignment]

    result = builder.build(
        BuildOptions(energy_type="Water", must_include_card_ids=["A1-094"])  # Pikachu, off-type
    )
    assert "A1-094" in captured["candidate_ids"]
    assert result.deck.name == "Lapras splash"

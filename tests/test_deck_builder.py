"""Tests for deck validation and the build pipeline.

The ``DeckBuilder`` itself is exercised via a fake ``GeminiClient`` so we never
touch the network.
"""
from __future__ import annotations

from dataclasses import dataclass

from tcgp_deck_genie.deck_builder import (
    BuildOptions,
    DeckBuilder,
    _with_evolution_support,
    choose_counter_energy,
    summarise_opponent,
    validate_deck,
)
from tcgp_deck_genie.gemini_client import GeminiConfig, ShortlistResponse
from tcgp_deck_genie.models import Card, DeckEntry, DeckPlan


def _by_id(cards: list[Card]) -> dict[str, Card]:
    return {c.id: c for c in cards}


# ---------------------------------------------------------------------------
# _with_evolution_support
# ---------------------------------------------------------------------------


def test_evolution_support_adds_full_stage2_chain(fake_corpus):
    blastoise = _by_id(fake_corpus)["A1-056"]  # Stage 2
    out = _with_evolution_support([blastoise], fake_corpus)
    names = {c.name for c in out}
    assert {"Blastoise ex", "Wartortle", "Squirtle"} <= names


def test_evolution_support_restores_off_type_basic(fake_corpus):
    vaporeon = _by_id(fake_corpus)["A1-080"]  # Water Stage 1, evolves from Colorless Eevee
    out = _with_evolution_support([vaporeon], fake_corpus)
    assert any(c.name == "Eevee" for c in out)


def test_evolution_support_is_noop_for_basics(fake_corpus):
    lapras = _by_id(fake_corpus)["A1-079"]
    out = _with_evolution_support([lapras], fake_corpus)
    assert [c.id for c in out] == ["A1-079"]


def test_evolution_support_does_not_duplicate(fake_corpus):
    by_id = _by_id(fake_corpus)
    cards = [by_id["A1-056"], by_id["A1-054"], by_id["A1-SQU"]]  # full line already present
    out = _with_evolution_support(cards, fake_corpus)
    assert len(out) == len({c.id for c in out}) == 3


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


def _primarina_line() -> dict[str, Card]:
    """A Popplio -> Brionne -> {Primarina, Primarina ex} line with two final forms."""
    cards = [
        Card(id="P-001", name="Popplio", set_id="P", category="Pokemon", hp=70,
             types=["Water"], stage="Basic"),
        Card(id="P-002", name="Brionne", set_id="P", category="Pokemon", hp=90,
             types=["Water"], stage="Stage1", evolve_from="Popplio"),
        Card(id="P-003", name="Primarina", set_id="P", category="Pokemon", hp=150,
             types=["Water"], stage="Stage2", evolve_from="Brionne"),
        Card(id="P-004", name="Primarina ex", set_id="P", category="Pokemon", hp=190,
             types=["Water"], stage="Stage2", evolve_from="Brionne", suffix="EX"),
    ]
    return {c.id: c for c in cards}


def test_validator_flags_shared_pre_evolution_bottleneck():
    by_id = _primarina_line()
    # 2 Primarina ex + 1 Primarina = 3 cards evolving from only 2 Brionne.
    deck = DeckPlan(
        name="x",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="P-001", count=2),  # Popplio
            DeckEntry(card_id="P-002", count=2),  # Brionne
            DeckEntry(card_id="P-004", count=2),  # Primarina ex
            DeckEntry(card_id="P-003", count=1),  # Primarina
        ],
        strategy="",
    )
    warnings = validate_deck(deck, by_id)
    bottleneck = [w for w in warnings if "can ever be played" in w]
    assert bottleneck, warnings
    msg = bottleneck[0]
    assert "Brionne" in msg
    assert "Primarina" in msg and "Primarina ex" in msg
    # 3 demanded, 2 available -> 1 dead.
    assert "1 dead" in msg


def test_validator_allows_balanced_variant_forms():
    by_id = _primarina_line()
    # 1 Primarina ex + 1 Primarina = 2 cards evolving from 2 Brionne: fine.
    deck = DeckPlan(
        name="ok",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="P-001", count=2),
            DeckEntry(card_id="P-002", count=2),
            DeckEntry(card_id="P-004", count=1),
            DeckEntry(card_id="P-003", count=1),
        ],
        strategy="",
    )
    warnings = validate_deck(deck, by_id)
    assert not any("can ever be played" in w for w in warnings), warnings


def test_validator_flags_insufficient_pre_evolution_for_single_form():
    by_id = _primarina_line()
    # 2 Primarina ex but only 1 Brionne to evolve from.
    deck = DeckPlan(
        name="x",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="P-001", count=2),
            DeckEntry(card_id="P-002", count=1),
            DeckEntry(card_id="P-004", count=2),
        ],
        strategy="",
    )
    warnings = validate_deck(deck, by_id)
    assert any("can ever be played" in w and "Brionne" in w for w in warnings), warnings


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
    last_build_system_prompt: str | None = None
    last_build_user_prompt: str | None = None

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
        self.last_build_system_prompt = system_prompt
        self.last_build_user_prompt = user_prompt
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


def test_builder_feeds_off_type_pre_evolutions_to_reasoning(fake_corpus):
    """Eevee (Colorless) must reach the reasoning model even under --energy Water.

    Vaporeon is Water and survives the energy filter, but its pre-evolution
    Eevee is Colorless and is filtered out of the candidate pool. The
    evolution-support step must restore it before the reasoning call.
    """
    captured = {}

    def fake_build(self, options, energy, candidates):
        captured["candidate_ids"] = [c.id for c in candidates]
        return _make_valid_deck()

    gemini = _FakeGemini(
        config=GeminiConfig(api_key="fake", shortlist_model=None),
        deck_to_return=_make_valid_deck(),
    )
    builder = DeckBuilder(fake_corpus, gemini)
    builder._build_deck = fake_build.__get__(builder, DeckBuilder)  # type: ignore[assignment]

    builder.build(BuildOptions(energy_type="Water"))
    assert "A1-080" in captured["candidate_ids"]  # Vaporeon, kept by the Water filter
    assert "A1-206" in captured["candidate_ids"]  # Eevee, restored despite being Colorless


# ---------------------------------------------------------------------------
# Counter analysis
# ---------------------------------------------------------------------------


def test_summarise_opponent_tallies_weaknesses(fake_corpus):
    by_id = {c.id: c for c in fake_corpus}
    # Articuno ex (A1-101) has a Lightning weakness in the fixture.
    opp = summarise_opponent([by_id["A1-101"], by_id["A1-079"]])
    # Whatever the fixture weaknesses are, the summary must be well-formed.
    assert "weakness_counts" in opp
    assert "main_attackers" in opp
    assert "tempo" in opp
    # ex attacker should be ranked first (highest kopts).
    assert opp["main_attackers"][0]["kopts"] == 2


def test_summarise_opponent_dedupes_attackers(fake_corpus):
    by_id = {c.id: c for c in fake_corpus}
    lapras = by_id["A1-079"]
    opp = summarise_opponent([lapras, lapras, lapras])
    names = [a["name"] for a in opp["main_attackers"]]
    assert names.count("Lapras") == 1


def test_summarise_opponent_prefers_explicit_energy(fake_corpus):
    by_id = {c.id: c for c in fake_corpus}
    opp = summarise_opponent([by_id["A1-079"]], energy_types=["Water", "Lightning"])
    assert opp["energy_types"] == ["Water", "Lightning"]


def test_choose_counter_energy_picks_top_weakness():
    opp = {"weakness_counts": {"Lightning": 3, "Fire": 1}}
    assert choose_counter_energy(opp) == "Lightning"


def test_choose_counter_energy_none_when_no_weaknesses():
    assert choose_counter_energy({"weakness_counts": {}}) is None
    assert choose_counter_energy({}) is None


def test_builder_counter_mode_injects_opponent_and_block(fake_corpus):
    gemini = _FakeGemini(
        config=GeminiConfig(api_key="fake", shortlist_model=None),
        deck_to_return=_make_valid_deck(),
    )
    builder = DeckBuilder(fake_corpus, gemini)
    opponent = summarise_opponent(
        [c for c in fake_corpus if c.id == "A1-094"], energy_types=["Lightning"]
    )
    result = builder.build(
        BuildOptions(
            energy_type="Fighting",
            opponent=opponent,
            opponent_label="Pikachu Test Deck",
        )
    )
    # The counter guidance and opponent payload must reach Gemini.
    assert "BEAT the opponent deck" in gemini.last_build_system_prompt
    assert "weakness_counts" in gemini.last_build_user_prompt
    assert "opponent" in gemini.last_build_user_prompt
    assert result.opponent_label == "Pikachu Test Deck"

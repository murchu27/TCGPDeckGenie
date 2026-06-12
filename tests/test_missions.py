"""Offline tests for mission parsing, resolution, lookup, and caching."""
from __future__ import annotations

import pytest

from tcgp_deck_genie.missions import (
    MissionCard,
    MissionCorpus,
    MissionDeck,
    MissionLookupError,
    _resolve_card_id,
    find_mission,
    load_missions,
    parse_mission_page,
    save_missions,
)

# A trimmed fixture mirroring the real Bulbapedia wikitext structure, covering:
# - level-2 difficulty headings + level-3 deck headings,
# - a Pokémon entry, a Promo-A Trainer entry (no rarity), and an unresolvable entry,
# - single- and multi-energy footers.
WIKITEXT = """\
== Beginner step-up battles ==

All opponents use the following accessories in battle:

=== Ivysaur Deck (Genetic Apex) ===
Ivysaur has the Poké Ball cosmetic [[flair]].

{{TCGPocketDeckList/Header|type=Grass|title=Ivysaur Deck (Genetic Apex)}}
{{TCGPocketDeckList/Entry|1|{{TCG ID|Genetic Apex|Bulbasaur|1}}|Grass|rarity=Diamond|rarity count=1}}
{{TCGPocketDeckList/Entry|2|{{TCG ID|Genetic Apex|Paras|14}}|Grass|rarity=Diamond|rarity count=1}}
{{TCGPocketDeckList/Entry|2|{{TCG ID|Promo-A|Poké Ball|5}}|Item}}
{{TCGPocketDeckList/Footer|type=Grass|energy={{e|Grass}}}}

== Expert solo battles ==

=== Charizard ex and Moltres ex Deck (Genetic Apex) ===

{{TCGPocketDeckList/Header|type=Fire|title=Charizard ex and Moltres ex Deck (Genetic Apex)}}
{{TCGPocketDeckList/Entry|2|{{TCG ID|Genetic Apex|Charmander|33}}|Fire|rarity=Diamond|rarity count=1}}
{{TCGPocketDeckList/Entry|2|{{TCG ID|Genetic Apex|Moltres ex|47}}|Fire|rarity=Double rare}}
{{TCGPocketDeckList/Entry|1|{{TCG ID|Genetic Apex|Unknownmon|999}}|Fire}}
{{TCGPocketDeckList/Footer|type=Fire|energy={{e|Fire}}{{e|Water}}}}
"""

NAME_TO_ID = {"Genetic Apex": "A1", "Promo-A": "P-A"}
VALID_IDS = {"A1-001", "A1-014", "A1-033", "A1-047", "P-A-005"}


# ---------------------------------------------------------------------------
# _resolve_card_id
# ---------------------------------------------------------------------------


def test_resolve_pads_number_and_maps_set():
    assert _resolve_card_id("Genetic Apex", "1", NAME_TO_ID, VALID_IDS) == "A1-001"
    assert _resolve_card_id("Genetic Apex", "14", NAME_TO_ID, VALID_IDS) == "A1-014"


def test_resolve_promo_alias():
    # "Promo-A" is not a TCGdex name; the alias maps it to P-A.
    assert _resolve_card_id("Promo-A", "5", {}, VALID_IDS) == "P-A-005"


def test_resolve_unknown_returns_none():
    assert _resolve_card_id("Genetic Apex", "999", NAME_TO_ID, VALID_IDS) is None
    assert _resolve_card_id("Nonexistent Set", "1", NAME_TO_ID, VALID_IDS) is None


# ---------------------------------------------------------------------------
# parse_mission_page
# ---------------------------------------------------------------------------


def test_parse_returns_all_decks():
    decks = parse_mission_page(
        WIKITEXT, set_name="Genetic Apex", set_id="A1",
        name_to_id=NAME_TO_ID, valid_ids=VALID_IDS,
    )
    assert len(decks) == 2
    assert [d.name for d in decks] == [
        "Ivysaur Deck (Genetic Apex)",
        "Charizard ex and Moltres ex Deck (Genetic Apex)",
    ]


def test_parse_first_deck_details():
    deck = parse_mission_page(
        WIKITEXT, set_name="Genetic Apex", set_id="A1",
        name_to_id=NAME_TO_ID, valid_ids=VALID_IDS,
    )[0]
    assert deck.difficulty == "Beginner step-up battles"
    assert deck.energy_types == ["Grass"]
    assert deck.total_cards == 5  # 1 + 2 + 2
    ids = {c.name: c.card_id for c in deck.cards}
    assert ids == {"Bulbasaur": "A1-001", "Paras": "A1-014", "Poké Ball": "P-A-005"}
    assert deck.unresolved == []


def test_parse_tracks_difficulty_and_multi_energy():
    deck = parse_mission_page(
        WIKITEXT, set_name="Genetic Apex", set_id="A1",
        name_to_id=NAME_TO_ID, valid_ids=VALID_IDS,
    )[1]
    assert deck.difficulty == "Expert solo battles"
    assert deck.energy_types == ["Fire", "Water"]


def test_parse_records_unresolved_cards():
    deck = parse_mission_page(
        WIKITEXT, set_name="Genetic Apex", set_id="A1",
        name_to_id=NAME_TO_ID, valid_ids=VALID_IDS,
    )[1]
    assert deck.unresolved == ["Unknownmon"]
    unknown = next(c for c in deck.cards if c.name == "Unknownmon")
    assert unknown.card_id is None
    # The resolved attacker is still mapped.
    assert any(c.card_id == "A1-047" for c in deck.cards)


def test_resolved_card_ids_expands_counts():
    deck = parse_mission_page(
        WIKITEXT, set_name="Genetic Apex", set_id="A1",
        name_to_id=NAME_TO_ID, valid_ids=VALID_IDS,
    )[0]
    ids = deck.resolved_card_ids()
    assert ids.count("P-A-005") == 2
    assert ids.count("A1-001") == 1
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# find_mission
# ---------------------------------------------------------------------------


def _sample_decks() -> list[MissionDeck]:
    return [
        MissionDeck(name="Charizard ex and Moltres ex Deck", set_name="Genetic Apex",
                    set_id="A1", difficulty="Expert solo battles"),
        MissionDeck(name="Pikachu ex and Raichu Deck", set_name="Genetic Apex",
                    set_id="A1", difficulty="Expert solo battles"),
        MissionDeck(name="Gyarados ex Deck", set_name="Mythical Island",
                    set_id="A1a", difficulty="Advanced step-up battles"),
    ]


def test_find_mission_exact_case_insensitive():
    d = find_mission(_sample_decks(), "gyarados ex deck")
    assert d.set_id == "A1a"


def test_find_mission_substring():
    d = find_mission(_sample_decks(), "Raichu")
    assert d.name == "Pikachu ex and Raichu Deck"


def test_find_mission_filters_by_set():
    with pytest.raises(MissionLookupError):
        find_mission(_sample_decks(), "Gyarados", set_id="A1")


def test_find_mission_unknown_raises_with_suggestion():
    with pytest.raises(MissionLookupError) as exc:
        find_mission(_sample_decks(), "Charizrd ex and Moltres")
    assert "Did you mean" in str(exc.value)


# ---------------------------------------------------------------------------
# cache round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    decks = parse_mission_page(
        WIKITEXT, set_name="Genetic Apex", set_id="A1",
        name_to_id=NAME_TO_ID, valid_ids=VALID_IDS,
    )
    corpus = MissionCorpus(decks=decks, sets_included=["A1"], fetched_at=123.0)
    save_missions(corpus, cache_dir=tmp_path)
    loaded = load_missions(cache_dir=tmp_path)
    assert loaded is not None
    assert [d.name for d in loaded.decks] == [d.name for d in decks]
    assert loaded.sets_included == ["A1"]


def test_load_missing_returns_none(tmp_path):
    assert load_missions(cache_dir=tmp_path) is None


def test_mission_card_model_defaults():
    c = MissionCard(count=2, name="Poké Ball")
    assert c.card_id is None

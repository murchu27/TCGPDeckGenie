from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcgp_deck_genie.cli import _parse_counter_deck as parse_counter_deck
from tcgp_deck_genie.models import (
    ENERGY_TYPES,
    Ability,
    Attack,
    Card,
    DeckEntry,
    DeckPlan,
    OpponentDeckSpec,
)


def test_card_compact_dict_drops_empty_fields():
    card = Card(
        id="A1-079",
        name="Lapras",
        set_id="A1",
        category="Pokemon",
        hp=140,
        types=["Water"],
        stage="Basic",
        retreat=3,
        attacks=[Attack(name="Hydro Pump", cost=["Water", "Water", "Colorless"], damage="80")],
    )
    d = card.compact_dict()
    assert d["id"] == "A1-079"
    assert d["cat"] == "P"
    assert d["type"] == "Water"
    assert d["stage"] == "Basic"
    assert d["kopts"] == 1
    assert d["attacks"][0]["cost"] == "WWC"
    # No empty noise:
    assert "abilities" not in d
    assert "weak" not in d


def _pokemon(name: str, **kw) -> Card:
    return Card(
        id=kw.pop("id", "X-001"),
        name=name,
        set_id="X",
        category="Pokemon",
        hp=kw.pop("hp", 100),
        types=kw.pop("types", ["Water"]),
        stage=kw.pop("stage", "Basic"),
        **kw,
    )


def test_ko_points_three_for_mega():
    assert _pokemon("Mega Venusaur ex", suffix="EX", rarity="Four Diamond").ko_points == 3
    # "Mega" wins even without a high rarity recorded.
    assert _pokemon("Mega Gengar", rarity=None).ko_points == 3


def test_ko_points_two_for_four_diamond_and_ex():
    assert _pokemon("Charizard ex", suffix="EX").ko_points == 2
    # Rarity-driven even when the ex suffix is absent.
    assert _pokemon("Some Beefy Pokemon", rarity="Four Diamond").ko_points == 2


def test_ko_points_one_for_ordinary_pokemon():
    assert _pokemon("Squirtle", rarity="One Diamond").ko_points == 1
    assert _pokemon("Pidgey", rarity=None).ko_points == 1


def test_ko_points_zero_for_trainers():
    trainer = Card(
        id="A1-223",
        name="Giovanni",
        set_id="A1",
        category="Trainer",
        trainer_type="Supporter",
        rarity="Four Diamond",  # rarity must not matter for non-Pokémon
    )
    assert trainer.ko_points == 0


def test_compact_dict_reports_kopts_for_ex():
    d = _pokemon("Charizard ex", suffix="EX").compact_dict()
    assert d["kopts"] == 2
    assert d["suffix"] == "EX"


def test_compact_dict_for_trainer():
    trainer = Card(
        id="A1-223",
        name="Giovanni",
        set_id="A1",
        category="Trainer",
        trainer_type="Supporter",
        effect="Attacks do +10.",
    )
    d = trainer.compact_dict()
    assert d == {
        "id": "A1-223",
        "name": "Giovanni",
        "cat": "T",
        "ttype": "Supporter",
        "fx": "Attacks do +10.",
    }


def test_deck_plan_round_trip():
    plan = DeckPlan(
        name="Mono-water synergy",
        energy_types=["Water"],
        cards=[
            DeckEntry(card_id="A1-079", count=2, role="main attacker"),
            DeckEntry(card_id="A1-220", count=2, role="energy accel"),
        ],
        strategy="Open with Lapras, use Misty to spike energy.",
        key_synergies=["Misty fuels Lapras's Hydro Pump."],
        standalone_value=["Professor's Research gives consistency."],
        weaknesses=["Slow against fast Lightning aggro."],
    )
    payload = plan.model_dump_json()
    reparsed = DeckPlan.model_validate(json.loads(payload))
    assert reparsed == plan
    assert reparsed.total_cards == 4


def test_deck_plan_rejects_unknown_energy():
    with pytest.raises(ValueError):
        DeckPlan(
            name="Bad",
            energy_types=["Plasma"],  # not a real TCGP energy
            cards=[DeckEntry(card_id="X", count=1)],
            strategy="x",
        )


def test_energy_types_constant_includes_basics():
    for required in ("Water", "Fire", "Grass", "Psychic", "Colorless"):
        assert required in ENERGY_TYPES


def test_ability_is_used_in_compact_dict():
    card = Card(
        id="A1-101",
        name="Articuno ex",
        set_id="A1",
        category="Pokemon",
        hp=140,
        types=["Water"],
        stage="Basic",
        suffix="EX",
        retreat=2,
        attacks=[Attack(name="Blizzard", cost=["Water"], damage="80")],
        abilities=[Ability(name="Frost Bind", effect="Switch out.")],
    )
    d = card.compact_dict()
    assert d["suffix"] == "EX"
    assert d["abilities"] == [{"n": "Frost Bind", "fx": "Switch out."}]


def test_opponent_deck_spec_cards_only():
    spec = OpponentDeckSpec.model_validate(
        {"cards": [{"card_id": "A1-079", "count": 2}]}
    )
    assert spec.name is None
    assert spec.energy_types is None
    assert spec.total_cards == 2


def test_opponent_deck_spec_ignores_unknown_keys():
    spec = OpponentDeckSpec.model_validate(
        {
            "_format": {"description": "docs"},
            "name": "Rival",
            "cards": [{"card_id": "A1-079", "count": 1}],
        }
    )
    assert spec.name == "Rival"


def test_parse_counter_deck_minimal_root():
    name, energy, cards = parse_counter_deck(
        {"cards": [{"card_id": "A1-079", "count": 2}]}
    )
    assert name is None
    assert energy is None
    assert len(cards) == 1
    assert cards[0].card_id == "A1-079"


def test_parse_counter_deck_minimal_under_deck_key():
    name, energy, cards = parse_counter_deck(
        {
            "deck": {
                "name": "Misty's Tide",
                "energy_types": ["Water"],
                "cards": [{"card_id": "A1-079", "count": 2}],
            }
        }
    )
    assert name == "Misty's Tide"
    assert energy == ["Water"]
    assert cards[0].count == 2


def test_parse_counter_deck_full_saved_deck():
    plan = DeckPlan(
        name="Saved",
        energy_types=["Water"],
        cards=[DeckEntry(card_id="A1-079", count=2)],
        strategy="Attack with Lapras.",
    )
    name, energy, cards = parse_counter_deck({"deck": plan.model_dump(mode="json")})
    assert name == "Saved"
    assert energy == ["Water"]
    assert cards[0].card_id == "A1-079"


def test_parse_counter_deck_reads_example_file():
    path = Path(__file__).resolve().parents[1] / "example_opponent.json"
    payload = json.loads(path.read_text())
    name, energy, cards = parse_counter_deck(payload)
    assert name == "Misty's Tide"
    assert energy == ["Water"]
    assert sum(e.count for e in cards) == 20

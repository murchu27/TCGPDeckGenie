from __future__ import annotations

import json

import pytest

from tcgp_deck_genie.models import (
    ENERGY_TYPES,
    Ability,
    Attack,
    Card,
    DeckEntry,
    DeckPlan,
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
    assert d["attacks"][0]["cost"] == "WWC"
    # No empty noise:
    assert "abilities" not in d
    assert "weak" not in d


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

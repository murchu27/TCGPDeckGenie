from __future__ import annotations

import json

from click.testing import CliRunner

from tcgp_deck_genie.cache import Corpus, save_corpus
from tcgp_deck_genie.cli import main
from tcgp_deck_genie.missions import MissionCorpus, MissionDeck, save_missions
from tcgp_deck_genie.models import Card, DeckEntry, DeckPlan


def _minimal_card() -> Card:
    return Card(
        id="A1-079",
        name="Lapras",
        set_id="A1",
        category="Pokemon",
        hp=140,
        types=["Water"],
        stage="Basic",
    )


def _write_card_cache(cache_dir) -> None:
    save_corpus(
        Corpus(cards=[_minimal_card()], sets_included=["A1"], fetched_at=100.0),
        cache_dir=cache_dir,
    )


def _write_mission_cache(cache_dir) -> None:
    save_missions(
        MissionCorpus(
            decks=[
                MissionDeck(
                    name="Test Deck",
                    set_name="Genetic Apex",
                    set_id="A1",
                    difficulty="Expert solo battles",
                )
            ],
            sets_included=["A1"],
            fetched_at=200.0,
        ),
        cache_dir=cache_dir,
    )


def test_info_shows_both_caches(tmp_path):
    _write_card_cache(tmp_path)
    _write_mission_cache(tmp_path)
    result = CliRunner().invoke(main, ["--cache-dir", str(tmp_path), "info"])
    assert result.exit_code == 0
    assert "Card corpus" in result.output
    assert "Mission decks" in result.output
    assert "Cards" in result.output
    assert "Decks" in result.output


def test_info_card_only_shows_mission_hint(tmp_path):
    _write_card_cache(tmp_path)
    result = CliRunner().invoke(main, ["--cache-dir", str(tmp_path), "info"])
    assert result.exit_code == 0
    assert "Card corpus" in result.output
    assert "sync-missions" in result.output


def test_info_mission_only_exits_error(tmp_path):
    _write_mission_cache(tmp_path)
    result = CliRunner().invoke(main, ["--cache-dir", str(tmp_path), "info"])
    assert result.exit_code == 1
    assert "sync" in result.output
    assert "Mission decks" in result.output


def test_info_neither_cache_exits_error(tmp_path):
    result = CliRunner().invoke(main, ["--cache-dir", str(tmp_path), "info"])
    assert result.exit_code == 1
    assert "sync" in result.output
    assert "sync-missions" in result.output

def _saved_deck_payload() -> dict:
    return {
        "deck": DeckPlan(
            name="Misty's Tide",
            energy_types=["Water"],
            cards=[DeckEntry(card_id="A1-079", count=2, role="main attacker")],
            strategy="Open with Lapras.",
        ).model_dump(mode="json"),
        "candidate_pool_size": 42,
        "validation_warnings": [],
    }


def test_show_deck_renders_saved_file(tmp_path):
    _write_card_cache(tmp_path)
    deck_path = tmp_path / "water.deck.json"
    deck_path.write_text(json.dumps(_saved_deck_payload()))
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "show-deck", str(deck_path)]
    )
    assert result.exit_code == 0
    assert "Misty's Tide" in result.output
    assert "Lapras" in result.output


def test_show_deck_invalid_json(tmp_path):
    _write_card_cache(tmp_path)
    deck_path = tmp_path / "bad.deck.json"
    deck_path.write_text("{not json")
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "show-deck", str(deck_path)]
    )
    assert result.exit_code == 2
    assert "not valid JSON" in result.output


def test_show_deck_missing_deck_key(tmp_path):
    _write_card_cache(tmp_path)
    deck_path = tmp_path / "bad.deck.json"
    deck_path.write_text(json.dumps({"candidate_pool_size": 1}))
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "show-deck", str(deck_path)]
    )
    assert result.exit_code == 2
    assert "missing" in result.output
    assert "deck" in result.output


def test_show_deck_invalid_deck_shape(tmp_path):
    _write_card_cache(tmp_path)
    deck_path = tmp_path / "bad.deck.json"
    deck_path.write_text(json.dumps({"deck": {"name": "x"}}))
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "show-deck", str(deck_path)]
    )
    assert result.exit_code == 2
    assert "Invalid deck file" in result.output

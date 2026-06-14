from __future__ import annotations

from click.testing import CliRunner

from tcgp_deck_genie.cache import Corpus, save_corpus
from tcgp_deck_genie.cli import main
from tcgp_deck_genie.missions import MissionCorpus, MissionDeck, save_missions
from tcgp_deck_genie.models import Card


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

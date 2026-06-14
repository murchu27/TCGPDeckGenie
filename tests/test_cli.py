from __future__ import annotations

import json

from click.testing import CliRunner

from tcgp_deck_genie.cache import Corpus, save_corpus
from tcgp_deck_genie.cli import main
from tcgp_deck_genie.deck_builder import BuildOptions, BuildResult
from tcgp_deck_genie.missions import MissionCard, MissionCorpus, MissionDeck, save_missions
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


def _write_fake_corpus(cache_dir, fake_corpus) -> None:
    save_corpus(
        Corpus(
            cards=fake_corpus,
            sets_included=sorted({c.set_id for c in fake_corpus}),
            fetched_at=100.0,
        ),
        cache_dir=cache_dir,
    )


def _write_mission_with_cards(cache_dir) -> None:
    save_missions(
        MissionCorpus(
            decks=[
                MissionDeck(
                    name="Test Deck",
                    set_name="Genetic Apex",
                    set_id="A1",
                    difficulty="Expert solo battles",
                    energy_types=["Water"],
                    cards=[MissionCard(count=2, name="Lapras", card_id="A1-079")],
                )
            ],
            sets_included=["A1"],
            fetched_at=200.0,
        ),
        cache_dir=cache_dir,
    )


def _fake_build_result() -> BuildResult:
    return BuildResult(
        deck=DeckPlan(
            name="Lapras splash",
            energy_types=["Water"],
            cards=[DeckEntry(card_id="A1-079", count=2, role="attacker")],
            strategy="Use Misty to power Lapras early.",
        ),
        cards_used=[],
        candidate_pool_size=12,
        shortlist_size=None,
        validation_warnings=[],
    )


class _FakeBuilder:
    """Records CLI wiring without calling Gemini."""

    def __init__(self, cards, gemini) -> None:
        self.last_options: BuildOptions | None = None

    def build(self, options: BuildOptions) -> BuildResult:
        self.last_options = options
        return _fake_build_result()


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

# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_filters_by_energy(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "search", "--energy", "Water"]
    )
    assert result.exit_code == 0
    assert "match(es)" in result.output
    assert "Lapras" in result.output
    assert "Pikachu" not in result.output


def test_search_filters_by_keyword(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "search", "--keyword", "flip a coin"]
    )
    assert result.exit_code == 0
    assert "Misty" in result.output


def test_search_no_cache_exits_error(tmp_path):
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "search", "--energy", "Water"]
    )
    assert result.exit_code == 1
    assert "sync" in result.output


# ---------------------------------------------------------------------------
# missions
# ---------------------------------------------------------------------------


def test_missions_lists_cached_decks(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    _write_mission_cache(tmp_path)
    result = CliRunner().invoke(main, ["--cache-dir", str(tmp_path), "missions"])
    assert result.exit_code == 0
    assert "1 mission(s)" in result.output
    assert "Test Deck" in result.output


def test_missions_shows_detail_by_name(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    _write_mission_with_cards(tmp_path)
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "missions", "Test Deck"]
    )
    assert result.exit_code == 0
    assert "Lapras" in result.output
    assert "A1-079" in result.output


def test_missions_lookup_error(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    _write_mission_cache(tmp_path)
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "missions", "No Such Deck"]
    )
    assert result.exit_code == 2


def test_missions_no_mission_cache_exits_error(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    result = CliRunner().invoke(main, ["--cache-dir", str(tmp_path), "missions"])
    assert result.exit_code == 1
    assert "sync-missions" in result.output


# ---------------------------------------------------------------------------
# build-deck (mocked — no Gemini / network)
# ---------------------------------------------------------------------------


def test_build_deck_requires_energy(tmp_path, fake_corpus):
    _write_fake_corpus(tmp_path, fake_corpus)
    result = CliRunner().invoke(
        main, ["--cache-dir", str(tmp_path), "build-deck"]
    )
    assert result.exit_code == 2
    assert "--energy is required" in result.output


def test_build_deck_rejects_both_counter_modes(tmp_path, fake_corpus, tmp_path_factory):
    _write_fake_corpus(tmp_path, fake_corpus)
    counter = tmp_path / "opp.json"
    counter.write_text(json.dumps({"cards": [{"card_id": "A1-079", "count": 2}]}))
    result = CliRunner().invoke(
        main,
        [
            "--cache-dir",
            str(tmp_path),
            "build-deck",
            "--counter-mission",
            "Test Deck",
            "--counter-file",
            str(counter),
        ],
    )
    assert result.exit_code == 2
    assert "only one of" in result.output


def test_build_deck_success_with_mocked_builder(tmp_path, fake_corpus, monkeypatch):
    _write_fake_corpus(tmp_path, fake_corpus)
    fake_builder = _FakeBuilder([], object())
    monkeypatch.setattr("tcgp_deck_genie.cli.GeminiClient", lambda: object())
    monkeypatch.setattr("tcgp_deck_genie.cli.DeckBuilder", lambda cards, gemini: fake_builder)

    result = CliRunner().invoke(
        main,
        [
            "--cache-dir",
            str(tmp_path),
            "build-deck",
            "--energy",
            "Water",
            "--no-shortlist",
            "--brief",
            "Aggro water",
        ],
    )
    assert result.exit_code == 0
    assert "Lapras splash" in result.output
    assert fake_builder.last_options is not None
    assert fake_builder.last_options.energy_type == "Water"
    assert fake_builder.last_options.user_brief == "Aggro water"


def test_build_deck_writes_out_file(tmp_path, fake_corpus, monkeypatch):
    _write_fake_corpus(tmp_path, fake_corpus)
    monkeypatch.setattr("tcgp_deck_genie.cli.GeminiClient", lambda: object())
    monkeypatch.setattr("tcgp_deck_genie.cli.DeckBuilder", _FakeBuilder)

    out = tmp_path / "water.deck.json"
    result = CliRunner().invoke(
        main,
        [
            "--cache-dir",
            str(tmp_path),
            "build-deck",
            "--energy",
            "Water",
            "--no-shortlist",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["deck"]["name"] == "Lapras splash"

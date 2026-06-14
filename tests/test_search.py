from __future__ import annotations

import pytest

from tcgp_deck_genie.search import (
    SearchFilter,
    apply_filter,
    candidate_score,
    top_candidates,
)


def test_energy_filter_keeps_water_pokemon_and_all_trainers(fake_corpus):
    spec = SearchFilter(energy_type="Water")
    out = apply_filter(fake_corpus, spec)
    names = {c.name for c in out}
    # Water Pokémon kept:
    assert {"Lapras", "Squirtle", "Wartortle", "Blastoise ex", "Articuno ex"}.issubset(names)
    # Off-type Pokémon dropped:
    assert "Pikachu" not in names
    # Trainers always survive the energy filter (they're typeless):
    assert {"Misty", "Giovanni", "Professor's Research"}.issubset(names)


def test_energy_filter_rejects_unknown_type(fake_corpus):
    spec = SearchFilter(energy_type="plasma")
    with pytest.raises(ValueError):
        apply_filter(fake_corpus, spec)


def test_no_ex_filter_excludes_ex(fake_corpus):
    spec = SearchFilter(energy_type="Water", include_ex=False)
    out = apply_filter(fake_corpus, spec)
    assert all(not c.is_ex for c in out if c.is_pokemon)


def test_keyword_filter_searches_attack_and_effect_text(fake_corpus):
    spec = SearchFilter(keywords=["flip a coin"])
    out = apply_filter(fake_corpus, spec)
    names = {c.name for c in out}
    assert "Misty" in names
    assert "Lapras" not in names


def test_set_filter_restricts_results(fake_corpus):
    spec = SearchFilter(set_ids={"P-A"})
    out = apply_filter(fake_corpus, spec)
    assert {c.set_id for c in out} == {"P-A"}


def test_category_filter(fake_corpus):
    spec = SearchFilter(category="Trainer")
    out = apply_filter(fake_corpus, spec)
    assert {c.category for c in out} == {"Trainer"}


def test_candidate_score_prefers_useful_cards(fake_corpus):
    # Lapras (high HP basic) should outscore Squirtle (low HP basic).
    by_name = {c.name: c for c in fake_corpus}
    assert candidate_score(by_name["Lapras"]) > candidate_score(by_name["Squirtle"])
    # Trainers with "draw" should beat plain Supporters.
    assert candidate_score(by_name["Professor's Research"]) > candidate_score(
        by_name["Giovanni"]
    )


def test_top_candidates_caps_results(fake_corpus):
    kept = top_candidates(fake_corpus, limit=3)
    assert len(kept) == 3

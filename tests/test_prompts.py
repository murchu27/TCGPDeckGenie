from __future__ import annotations

import json

from tcgp_deck_genie.prompts import (
    DECK_SIZE,
    MAX_COPIES_PER_NAME,
    POCKET_RULES_BLOCK,
    deck_system_prompt,
    deck_user_prompt,
    shortlist_system_prompt,
    shortlist_user_prompt,
)


def test_rules_block_mentions_format_constants():
    assert str(DECK_SIZE) in POCKET_RULES_BLOCK
    assert str(MAX_COPIES_PER_NAME) in POCKET_RULES_BLOCK
    assert "Basic Pokémon" in POCKET_RULES_BLOCK


def test_shortlist_user_prompt_is_valid_json_after_marker(fake_corpus):
    prompt = shortlist_user_prompt(
        user_brief="fun deck",
        energy_type="Water",
        candidates=fake_corpus,
        shortlist_size=5,
    )
    payload = prompt.split("```json", 1)[1].rsplit("```", 1)[0]
    decoded = json.loads(payload)
    assert decoded["energy_type"] == "Water"
    assert decoded["shortlist_size"] == 5
    assert len(decoded["candidates"]) == len(fake_corpus)


def test_deck_user_prompt_includes_must_includes(fake_corpus):
    prompt = deck_user_prompt(
        user_brief="agro",
        energy_type="Water",
        candidates=fake_corpus,
        must_include_ids=["A1-079"],
    )
    payload = prompt.split("```json", 1)[1].rsplit("```", 1)[0]
    decoded = json.loads(payload)
    assert decoded["must_include_card_ids"] == ["A1-079"]


def test_system_prompts_share_rules_block():
    assert POCKET_RULES_BLOCK in shortlist_system_prompt()
    assert POCKET_RULES_BLOCK in deck_system_prompt()

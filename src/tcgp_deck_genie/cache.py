"""On-disk cache for the TCGP card corpus.

The TCGdex catalogue changes only when new sets ship, so fetching every card
once and reusing it across runs is the single biggest cost win we have:

* Zero TCGdex API traffic for routine deck building.
* The corpus loads in <50 ms from local JSON, instead of a few hundred HTTP
  calls (one per card) through the SDK.
* Reproducible runs - the same cached corpus produces the same prompts.

The cache is a single JSON file at ``$TCGP_CACHE_DIR/cards_tcgp.json`` with a
small metadata header so we know when it was built and which sets are covered.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .models import Card

DEFAULT_CACHE_DIRNAME = ".tcgp_deck_genie"
CORPUS_FILENAME = "cards_tcgp.json"
CORPUS_SCHEMA_VERSION = 1


def default_cache_dir() -> Path:
    """Return the cache dir, honouring ``TCGP_CACHE_DIR`` if set."""
    env = os.environ.get("TCGP_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / DEFAULT_CACHE_DIRNAME


@dataclass
class Corpus:
    """In-memory view of the cached card corpus."""

    cards: list[Card]
    sets_included: list[str]
    fetched_at: float
    schema_version: int = CORPUS_SCHEMA_VERSION

    def by_id(self) -> dict[str, Card]:
        return {c.id: c for c in self.cards}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def corpus_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_cache_dir()) / CORPUS_FILENAME


def save_corpus(corpus: Corpus, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    _ensure_dir(cache_dir)
    payload = {
        "schema_version": corpus.schema_version,
        "fetched_at": corpus.fetched_at,
        "sets_included": corpus.sets_included,
        "cards": [c.model_dump(mode="json") for c in corpus.cards],
    }
    out = corpus_path(cache_dir)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(out)
    return out


def load_corpus(cache_dir: Path | None = None) -> Corpus | None:
    path = corpus_path(cache_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("schema_version") != CORPUS_SCHEMA_VERSION:
        return None
    cards = [Card.model_validate(c) for c in data["cards"]]
    return Corpus(
        cards=cards,
        sets_included=list(data.get("sets_included", [])),
        fetched_at=float(data.get("fetched_at", time.time())),
        schema_version=int(data["schema_version"]),
    )


def corpus_info(cache_dir: Path | None = None) -> dict | None:
    """Return a small summary dict without deserialising every card."""
    path = corpus_path(cache_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        "path": str(path),
        "fetched_at": data.get("fetched_at"),
        "sets_included": data.get("sets_included", []),
        "card_count": len(data.get("cards", [])),
        "schema_version": data.get("schema_version"),
    }

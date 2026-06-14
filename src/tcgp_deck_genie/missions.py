"""Solo-battle (mission) opponent decks, sourced from Bulbapedia.

TCG Pocket's single-player content pits you against fixed AI decks. Knowing the
exact list the computer plays lets us build a deck specifically to counter it -
and lets a user look a mission up by name instead of transcribing 20 cards.

Why Bulbapedia:

* Its "List of <expansion> solo battles in Pokémon TCG Pocket" pages expose the
  decks as structured wikitext templates via the MediaWiki API - no HTML
  scraping. Each entry carries exact card identity, e.g.
  ``{{TCGPocketDeckList/Entry|2|{{TCG ID|Genetic Apex|Paras|14}}|Grass|...}}``,
  which maps straight onto a TCGdex id (``A1-014``).
* The data changes only when new solo battles ship, so - exactly like the card
  corpus - we fetch once into an on-disk cache and never hit the network again
  during routine use.

Content from Bulbapedia is licensed CC BY-NC-SA; see the README for attribution.

This module keeps the *parsing and resolution* logic as pure functions (easy to
unit-test offline) and isolates the network in ``BulbapediaClient``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from pydantic import BaseModel, Field

from .cache import default_cache_dir

logger = logging.getLogger(__name__)

MISSIONS_FILENAME = "missions_tcgp.json"
MISSIONS_SCHEMA_VERSION = 1

# Bulbapedia's TCG ID templates name a few sets differently from TCGdex (and the
# promo set has no expansion page of its own), so we alias them onto TCGdex ids.
SET_NAME_ALIASES: dict[str, str] = {
    "Promo-A": "P-A",
    "Promos-A": "P-A",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MissionCard(BaseModel):
    """One line of an opponent's deck."""

    count: int
    name: str
    card_id: str | None = None  # resolved TCGdex id, or None if unresolved


class MissionDeck(BaseModel):
    """A single computer-opponent deck from a solo battle."""

    name: str = Field(..., description="Deck name, e.g. 'Charizard ex and Moltres ex Deck'.")
    set_name: str = Field(..., description="Expansion the battle belongs to.")
    set_id: str | None = Field(None, description="TCGdex set id for the expansion.")
    difficulty: str | None = Field(None, description="Difficulty tier heading, if known.")
    energy_types: list[str] = Field(default_factory=list)
    cards: list[MissionCard] = Field(default_factory=list)
    unresolved: list[str] = Field(
        default_factory=list,
        description="Card names that could not be mapped to a TCGdex id.",
    )

    @property
    def total_cards(self) -> int:
        return sum(c.count for c in self.cards)

    def resolved_card_ids(self) -> list[str]:
        """Flattened, count-expanded list of resolved TCGdex ids."""
        out: list[str] = []
        for c in self.cards:
            if c.card_id:
                out.extend([c.card_id] * c.count)
        return out


# ---------------------------------------------------------------------------
# Cache (mirrors cache.py for the card corpus)
# ---------------------------------------------------------------------------


@dataclass
class MissionCorpus:
    """In-memory view of the cached mission decks."""

    decks: list[MissionDeck]
    sets_included: list[str]
    fetched_at: float
    schema_version: int = MISSIONS_SCHEMA_VERSION


def missions_path(cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_cache_dir()) / MISSIONS_FILENAME


def save_missions(corpus: MissionCorpus, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": corpus.schema_version,
        "fetched_at": corpus.fetched_at,
        "sets_included": corpus.sets_included,
        "decks": [d.model_dump(mode="json") for d in corpus.decks],
    }
    out = missions_path(cache_dir)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(out)
    return out


def load_missions(cache_dir: Path | None = None) -> MissionCorpus | None:
    path = missions_path(cache_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("schema_version") != MISSIONS_SCHEMA_VERSION:
        return None
    decks = [MissionDeck.model_validate(d) for d in data["decks"]]
    return MissionCorpus(
        decks=decks,
        sets_included=list(data.get("sets_included", [])),
        fetched_at=float(data.get("fetched_at", time.time())),
        schema_version=int(data["schema_version"]),
    )


def missions_info(cache_dir: Path | None = None) -> dict | None:
    path = missions_path(cache_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        "path": str(path),
        "fetched_at": data.get("fetched_at"),
        "sets_included": data.get("sets_included", []),
        "deck_count": len(data.get("decks", [])),
        "schema_version": data.get("schema_version"),
    }


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class MissionLookupError(LookupError):
    """Raised when a mission name cannot be resolved to a single deck."""


def find_mission(
    decks: list[MissionDeck],
    query: str,
    *,
    set_id: str | None = None,
    difficulty: str | None = None,
) -> MissionDeck:
    """Find exactly one mission deck by (fuzzy) name, optionally filtered.

    Resolution order: exact (case-insensitive) name, then unique substring match,
    then fuzzy close-match. Raises ``MissionLookupError`` with suggestions when
    nothing matches or the match is ambiguous.
    """
    pool = decks
    if set_id:
        pool = [d for d in pool if d.set_id == set_id]
    if difficulty:
        dl = difficulty.lower()
        pool = [d for d in pool if d.difficulty and dl in d.difficulty.lower()]

    if not pool:
        raise MissionLookupError(
            f"No missions match the given filters (set_id={set_id!r}, "
            f"difficulty={difficulty!r})."
        )

    q = query.strip().lower()

    exact = [d for d in pool if d.name.lower() == q]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise MissionLookupError(_ambiguous_msg(query, exact))

    substr = [d for d in pool if q in d.name.lower()]
    if len(substr) == 1:
        return substr[0]
    if len(substr) > 1:
        raise MissionLookupError(_ambiguous_msg(query, substr))

    names = [d.name for d in pool]
    close = get_close_matches(query, names, n=5, cutoff=0.4)
    if close:
        raise MissionLookupError(
            f"No mission named {query!r}. Did you mean: " + "; ".join(close) + "?"
        )
    raise MissionLookupError(
        f"No mission named {query!r}. Run 'tcgp-deck-genie missions' to list them."
    )


def _ambiguous_msg(query: str, matches: list[MissionDeck]) -> str:
    labels = [f"{d.name} ({d.set_id}/{d.difficulty})" for d in matches]
    return (
        f"{query!r} matches multiple missions; narrow with --set/--difficulty or "
        f"use the exact name: " + "; ".join(labels)
    )


# ---------------------------------------------------------------------------
# Wikitext parsing (pure functions)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.MULTILINE)
_HEADER_RE = re.compile(r"\{\{TCGPocketDeckList/Header\|([^{}]*)\}\}")
_ENTRY_RE = re.compile(
    r"\{\{TCGPocketDeckList/Entry\|\s*(\d+)\s*\|\s*"
    r"\{\{TCG ID\|([^|]+)\|([^|}]+)\|([^|}]+)\}\}"
)
_ENERGY_RE = re.compile(r"\{\{e\|([^}|]+)\}\}")


def _resolve_card_id(
    set_name: str,
    number: str,
    name_to_id: dict[str, str],
    valid_ids: set[str],
) -> str | None:
    """Map a TCG ID ``(set name, local number)`` to a TCGdex card id.

    Returns ``None`` when the set is unknown or the constructed id is not present
    in the card corpus (e.g. an excluded rarity or a numbering mismatch).
    """
    set_id = name_to_id.get(set_name.strip()) or SET_NAME_ALIASES.get(set_name.strip())
    if not set_id:
        return None
    try:
        local = int(re.sub(r"[^0-9]", "", number))
    except ValueError:
        return None
    card_id = f"{set_id}-{local:03d}"
    return card_id if card_id in valid_ids else None


def _parse_header_params(raw: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for chunk in raw.split("|"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def parse_mission_page(
    wikitext: str,
    *,
    set_name: str,
    set_id: str | None,
    name_to_id: dict[str, str],
    valid_ids: set[str],
) -> list[MissionDeck]:
    """Parse every solo-battle deck out of one expansion's wikitext page."""
    # Pre-compute level-2 headings (difficulty tiers) and their positions so each
    # deck can be tagged with the tier it sits under.
    tiers: list[tuple[int, str]] = []
    for m in _HEADING_RE.finditer(wikitext):
        if len(m.group(1)) == 2:
            tiers.append((m.start(), m.group(2).strip()))

    def difficulty_at(pos: int) -> str | None:
        current = None
        for start, title in tiers:
            if start < pos:
                current = title
            else:
                break
        return current

    headers = list(_HEADER_RE.finditer(wikitext))
    decks: list[MissionDeck] = []
    for i, hm in enumerate(headers):
        block_start = hm.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(wikitext)
        block = wikitext[block_start:block_end]

        params = _parse_header_params(hm.group(1))
        name = params.get("title") or f"{set_name} deck {i + 1}"

        cards: list[MissionCard] = []
        unresolved: list[str] = []
        for em in _ENTRY_RE.finditer(block):
            count = int(em.group(1))
            entry_set = em.group(2).strip()
            card_name = em.group(3).strip()
            number = em.group(4).strip()
            cid = _resolve_card_id(entry_set, number, name_to_id, valid_ids)
            cards.append(MissionCard(count=count, name=card_name, card_id=cid))
            if cid is None:
                unresolved.append(card_name)

        energy_types: list[str] = []
        for en in _ENERGY_RE.finditer(block):
            t = en.group(1).strip()
            if t and t not in energy_types:
                energy_types.append(t)

        decks.append(
            MissionDeck(
                name=name,
                set_name=set_name,
                set_id=set_id,
                difficulty=difficulty_at(hm.start()),
                energy_types=energy_types,
                cards=cards,
                unresolved=unresolved,
            )
        )
    return decks


# ---------------------------------------------------------------------------
# Network client
# ---------------------------------------------------------------------------


@dataclass
class MissionFetchProgress:
    """Progress payload for the mission sync (mirrors FetchProgress)."""

    set_name: str
    completed: int
    total: int


MissionProgressCallback = Callable[[MissionFetchProgress], None] | None


class BulbapediaClient:
    """Fetches and parses solo-battle pages from Bulbapedia's MediaWiki API."""

    BASE_URL = "https://bulbapedia.bulbagarden.net/w/api.php"
    # Bulbapedia 403s the default urllib UA; a descriptive UA is required.
    USER_AGENT = os.environ.get(
        "TCGP_BULBAPEDIA_UA",
        "TCGPDeckGenie/0.1 (portfolio project; +https://github.com/murchu27/TCGPDeckGenie)",
    )

    def __init__(self, name_to_id: dict[str, str], valid_ids: set[str], timeout: float = 30.0):
        self._name_to_id = dict(name_to_id)
        self._valid_ids = set(valid_ids)
        self._timeout = timeout

    @staticmethod
    def page_title(expansion_name: str) -> str:
        return f"List of {expansion_name} solo battles in Pokémon TCG Pocket"

    def fetch_page_wikitext(self, expansion_name: str) -> str | None:
        """Fetch raw wikitext for an expansion's solo-battle page, or None if absent."""
        params = urllib.parse.urlencode(
            {
                "action": "parse",
                "page": self.page_title(expansion_name),
                "prop": "wikitext",
                "format": "json",
                "redirects": 1,
            }
        )
        url = f"{self.BASE_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network failure
            logger.warning("Bulbapedia HTTP %s for %r", exc.code, expansion_name)
            return None
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            logger.warning("Bulbapedia request failed for %r: %s", expansion_name, exc)
            return None
        if "error" in data:
            logger.info("No Bulbapedia page for %r (%s)", expansion_name, data["error"].get("code"))
            return None
        try:
            return data["parse"]["wikitext"]["*"]
        except (KeyError, TypeError):  # pragma: no cover - unexpected shape
            logger.warning("Unexpected Bulbapedia response shape for %r", expansion_name)
            return None

    def fetch_mission_decks(
        self,
        expansions: Iterable[tuple[str, str]],
        progress: MissionProgressCallback = None,
    ) -> tuple[list[MissionDeck], list[str]]:
        """Fetch and parse decks for ``(set_id, set_name)`` pairs.

        Returns ``(decks, skipped_set_names)``; a set is skipped when it has no
        solo-battle page yet (common for the newest expansions).
        """
        expansions = list(expansions)
        total = len(expansions)
        decks: list[MissionDeck] = []
        skipped: list[str] = []
        for idx, (set_id, set_name) in enumerate(expansions, start=1):
            if progress is not None:
                progress(MissionFetchProgress(set_name=set_name, completed=idx - 1, total=total))
            wikitext = self.fetch_page_wikitext(set_name)
            if wikitext is None:
                skipped.append(set_name)
            else:
                decks.extend(
                    parse_mission_page(
                        wikitext,
                        set_name=set_name,
                        set_id=set_id,
                        name_to_id=self._name_to_id,
                        valid_ids=self._valid_ids,
                    )
                )
            if progress is not None:
                progress(MissionFetchProgress(set_name=set_name, completed=idx, total=total))
        return decks, skipped

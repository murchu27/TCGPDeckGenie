# TCGPDeckGenie

A small portfolio-grade Python application that designs **Pokémon TCG Pocket** decks by combining:

* The [**TCGdex** Python SDK](https://pypi.org/project/tcgdex-sdk/) for the card data, and
* **Google Gemini** (free tier) for the reasoning - explicitly weighing *card synergy* against *raw standalone utility*, the two ways a card earns its slot in a 20-card deck.

The whole thing is built to be **runnable on Gemini's free tier**. The architecture below explains how that constraint shaped the design.

---

## Quickstart

```bash
# Clone and enter the repo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# Paste a free key from https://aistudio.google.com/apikey into GEMINI_API_KEY

# 1. Fetch the TCGP card corpus once (writes ~3 MB to ~/.tcgp_deck_genie/).
#    Star / Shiny / Crown rares are filtered out by default; pass --include-rares
#    if you have access to them and want them in the candidate pool.
tcgp-deck-genie sync

# 2. Browse cards locally - zero API calls, zero cost.
tcgp-deck-genie search --energy Water --keyword "flip a coin"

# 3. Ask Gemini to design a deck.
tcgp-deck-genie build-deck \
    --energy Water \
    --brief "Aggressive Water deck that wins by turn 4 using coin-flip energy accel" \
    --out water_aggro.deck.json

# 4. Re-display a previously saved deck (no API cost).
tcgp-deck-genie show-deck water_aggro.deck.json
```

## Countering mission decks

TCG Pocket's solo battles pit you against fixed AI decks. TCGPDeckGenie can fetch
those decks and build a deck specifically to beat one:

```bash
# 1. Fetch the solo-battle (mission) opponent decks once (one-time, like `sync`).
tcgp-deck-genie sync-missions

# 2. Browse / search the mission decks locally (no API cost).
tcgp-deck-genie missions --difficulty expert
tcgp-deck-genie missions "Charizard ex and Moltres ex"   # show one in detail

# 3. Build a counter. --energy is optional here: if omitted it is auto-picked
#    from the type that exploits the most of the opponent's weaknesses.
tcgp-deck-genie build-deck \
    --counter-mission "Charizard ex and Moltres ex" \
    --out anti_charizard.deck.json

# You can also counter any deck you previously saved with --out:
tcgp-deck-genie build-deck --counter-file water_aggro.deck.json
```

Before the LLM is involved, the opponent deck is digested locally into a compact
summary (weakness tally, key threats with prize values, tempo profile). Only that
summary - not the raw 20-card list - goes into the prompt, so countering costs the
same 1-2 Gemini calls as a normal build.

Mission data comes from [Bulbapedia](https://bulbapedia.bulbagarden.net)'s
"List of &lt;expansion&gt; solo battles in Pokémon TCG Pocket" pages, fetched via the
MediaWiki API and cached locally. Bulbapedia content is licensed
[CC BY-NC-SA 2.5](https://creativecommons.org/licenses/by-nc-sa/2.5/).

## What you get back

Each deck is a structured JSON object Gemini fills in against a Pydantic schema:

```json
{
  "name": "Misty's Tide",
  "energy_types": ["Water"],
  "cards": [
    { "card_id": "A1-079", "count": 2, "role": "main attacker" },
    { "card_id": "A1-220", "count": 2, "role": "energy accel" },
    ...
  ],
  "strategy": "Open with Lapras, use Misty turn 1-2 for explosive Hydro Pumps.",
  "key_synergies": [
    "Misty + Lapras: heads-flip accel powers a turn-2 Hydro Pump.",
    "Articuno ex's Frost Bind switches a stranded attacker back to the bench."
  ],
  "standalone_value": [
    "Professor's Research is included purely for draw consistency.",
    "Giovanni is a flat +10 damage anywhere - no synergy required."
  ],
  "weaknesses": [
    "Lightning aggro hits Water weakness early.",
    "Blastoise ex line is bricked without a Squirtle in opening hand."
  ]
}
```

After Gemini answers, a local validator double-checks the format constraints (exactly 20 cards, ≤2 copies per name, at least one Basic, evolution pre-requisites, attack energy is reachable from the declared types). Violations are surfaced as warnings rather than errors so you can still see and inspect off-spec decks.

---

## How it stays cheap

This is a portfolio project, so **cost-awareness is a core design constraint**, not an afterthought. Here's where each design decision pays off:

| Concern | Design choice | Effect |
| --- | --- | --- |
| Avoid redundant TCGdex traffic | A one-shot `sync` command writes the whole TCGP corpus to a local JSON cache. | Subsequent runs make **zero** TCGdex calls. |
| Don't recommend cards the user can't realistically obtain | `sync` filters out Star / Shiny / Crown rarities by default (toggle with `--include-rares`). | The candidate pool only contains cards the player can plausibly own. |
| Avoid stuffing 2,000 cards into every prompt | A deterministic local filter (energy type, set, keywords, retreat cost, ex toggle) narrows the corpus to ~100-250 candidates *before* any LLM call. | The reasoning model only ever sees a few-hundred-card payload, not the whole catalogue. |
| Avoid sending verbose JSON | A `Card.compact_dict()` projection drops illustrator / image / pricing / variant fields and abbreviates attack costs (`"WWC"` instead of `["Water","Water","Colorless"]`). | A typical card serialises in ~150 tokens instead of ~500. |
| Reduce reasoning-model tokens | A cheap **shortlist pass** (default `gemini-2.5-flash-lite`, thinking disabled) picks the ~40 most promising candidates first. The expensive reasoning pass (`gemini-2.5-flash` with a 2k thinking budget) then sees only that subset. | The pricey call's input shrinks ~3-5×; the shortlist call is small enough to be free-tier-friendly. |
| Avoid free-form text we can't parse | Both Gemini calls use `response_mime_type="application/json"` with a `response_json_schema` derived from Pydantic models. | We never re-prompt for "please reply in JSON"; outputs validate or raise. |
| Don't repeat expensive runs | Decks save to disk via `--out`; `show-deck <path>` re-renders them with full prettification and re-runs validation locally - no API call. | Iterating on deck *interpretation* (vs deck *design*) is free. |
| Tune reasoning vs latency per run | `--thinking-budget 0` disables thinking entirely; the default 2,048 is enough to surface real synergy reasoning without burning the daily quota. | One knob, transparent cost. |

The default model choices target the [Gemini API free tier](https://ai.google.dev/gemini-api/docs/rate-limits) (currently 10 RPM / 250 RPD on `gemini-2.5-flash`, 30 RPM / 1,000 RPD on `gemini-2.5-flash-lite`). A typical deck build is 1-2 requests. You can override both models in `.env`:

```bash
TCGP_GEMINI_REASONING_MODEL=gemini-2.5-flash       # default
TCGP_GEMINI_SHORTLIST_MODEL=gemini-2.5-flash-lite  # default; empty disables the shortlist step
TCGP_THINKING_BUDGET=2048                          # default
```

### Falling back to a local model

The `gemini_client.GeminiClient` exposes exactly two methods (`shortlist` and `build_deck`) and the rest of the codebase never imports Google's SDK directly. To add an Ollama / llama.cpp fallback, drop a `local_client.py` next to it that implements the same two methods and swap it into the CLI - no other code changes required.

---

## Architecture

```
src/tcgp_deck_genie/
├── models.py         # Pydantic: Card, Attack, Ability, DeckEntry, DeckPlan
├── cache.py          # On-disk JSON cache for the TCGP corpus
├── tcgp_client.py    # TCGdex SDK wrapper: fetch, normalise, dedupe reprints
├── missions.py       # Bulbapedia solo-battle decks: fetch, parse, resolve, cache
├── search.py         # Deterministic local filter + cheap candidate scoring
├── prompts.py        # All Gemini prompts in one place (rules + counter block + templates)
├── gemini_client.py  # google-genai wrapper: structured output + thinking budget + retries
├── deck_builder.py   # Two-stage pipeline + counter analysis + deck validator
└── cli.py            # Click CLI: sync, sync-missions, info, search, missions, build-deck, show-deck
```

The pipeline at runtime:

```
TCGdex SDK ──[ sync, once ]──▶ local JSON cache
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        deterministic SearchFilter ─▶ top-N by cheap heuristic
                       │
                       ▼
       (optional) shortlist call to gemini-2.5-flash-lite
                       │
                       ▼
       reasoning call to gemini-2.5-flash (with thinking_budget)
                       │
                       ▼
                  validate_deck()
                       │
                       ▼
                rich-rendered output + optional --out JSON
```

---

## CLI reference

```text
tcgp-deck-genie sync           # download the TCGP corpus (one-time, ~30-90 s)
tcgp-deck-genie sync-missions  # download solo-battle opponent decks (one-time)
tcgp-deck-genie info           # print cache summary
tcgp-deck-genie search         # filter the local corpus, no API cost
tcgp-deck-genie missions       # list/search/show solo-battle decks, no API cost
tcgp-deck-genie build-deck     # produce a 20-card deck via Gemini
tcgp-deck-genie show-deck      # re-render a saved deck, no API cost
```

Run any subcommand with `--help` for its full option list. Key flags on `build-deck`:

| Flag | Default | Notes |
| --- | --- | --- |
| `--energy` | – | One of `Grass Fire Water Lightning Psychic Fighting Darkness Metal Dragon Colorless`. Required unless in counter mode (then auto-picked from opponent weaknesses if omitted). |
| `--counter-mission` | – | Name of a cached solo-battle mission to counter. |
| `--counter-file` | – | Path to a saved deck JSON (from `--out`) to counter. |
| `--brief` | `"Build a strong, fun deck."` | Short free-text framing of the deck. |
| `--set` (repeatable) | all TCGP sets | Restrict candidates to specific sets. |
| `--keyword` (repeatable) | – | Substring(s) that must appear in card name/text. |
| `--must-include` (repeatable) | – | Card ids the deck must contain. |
| `--no-ex` | off | Exclude Pokémon ex from candidates. |
| `--no-shortlist` | off | Skip the cheap shortlist call. |
| `--shortlist-size` | 40 | How many cards the shortlist keeps. |
| `--candidate-cap` | 120 | Hard cap on candidates before any LLM stage. |
| `--thinking-budget` | env / 2048 | 0 disables thinking. |
| `--out` | – | Save the JSON deck to a file. |

---

## Development

```bash
pip install -e ".[dev]"
pytest -q       # network-free tests cover models, search, validation, prompts, and the builder pipeline
ruff check .
```

All tests use a fake Gemini backend (`tests/test_deck_builder.py::_FakeGemini`), so the suite runs offline and burns zero quota.

## Why TCG Pocket?

Pokémon TCG Pocket has tight format constraints (20 cards, 1-3 energy types, no in-deck basic energy) that make it a perfect playground for LLM-driven deck design: the search space is small enough that a single reasoning call can take a real swing at it, but large enough that *which* 20 cards you pick - and how they support each other - genuinely matters.

## License

GPL v3.

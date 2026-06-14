"""Click-based command-line interface for TCGPDeckGenie."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .cache import Corpus, corpus_info, corpus_path, default_cache_dir, load_corpus, save_corpus
from .deck_builder import (
    BuildOptions,
    BuildResult,
    DeckBuilder,
    DeckBuildError,
    choose_counter_energy,
    summarise_opponent,
)
from .gemini_client import GeminiClient, GeminiClientError
from .missions import (
    SET_NAME_ALIASES,
    BulbapediaClient,
    MissionCorpus,
    MissionDeck,
    MissionFetchProgress,
    MissionLookupError,
    find_mission,
    load_missions,
    missions_info,
    missions_path,
    save_missions,
)
from .models import ENERGY_TYPES, Card, DeckEntry, DeckPlan, OpponentDeckSpec
from .search import SearchFilter, apply_filter
from .tcgp_client import FetchProgress, TCGPClient

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # rich handles its own colour for the user-facing output; the standard
    # logger is for diagnostics.


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="tcgp-deck-genie")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the on-disk cache directory (defaults to ~/.tcgp_deck_genie).",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, cache_dir: Path | None) -> None:
    """TCGPDeckGenie: build Pokémon TCG Pocket decks with LLM-assisted reasoning."""
    load_dotenv()
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["cache_dir"] = cache_dir or default_cache_dir()


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--set",
    "set_ids",
    multiple=True,
    help="Fetch only specific TCGP set ids (e.g. --set A1 --set A1a). Defaults to all sets.",
)
@click.option(
    "--concurrency", default=12, show_default=True, help="Parallel TCGdex requests."
)
@click.option(
    "--exclude-rares/--include-rares",
    "exclude_rares",
    default=True,
    show_default=True,
    help=(
        "Drop high-rarity printings (Star/Shiny/Crown) from the cached corpus. "
        "These are very hard to obtain in TCG Pocket, so the default is to "
        "exclude them. Use --include-rares only if you have the relevant rares."
    ),
)
@click.pass_context
def sync(
    ctx: click.Context,
    set_ids: tuple[str, ...],
    concurrency: int,
    exclude_rares: bool,
) -> None:
    """Fetch the TCG Pocket card corpus from TCGdex into the local cache.

    This is the only step that talks to TCGdex. Run it once after install,
    then again whenever a new set drops.
    """
    cache_dir: Path = ctx.obj["cache_dir"]
    client = TCGPClient(concurrency=concurrency)

    sets = list(set_ids) if set_ids else client.list_set_ids()
    mode = (
        "excluding Star/Shiny/Crown rares"
        if exclude_rares
        else "including every rarity"
    )
    console.print(
        f"[bold]Syncing {len(sets)} set(s)[/] ({mode}): {', '.join(sets)}"
    )

    all_cards: list[Card] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for set_id in sets:
            task_id = progress.add_task(f"Set {set_id}", total=None)

            def cb(p: FetchProgress, _task=task_id) -> None:
                progress.update(_task, total=p.total, completed=p.completed)

            cards = client.fetch_cards(
                set_ids=[set_id], progress=cb, exclude_rares=exclude_rares
            )
            all_cards.extend(cards)
            progress.update(task_id, description=f"Set {set_id} ({len(cards)} cards)")

    # Dedupe again across sets just to be defensive (some promos reuse ids).
    seen: dict[str, Card] = {}
    for c in all_cards:
        seen.setdefault(c.id, c)
    final = list(seen.values())

    corpus = Corpus(cards=final, sets_included=sets, fetched_at=time.time())
    out = save_corpus(corpus, cache_dir=cache_dir)
    console.print(
        f"[green]✓[/] Saved {len(final)} cards to [bold]{out}[/]"
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def _print_cache_section(title: str, summary: dict, count_label: str, count_key: str) -> None:
    table = Table(title=title, show_header=False)
    table.add_row("Path", str(summary["path"]))
    table.add_row(count_label, str(summary[count_key]))
    table.add_row("Sets", ", ".join(summary["sets_included"]))
    table.add_row(
        "Fetched at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary["fetched_at"] or 0)),
    )
    table.add_row("Schema version", str(summary["schema_version"]))
    console.print(table)


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show what's in the local card and mission caches."""
    cache_dir: Path = ctx.obj["cache_dir"]
    card_summary = corpus_info(cache_dir)
    mission_summary = missions_info(cache_dir)

    if card_summary:
        _print_cache_section("Card corpus", card_summary, "Cards", "card_count")
    else:
        console.print(
            f"[yellow]No card cache found at {corpus_path(cache_dir)}[/].\n"
            "Run [bold]tcgp-deck-genie sync[/] first."
        )

    console.print()

    if mission_summary:
        _print_cache_section("Mission decks", mission_summary, "Decks", "deck_count")
    else:
        console.print(
            f"[yellow]No mission cache found at {missions_path(cache_dir)}[/].\n"
            "Run [bold]tcgp-deck-genie sync-missions[/] first."
        )

    if card_summary is None:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# sync-missions
# ---------------------------------------------------------------------------


@main.command("sync-missions")
@click.option(
    "--set",
    "set_ids",
    multiple=True,
    help="Fetch only specific TCGP set ids (defaults to all expansions). Repeatable.",
)
@click.pass_context
def sync_missions(ctx: click.Context, set_ids: tuple[str, ...]) -> None:
    """Fetch solo-battle (mission) opponent decks from Bulbapedia into the cache.

    Like 'sync', this is a one-time network step; reuse the cache afterwards.
    Requires the card corpus ('sync') first, which it uses to resolve card ids.
    """
    cache_dir: Path = ctx.obj["cache_dir"]
    corpus = _load_or_fail(ctx)

    client = TCGPClient()
    all_sets = client.list_sets()  # [(id, name)] from TCGdex
    name_to_id = {name: sid for sid, name in all_sets}
    name_to_id.update(SET_NAME_ALIASES)
    valid_ids = {c.id for c in corpus.cards}

    # The promo set (P-A) has no solo-battle page of its own; skip it as a target
    # while still keeping it in name_to_id so promo cards inside decks resolve.
    targets = [
        (sid, name)
        for sid, name in all_sets
        if sid != "P-A" and (not set_ids or sid in set_ids)
    ]
    if not targets:
        console.print("[yellow]No matching expansions to fetch.[/]")
        raise SystemExit(1)

    console.print(f"[bold]Fetching solo battles for {len(targets)} expansion(s)…[/]")
    bulba = BulbapediaClient(name_to_id, valid_ids)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Expansions", total=len(targets))

        def cb(p: MissionFetchProgress, _task=task_id) -> None:
            progress.update(_task, total=p.total, completed=p.completed, description=p.set_name)

        decks, skipped = bulba.fetch_mission_decks(targets, progress=cb)

    corpus_obj = MissionCorpus(
        decks=decks,
        sets_included=[sid for sid, name in targets if name not in skipped],
        fetched_at=time.time(),
    )
    out = save_missions(corpus_obj, cache_dir=cache_dir)

    unresolved = sum(len(d.unresolved) for d in decks)
    console.print(f"[green]✓[/] Saved {len(decks)} mission deck(s) to [bold]{out}[/]")
    if skipped:
        console.print(
            f"[dim]No solo-battle page yet for: {', '.join(skipped)} (skipped).[/]"
        )
    if unresolved:
        console.print(
            f"[dim]{unresolved} card reference(s) could not be resolved to the "
            "current corpus (likely excluded rarities or un-synced sets).[/]"
        )


# ---------------------------------------------------------------------------
# missions
# ---------------------------------------------------------------------------


@main.command()
@click.option("--set", "set_id", default=None, help="Filter to a single set id.")
@click.option("--difficulty", default=None, help="Filter by difficulty (substring match).")
@click.argument("query", nargs=-1)
@click.pass_context
def missions(
    ctx: click.Context,
    set_id: str | None,
    difficulty: str | None,
    query: tuple[str, ...],
) -> None:
    """List solo-battle opponent decks, or show one in detail by name.

    With no NAME, lists matching missions. With a NAME, resolves it (fuzzily)
    and prints the full opposing deck.
    """
    corpus = _load_or_fail(ctx)
    mission_corpus = _load_missions_or_fail(ctx)
    by_id = corpus.by_id()

    if query:
        name = " ".join(query)
        try:
            deck = find_mission(
                mission_corpus.decks, name, set_id=set_id, difficulty=difficulty
            )
        except MissionLookupError as exc:
            console.print(f"[red]{exc}[/]")
            raise SystemExit(2) from exc
        _print_mission(deck, by_id)
        return

    decks = mission_corpus.decks
    if set_id:
        decks = [d for d in decks if d.set_id == set_id]
    if difficulty:
        dl = difficulty.lower()
        decks = [d for d in decks if d.difficulty and dl in d.difficulty.lower()]

    console.print(f"[bold]{len(decks)}[/] mission(s)")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Set")
    table.add_column("Difficulty")
    table.add_column("Cards", justify="right")
    table.add_column("Energy")
    for d in sorted(decks, key=lambda d: (d.set_id or "", d.difficulty or "", d.name)):
        table.add_row(
            d.name,
            d.set_id or "",
            d.difficulty or "",
            str(d.total_cards),
            ", ".join(d.energy_types) or "—",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@main.command()
@click.option("--energy", type=str, default=None, help="Filter Pokémon to this energy type.")
@click.option(
    "--set",
    "set_ids",
    multiple=True,
    help="Limit to specific set ids (repeatable).",
)
@click.option(
    "--keyword",
    "keywords",
    multiple=True,
    help="Substring(s) that must appear in card name/text (repeatable).",
)
@click.option(
    "--category",
    type=click.Choice(["Pokemon", "Trainer"], case_sensitive=False),
    default=None,
)
@click.option("--limit", default=30, show_default=True, help="Maximum rows to show.")
@click.option("--no-ex", is_flag=True, help="Exclude Pokémon ex.")
@click.pass_context
def search(
    ctx: click.Context,
    energy: str | None,
    set_ids: tuple[str, ...],
    keywords: tuple[str, ...],
    category: str | None,
    limit: int,
    no_ex: bool,
) -> None:
    """Search the local card corpus."""
    corpus = _load_or_fail(ctx)
    spec = SearchFilter(
        energy_type=energy,
        set_ids=set(set_ids) if set_ids else None,
        keywords=list(keywords),
        category=category.capitalize() if category else None,
        include_ex=not no_ex,
    )
    results = apply_filter(corpus.cards, spec)
    console.print(f"[bold]{len(results)}[/] match(es)")
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Cat")
    table.add_column("HP", justify="right")
    table.add_column("Type")
    table.add_column("Stage")
    table.add_column("Text", overflow="fold", max_width=60)
    for c in results[:limit]:
        text = c.effect or _summarise_attacks(c) or ""
        table.add_row(
            c.id,
            c.name + (" [ex]" if c.is_ex else ""),
            c.category[0],
            str(c.hp) if c.hp else "",
            c.primary_type() or "",
            c.stage or "",
            text,
        )
    console.print(table)
    if len(results) > limit:
        console.print(f"[dim]…{len(results) - limit} more not shown (use --limit).[/]")


def _summarise_attacks(card: Card) -> str:
    if not card.attacks:
        return ""
    parts = []
    for a in card.attacks:
        cost = "".join(c[0] for c in a.cost) if a.cost else "0"
        dmg = f" {a.damage}" if a.damage else ""
        parts.append(f"[{cost}]{a.name}{dmg}")
    return " / ".join(parts)


# ---------------------------------------------------------------------------
# build-deck
# ---------------------------------------------------------------------------


@main.command("build-deck")
@click.option(
    "--energy",
    default=None,
    help=(
        f"Single energy type: {', '.join(ENERGY_TYPES)}. "
        "Optional in counter mode (auto-picked from opponent weaknesses)."
    ),
)
@click.option(
    "--brief",
    default="Build a strong, fun deck.",
    show_default=True,
    help="Short brief that frames what kind of deck you want.",
)
@click.option(
    "--counter-mission",
    default=None,
    help="Name of a cached solo-battle mission whose opponent deck to counter.",
)
@click.option(
    "--counter-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to an opponent deck JSON (minimal card list or a saved --out file).",
)
@click.option("--set", "set_ids", multiple=True, help="Limit candidate pool to these sets.")
@click.option(
    "--keyword",
    "keywords",
    multiple=True,
    help="Force candidates to mention these substrings (repeatable).",
)
@click.option(
    "--must-include",
    "must_include",
    multiple=True,
    help="Card id(s) the deck must include (repeatable).",
)
@click.option("--no-ex", is_flag=True, help="Exclude Pokémon ex from the candidate pool.")
@click.option(
    "--no-shortlist",
    is_flag=True,
    help="Skip the cheap shortlist call (uses a single, larger reasoning call instead).",
)
@click.option(
    "--shortlist-size",
    default=40,
    show_default=True,
    help="How many cards the shortlist step keeps.",
)
@click.option(
    "--candidate-cap",
    default=120,
    show_default=True,
    help="Maximum candidates considered before the LLM stages.",
)
@click.option(
    "--thinking-budget",
    type=int,
    default=None,
    help="Override TCGP_THINKING_BUDGET (tokens). 0 disables thinking.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to save the deck JSON (defaults to printing only).",
)
@click.pass_context
def build_deck_cmd(
    ctx: click.Context,
    energy: str | None,
    brief: str,
    counter_mission: str | None,
    counter_file: Path | None,
    set_ids: tuple[str, ...],
    keywords: tuple[str, ...],
    must_include: tuple[str, ...],
    no_ex: bool,
    no_shortlist: bool,
    shortlist_size: int,
    candidate_cap: int,
    thinking_budget: int | None,
    out: Path | None,
) -> None:
    """Ask Gemini to design a 20-card TCG Pocket deck.

    In counter mode (--counter-mission or --counter-file) the deck is built to
    beat a specific opponent; --energy then becomes optional and is auto-picked
    from the opponent's weaknesses when omitted.
    """
    corpus = _load_or_fail(ctx)
    by_id = corpus.by_id()

    if counter_mission and counter_file:
        console.print("[red]Use only one of --counter-mission / --counter-file.[/]")
        raise SystemExit(2)

    opponent_summary: dict | None = None
    opponent_label: str | None = None
    counter_brief = brief
    if counter_mission or counter_file:
        opponent_cards, opponent_energy, opponent_label = _resolve_opponent(
            ctx, by_id, counter_mission, counter_file
        )
        opponent_summary = summarise_opponent(opponent_cards, opponent_energy)
        if not energy:
            energy = choose_counter_energy(opponent_summary)
            if not energy:
                console.print(
                    "[red]Could not auto-pick an energy type (opponent has no "
                    "recorded weaknesses). Re-run with --energy.[/]"
                )
                raise SystemExit(2)
            console.print(
                f"[dim]Auto-selected [bold]{energy}[/] energy "
                f"(exploits the most opponent weaknesses).[/]"
            )
        if brief == "Build a strong, fun deck.":
            counter_brief = f"Build a deck to beat the {opponent_label} deck."
    elif not energy:
        console.print(
            "[red]--energy is required unless you pass --counter-mission/--counter-file.[/]"
        )
        raise SystemExit(2)

    try:
        gemini = GeminiClient()
    except GeminiClientError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(2) from exc

    builder = DeckBuilder(corpus.cards, gemini)
    options = BuildOptions(
        energy_type=energy,
        user_brief=counter_brief,
        set_ids=set(set_ids) if set_ids else None,
        keywords=list(keywords),
        must_include_card_ids=list(must_include),
        include_ex=not no_ex,
        use_shortlist=not no_shortlist,
        shortlist_size=shortlist_size,
        preshortlist_cap=candidate_cap,
        thinking_budget=thinking_budget,
        opponent=opponent_summary,
        opponent_label=opponent_label,
    )

    status = (
        f"[bold]Designing {energy.capitalize()} counter to {opponent_label}…[/]"
        if opponent_label
        else f"[bold]Designing {energy.capitalize()} deck…[/]"
    )
    with console.status(status, spinner="dots"):
        try:
            result = builder.build(options)
        except (DeckBuildError, GeminiClientError) as exc:
            console.print(f"[red]Deck build failed:[/] {exc}")
            raise SystemExit(2) from exc

    _print_deck(result)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_deck_to_json(result), ensure_ascii=False, indent=2))
        console.print(f"[green]✓[/] Saved deck to [bold]{out}[/]")


def _parse_counter_deck(payload: dict) -> tuple[str | None, list[str] | None, list[DeckEntry]]:
    """Parse JSON for ``--counter-file``.

    Accepts either a minimal :class:`OpponentDeckSpec` (at the root or under
    ``deck``) or a full :class:`DeckPlan` saved via ``build-deck --out`` (detected
    by the presence of ``strategy``).
    """
    inner = payload.get("deck", payload)
    if "strategy" in inner:
        plan = DeckPlan.model_validate(inner)
        return plan.name, list(plan.energy_types), plan.cards
    spec = OpponentDeckSpec.model_validate(inner)
    return spec.name, spec.energy_types, spec.cards


def _resolve_opponent(
    ctx: click.Context,
    by_id: dict[str, Card],
    counter_mission: str | None,
    counter_file: Path | None,
) -> tuple[list[Card], list[str] | None, str]:
    """Resolve the opponent deck into (cards, energy_types, label)."""
    if counter_mission:
        mission_corpus = _load_missions_or_fail(ctx)
        try:
            deck = find_mission(mission_corpus.decks, counter_mission)
        except MissionLookupError as exc:
            console.print(f"[red]{exc}[/]")
            raise SystemExit(2) from exc
        cards = [by_id[cid] for cid in deck.resolved_card_ids() if cid in by_id]
        return cards, deck.energy_types, deck.name

    # counter_file: minimal opponent spec or a saved deck JSON from --out.
    path = Path(counter_file)
    payload = json.loads(path.read_text())
    try:
        name, energy_types, entries = _parse_counter_deck(payload)
    except ValidationError as exc:
        console.print(f"[red]Invalid counter deck file:[/] {exc}")
        raise SystemExit(2) from exc
    cards: list[Card] = []
    for entry in entries:
        card = by_id.get(entry.card_id)
        if card is not None:
            cards.extend([card] * entry.count)
    label = name or path.stem
    return cards, list(energy_types) if energy_types else None, label


# ---------------------------------------------------------------------------
# show-deck
# ---------------------------------------------------------------------------


@main.command("show-deck")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_deck_cmd(ctx: click.Context, path: Path) -> None:
    """Pretty-print a deck previously saved with --out."""
    corpus = _load_or_fail(ctx)
    payload = json.loads(path.read_text())
    deck = DeckPlan.model_validate(payload["deck"])
    by_id = corpus.by_id()
    used = [by_id[e.card_id] for e in deck.cards if e.card_id in by_id]
    warnings = payload.get("validation_warnings", [])
    result = BuildResult(
        deck=deck,
        cards_used=used,
        candidate_pool_size=int(payload.get("candidate_pool_size", 0)),
        shortlist_size=payload.get("shortlist_size"),
        validation_warnings=list(warnings),
        opponent_label=payload.get("opponent_label"),
    )
    _print_deck(result)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_or_fail(ctx: click.Context):
    cache_dir = ctx.obj["cache_dir"]
    corpus = load_corpus(cache_dir)
    if corpus is None:
        console.print(
            f"[yellow]No card cache found at {cache_dir / 'cards_tcgp.json'}[/].\n"
            "Run [bold]tcgp-deck-genie sync[/] first."
        )
        sys.exit(1)
    return corpus


def _load_missions_or_fail(ctx: click.Context) -> MissionCorpus:
    cache_dir = ctx.obj["cache_dir"]
    mc = load_missions(cache_dir)
    if mc is None:
        console.print(
            f"[yellow]No mission cache found at {cache_dir / 'missions_tcgp.json'}[/].\n"
            "Run [bold]tcgp-deck-genie sync-missions[/] first."
        )
        sys.exit(1)
    return mc


def _print_mission(deck: MissionDeck, by_id: dict[str, Card]) -> None:
    console.rule(
        f"[bold magenta]{deck.name}[/] · {deck.set_id or '?'} · "
        f"{deck.difficulty or '?'} · {deck.total_cards} cards"
    )
    if deck.energy_types:
        console.print(f"[dim]Energy: {', '.join(deck.energy_types)}[/]\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Count", justify="right")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Type")
    for c in deck.cards:
        card = by_id.get(c.card_id) if c.card_id else None
        cid = c.card_id or "[red]unresolved[/]"
        ctype = (card.primary_type() or "") if card else ""
        suffix = " [ex]" if card and card.is_ex else ""
        table.add_row(str(c.count), cid, c.name + suffix, ctype)
    console.print(table)
    if deck.unresolved:
        console.print(
            f"\n[dim]{len(deck.unresolved)} unresolved card(s): "
            f"{', '.join(deck.unresolved)}[/]"
        )


def _print_deck(result: BuildResult) -> None:
    deck = result.deck
    console.rule(f"[bold cyan]{deck.name}[/] · {', '.join(deck.energy_types)} · {deck.total_cards} cards")
    if result.opponent_label:
        console.print(f"[magenta]Counter to:[/] {result.opponent_label}")
    console.print(f"[italic]{deck.strategy}[/]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Count", justify="right")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Role", overflow="fold", max_width=40)
    used_by_id = {c.id: c for c in result.cards_used}
    for entry in deck.cards:
        c = used_by_id.get(entry.card_id)
        name = c.name + (" [ex]" if c and c.is_ex else "") if c else f"<unknown:{entry.card_id}>"
        table.add_row(str(entry.count), entry.card_id, name, entry.role or "")
    console.print(table)

    if deck.key_synergies:
        console.print("\n[bold]Key synergies[/]")
        for s in deck.key_synergies:
            console.print(f"  • {s}")
    if deck.standalone_value:
        console.print("\n[bold]Standalone value[/]")
        for s in deck.standalone_value:
            console.print(f"  • {s}")
    if deck.weaknesses:
        console.print("\n[bold]Weaknesses[/]")
        for s in deck.weaknesses:
            console.print(f"  • {s}")

    console.print(
        f"\n[dim]Candidate pool: {result.candidate_pool_size}"
        + (f" · shortlist: {result.shortlist_size}" if result.shortlist_size else "")
        + "[/]"
    )
    if result.validation_warnings:
        console.print("\n[bold yellow]Warnings[/]")
        for w in result.validation_warnings:
            console.print(f"  ⚠ {w}")


def _deck_to_json(result: BuildResult) -> dict:
    return {
        "deck": result.deck.model_dump(mode="json"),
        "candidate_pool_size": result.candidate_pool_size,
        "shortlist_size": result.shortlist_size,
        "validation_warnings": list(result.validation_warnings),
        "opponent_label": result.opponent_label,
    }


if __name__ == "__main__":
    main()

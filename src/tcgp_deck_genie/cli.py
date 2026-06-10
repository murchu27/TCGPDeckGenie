"""Click-based command-line interface for TCGPDeckGenie."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click
from dotenv import load_dotenv
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
from .cache import Corpus, corpus_info, default_cache_dir, load_corpus, save_corpus
from .deck_builder import BuildOptions, BuildResult, DeckBuilder, DeckBuildError
from .gemini_client import GeminiClient, GeminiClientError
from .models import ENERGY_TYPES, Card, DeckPlan
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
@click.pass_context
def sync(ctx: click.Context, set_ids: tuple[str, ...], concurrency: int) -> None:
    """Fetch the TCG Pocket card corpus from TCGdex into the local cache.

    This is the only step that talks to TCGdex. Run it once after install,
    then again whenever a new set drops.
    """
    cache_dir: Path = ctx.obj["cache_dir"]
    client = TCGPClient(concurrency=concurrency)

    sets = list(set_ids) if set_ids else client.list_set_ids()
    console.print(f"[bold]Syncing {len(sets)} set(s):[/] {', '.join(sets)}")

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

            cards = client.fetch_cards(set_ids=[set_id], progress=cb)
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


@main.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show what's in the local cache."""
    cache_dir: Path = ctx.obj["cache_dir"]
    summary = corpus_info(cache_dir)
    if summary is None:
        console.print(
            f"[yellow]No cache found at {cache_dir / 'cards_tcgp.json'}[/].\n"
            "Run [bold]tcgp-deck-genie sync[/] first."
        )
        raise SystemExit(1)
    table = Table(title="Cache summary", show_header=False)
    table.add_row("Path", str(summary["path"]))
    table.add_row("Cards", str(summary["card_count"]))
    table.add_row("Sets", ", ".join(summary["sets_included"]))
    table.add_row(
        "Fetched at",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary["fetched_at"] or 0)),
    )
    table.add_row("Schema version", str(summary["schema_version"]))
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
@click.option("--energy", required=True, help=f"Single energy type: {', '.join(ENERGY_TYPES)}.")
@click.option(
    "--brief",
    default="Build a strong, fun deck.",
    show_default=True,
    help="Short brief that frames what kind of deck you want.",
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
    energy: str,
    brief: str,
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
    """Ask Gemini to design a 20-card TCG Pocket deck."""
    corpus = _load_or_fail(ctx)

    try:
        gemini = GeminiClient()
    except GeminiClientError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(2) from exc

    builder = DeckBuilder(corpus.cards, gemini)
    options = BuildOptions(
        energy_type=energy,
        user_brief=brief,
        set_ids=set(set_ids) if set_ids else None,
        keywords=list(keywords),
        must_include_card_ids=list(must_include),
        include_ex=not no_ex,
        use_shortlist=not no_shortlist,
        shortlist_size=shortlist_size,
        preshortlist_cap=candidate_cap,
        thinking_budget=thinking_budget,
    )

    with console.status(
        f"[bold]Designing {energy.capitalize()} deck…[/]", spinner="dots"
    ):
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
    by_id = {c.id: c for c in corpus.cards}
    used = [by_id[e.card_id] for e in deck.cards if e.card_id in by_id]
    warnings = payload.get("validation_warnings", [])
    result = BuildResult(
        deck=deck,
        cards_used=used,
        candidate_pool_size=int(payload.get("candidate_pool_size", 0)),
        shortlist_size=payload.get("shortlist_size"),
        validation_warnings=list(warnings),
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


def _print_deck(result: BuildResult) -> None:
    deck = result.deck
    console.rule(f"[bold cyan]{deck.name}[/] · {', '.join(deck.energy_types)} · {deck.total_cards} cards")
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
    }


if __name__ == "__main__":
    main()

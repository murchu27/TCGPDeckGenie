"""Shared fixtures - in particular, a small fake card corpus."""
from __future__ import annotations

import pytest

from tcgp_deck_genie.models import Ability, Attack, Card


def _basic(name: str, set_id: str, **kw) -> Card:
    return Card(
        id=f"{set_id}-{name[:3].upper()}",
        name=name,
        set_id=set_id,
        category="Pokemon",
        hp=kw.get("hp", 70),
        types=[kw.get("ttype", "Water")],
        stage="Basic",
        attacks=kw.get("attacks", []),
        abilities=kw.get("abilities", []),
        weaknesses=kw.get("weaknesses", []),
        retreat=kw.get("retreat", 1),
    )


@pytest.fixture
def fake_corpus() -> list[Card]:
    """A small but interesting fake corpus that exercises real-world cases."""
    return [
        # Water main attacker w/ explicit synergy hook ("flip a coin")
        Card(
            id="A1-079",
            name="Lapras",
            set_id="A1",
            category="Pokemon",
            hp=140,
            types=["Water"],
            stage="Basic",
            retreat=3,
            attacks=[
                Attack(
                    name="Hydro Pump",
                    cost=["Water", "Water", "Colorless"],
                    damage="80",
                    effect="If you have at least 2 extra W energy, this attack does 40 more damage.",
                ),
            ],
        ),
        # Water Stage 2 line - Squirtle/Wartortle/Blastoise ex
        _basic(
            "Squirtle",
            "A1",
            attacks=[Attack(name="Water Gun", cost=["Water"], damage="20")],
        ),
        Card(
            id="A1-054",
            name="Wartortle",
            set_id="A1",
            category="Pokemon",
            hp=80,
            types=["Water"],
            stage="Stage1",
            evolve_from="Squirtle",
            retreat=1,
            attacks=[Attack(name="Surf", cost=["Water", "Water"], damage="40")],
        ),
        Card(
            id="A1-056",
            name="Blastoise ex",
            set_id="A1",
            category="Pokemon",
            hp=180,
            types=["Water"],
            stage="Stage2",
            evolve_from="Wartortle",
            suffix="EX",
            retreat=3,
            attacks=[
                Attack(name="Hydro Bazooka", cost=["Water", "Water", "Water"], damage="100"),
            ],
        ),
        # Coin-flip energy accel that pairs with Lapras
        Card(
            id="A1-220",
            name="Misty",
            set_id="A1",
            category="Trainer",
            trainer_type="Supporter",
            effect=(
                "Choose 1 of your Water Pokémon. Flip a coin until you get tails; "
                "attach a W Energy to it for each heads."
            ),
        ),
        # Generic damage-buff Supporter - mostly standalone utility
        Card(
            id="A1-223",
            name="Giovanni",
            set_id="A1",
            category="Trainer",
            trainer_type="Supporter",
            effect="During this turn, attacks used by your Pokémon do +10 damage to your opponent's Active Pokémon.",
        ),
        # Item draw Trainer - standalone utility
        Card(
            id="P-A-007",
            name="Professor's Research",
            set_id="P-A",
            category="Trainer",
            trainer_type="Supporter",
            effect="Draw 2 cards.",
        ),
        # Off-type Pokémon to make sure energy filter rejects it
        Card(
            id="A1-094",
            name="Pikachu",
            set_id="A1",
            category="Pokemon",
            hp=60,
            types=["Lightning"],
            stage="Basic",
            retreat=1,
            attacks=[Attack(name="Quick Attack", cost=["Lightning"], damage="10")],
        ),
        # Pokémon with ability we can search by keyword
        Card(
            id="A1-101",
            name="Articuno ex",
            set_id="A1",
            category="Pokemon",
            hp=140,
            types=["Water"],
            stage="Basic",
            suffix="EX",
            retreat=2,
            attacks=[Attack(name="Blizzard", cost=["Water", "Water", "Water"], damage="80")],
            abilities=[Ability(name="Frost Bind", effect="Once during your turn, may switch your active Pokémon.")],
        ),
        # Off-type evolution line: Vaporeon (Water) evolves from Eevee, but Eevee
        # is a Colorless Basic - so an `--energy Water` filter strips Eevee out of
        # the pool, making the line impossible unless evolution-support re-adds it.
        Card(
            id="A1-206",
            name="Eevee",
            set_id="A1",
            category="Pokemon",
            hp=70,
            types=["Colorless"],
            stage="Basic",
            retreat=1,
            attacks=[Attack(name="Tackle", cost=["Colorless"], damage="20")],
        ),
        Card(
            id="A1-080",
            name="Vaporeon",
            set_id="A1",
            category="Pokemon",
            hp=110,
            types=["Water"],
            stage="Stage1",
            evolve_from="Eevee",
            retreat=1,
            attacks=[Attack(name="Bubble Drain", cost=["Water", "Water"], damage="60")],
        ),
    ]

"""Pydantic models for the cards and decks we care about.

Two layers:

* ``Card`` / ``Attack`` / ``Ability`` mirror the slim subset of TCGdex fields we
  actually need for deck building. We deliberately drop illustrator, image URLs,
  pricing, variant metadata, etc. - these add tokens to every LLM prompt without
  improving the deck.
* ``DeckEntry`` / ``DeckPlan`` describe the structured JSON we ask Gemini to
  produce. Keeping them as Pydantic models lets us hand the schema straight to
  ``response_json_schema`` and validate the response in one step.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid Pokémon TCG energy / type strings used in TCG Pocket.
ENERGY_TYPES: tuple[str, ...] = (
    "Grass",
    "Fire",
    "Water",
    "Lightning",
    "Psychic",
    "Fighting",
    "Darkness",
    "Metal",
    "Dragon",
    "Colorless",
)

TRAINER_TYPES: tuple[str, ...] = ("Item", "Supporter", "Tool", "Stadium")

# Stages we see on actual Pokémon cards in TCGP.
STAGES: tuple[str, ...] = ("Basic", "Stage1", "Stage2")


class Attack(BaseModel):
    """A single attack on a Pokémon card."""

    name: str
    cost: list[str] = Field(default_factory=list)
    damage: str | None = None
    effect: str | None = None

    @property
    def energy_cost_total(self) -> int:
        return len(self.cost)


class Ability(BaseModel):
    """A passive ability on a Pokémon card."""

    name: str
    type: str | None = None
    effect: str | None = None


class Card(BaseModel):
    """Slim view of a TCGdex card record used throughout the app."""

    id: str
    name: str
    set_id: str
    category: Literal["Pokemon", "Trainer", "Energy"]

    # Pokémon-only fields
    hp: int | None = None
    types: list[str] = Field(default_factory=list)
    stage: str | None = None
    evolve_from: str | None = None
    suffix: str | None = None  # e.g. "EX"
    attacks: list[Attack] = Field(default_factory=list)
    abilities: list[Ability] = Field(default_factory=list)
    weaknesses: list[dict] = Field(default_factory=list)
    retreat: int | None = None

    # Trainer-only fields
    trainer_type: str | None = None
    effect: str | None = None  # supporter/item effect

    rarity: str | None = None

    @property
    def is_pokemon(self) -> bool:
        return self.category == "Pokemon"

    @property
    def is_trainer(self) -> bool:
        return self.category == "Trainer"

    @property
    def is_ex(self) -> bool:
        return (self.suffix or "").upper() == "EX"

    @property
    def is_basic(self) -> bool:
        return self.is_pokemon and (self.stage == "Basic")

    @property
    def ko_points(self) -> int:
        """Prize points the opponent scores for knocking this Pokémon out.

        TCG Pocket is won at 3 points. A KO is worth:
          * 3 (an instant loss) for a Pokémon with "Mega" in its name,
          * 2 for any other 4-diamond Pokémon (which includes essentially all
            ``ex`` cards), and
          * 1 for everything else.

        Non-Pokémon cards are never knocked out, so they score 0. We treat the
        ``ex`` suffix as 2-point even when the rarity string is missing (some
        promos lack a rarity) because ``ex`` always give up 2 points in-game.
        """
        if not self.is_pokemon:
            return 0
        if "mega" in self.name.lower():
            return 3
        if self.is_ex or (self.rarity and "four diamond" in self.rarity.lower()):
            return 2
        return 1

    def primary_type(self) -> str | None:
        return self.types[0] if self.types else None

    def compact_dict(self) -> dict:
        """A compact dict representation used when embedding cards in LLM prompts.

        We drop ``None`` and empty collections so a typical card fits in
        ~150 tokens instead of ~500.
        """
        out: dict = {"id": self.id, "name": self.name, "cat": self.category[0]}
        if self.is_pokemon:
            out.update(
                {
                    "hp": self.hp,
                    "type": self.primary_type(),
                    "stage": self.stage,
                    # Prize points the opponent scores for KO'ing this Pokémon.
                    "kopts": self.ko_points,
                }
            )
            if self.evolve_from:
                out["from"] = self.evolve_from
            if self.suffix:
                out["suffix"] = self.suffix
            if self.retreat is not None:
                out["retreat"] = self.retreat
            if self.attacks:
                out["attacks"] = [
                    {
                        "n": a.name,
                        "cost": "".join(c[0] for c in a.cost) if a.cost else "",
                        "dmg": a.damage,
                        **({"fx": a.effect} if a.effect else {}),
                    }
                    for a in self.attacks
                ]
            if self.abilities:
                out["abilities"] = [
                    {"n": ab.name, **({"fx": ab.effect} if ab.effect else {})}
                    for ab in self.abilities
                ]
            if self.weaknesses:
                out["weak"] = [w.get("type") for w in self.weaknesses if w.get("type")]
        else:
            out["ttype"] = self.trainer_type
            if self.effect:
                out["fx"] = self.effect
        return out


class DeckEntry(BaseModel):
    """One line of the final deck: a card and how many copies to include."""

    card_id: str = Field(..., description="TCGdex card id, e.g. 'A1-036'.")
    count: int = Field(..., ge=1, le=2, description="Number of copies (max 2 per name).")
    role: str | None = Field(
        None,
        description="Short role label assigned by the model (e.g. 'main attacker', 'draw support').",
    )


class DeckPlan(BaseModel):
    """Final deck object returned by the LLM and persisted to disk."""

    name: str = Field(..., description="Short, evocative deck name.")
    energy_types: list[str] = Field(
        ..., min_length=1, max_length=3, description="The 1-3 energy types this deck runs."
    )
    cards: list[DeckEntry] = Field(..., description="The cards in the deck, max 20 total.")
    strategy: str = Field(..., description="2-4 sentences describing the deck's game plan.")
    key_synergies: list[str] = Field(
        default_factory=list,
        description="Bullet-point notes on synergies between specific cards in the deck.",
    )
    standalone_value: list[str] = Field(
        default_factory=list,
        description="Bullet-point notes on cards selected primarily for their solo utility.",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Bullet-point notes on matchups or situations this deck struggles with.",
    )

    @field_validator("energy_types")
    @classmethod
    def _validate_energy(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if t not in ENERGY_TYPES]
        if bad:
            raise ValueError(f"Unknown energy type(s): {bad}; allowed={ENERGY_TYPES}")
        return v

    @property
    def total_cards(self) -> int:
        return sum(e.count for e in self.cards)

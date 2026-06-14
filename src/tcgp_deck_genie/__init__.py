"""TCGPDeckGenie - design Pokémon TCG Pocket decks with LLM-assisted reasoning."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tcgp-deck-genie")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]

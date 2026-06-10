"""Thin wrapper around the Google Gen AI SDK.

This module intentionally hides almost nothing about Gemini - we expose two
methods (``shortlist`` and ``build_deck``) so the rest of the codebase doesn't
need to know about ``response_json_schema`` or ``ThinkingConfig``. That makes
it easy to drop in an alternative backend (local Ollama, OpenAI, etc.) later:
implement the same two methods and the deck builder won't know the difference.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import DeckPlan

logger = logging.getLogger(__name__)


DEFAULT_REASONING_MODEL = "gemini-2.5-flash"
DEFAULT_SHORTLIST_MODEL = "gemini-2.5-flash-lite"
DEFAULT_THINKING_BUDGET = 2048


class ShortlistResponse(BaseModel):
    """JSON schema for the shortlist step."""

    card_ids: list[str] = Field(..., description="Selected TCGdex card ids.")
    reasoning: str = Field(
        "",
        description="One short paragraph explaining why these cards were chosen as a group.",
    )


@dataclass
class GeminiConfig:
    api_key: str | None = None
    reasoning_model: str = DEFAULT_REASONING_MODEL
    shortlist_model: str | None = DEFAULT_SHORTLIST_MODEL
    thinking_budget: int = DEFAULT_THINKING_BUDGET

    @classmethod
    def from_env(cls) -> GeminiConfig:
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            reasoning_model=os.environ.get(
                "TCGP_GEMINI_REASONING_MODEL", DEFAULT_REASONING_MODEL
            ),
            shortlist_model=os.environ.get(
                "TCGP_GEMINI_SHORTLIST_MODEL", DEFAULT_SHORTLIST_MODEL
            )
            or None,
            thinking_budget=int(
                os.environ.get("TCGP_THINKING_BUDGET", str(DEFAULT_THINKING_BUDGET))
            ),
        )


class GeminiClientError(RuntimeError):
    """Raised when Gemini returns something we can't make use of."""


class GeminiClient:
    """Concrete Gemini backend used by ``DeckBuilder``."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig.from_env()
        if not self.config.api_key:
            raise GeminiClientError(
                "GEMINI_API_KEY is not set. Get a free key from "
                "https://aistudio.google.com/apikey and add it to your .env file."
            )
        # Imported lazily so unit tests that never touch Gemini can run without
        # the SDK installed (and so an import-time failure is reported clearly).
        from google import genai  # noqa: WPS433

        self._genai = genai
        self._client = genai.Client(api_key=self.config.api_key)

    # -- shortlist ----------------------------------------------------------------

    def shortlist(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ShortlistResponse:
        """Cheap pre-filter call - returns the model's chosen subset."""
        if not self.config.shortlist_model:
            raise GeminiClientError("Shortlist step requested but no shortlist model configured.")
        return self._generate_structured(
            model=self.config.shortlist_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=ShortlistResponse,
            thinking_budget=0,  # cheap pre-filter - no thinking
        )

    # -- deck building ------------------------------------------------------------

    def build_deck(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        thinking_budget: int | None = None,
    ) -> DeckPlan:
        """Reasoning pass that returns the final structured deck plan."""
        return self._generate_structured(
            model=self.config.reasoning_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=DeckPlan,
            thinking_budget=(
                thinking_budget if thinking_budget is not None else self.config.thinking_budget
            ),
        )

    # -- internals ----------------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(GeminiClientError),
    )
    def _generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[BaseModel],
        thinking_budget: int,
    ):
        types = self._genai.types
        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": schema_model.model_json_schema(),
        }
        # Thinking config is only meaningful on models that support it; the SDK
        # silently ignores it elsewhere, but we still gate the kwarg so older
        # SDK versions don't choke.
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        logger.debug("Calling Gemini model=%s thinking_budget=%s", model, thinking_budget)
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # network / SDK errors are wrapped for retry
            raise GeminiClientError(f"Gemini request failed: {exc}") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GeminiClientError("Gemini returned an empty response.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiClientError(
                f"Gemini response was not valid JSON: {exc}; raw={text[:400]!r}"
            ) from exc

        try:
            return schema_model.model_validate(payload)
        except Exception as exc:
            raise GeminiClientError(
                f"Gemini response did not match {schema_model.__name__} schema: {exc}; "
                f"payload={payload!r}"
            ) from exc

"""Price tables: USD per 1M tokens, so cost (and --max-cost) works for providers that report none.

``prices.json`` next to this file is the bundled table (regenerate it with ``scripts/update_prices.py``);
``--prices FILE`` entries override or extend it. A price is ``{"input", "output", "cache_read",
"cache_write"}``; the old ``{"input", "output"}`` (or ``[input, output]``) form still works, with cache
reads at 0.1x and cache writes at 1.25x the input price (Anthropic's rates).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from brass_lantern.providers import ConfigError, parse_spec

BUNDLED_PRICES = Path(__file__).with_name("prices.json")
CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


class Price(NamedTuple):
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


def make_price(input: float, output: float, cache_read: float | None = None, cache_write: float | None = None) -> Price:
    return Price(
        float(input),
        float(output),
        float(input) * CACHE_READ_FACTOR if cache_read is None else float(cache_read),
        float(input) * CACHE_WRITE_FACTOR if cache_write is None else float(cache_write),
    )


def as_price(value: Any) -> Price | None:
    """A Price from ``None``, a Price, ``(input, output)``, ``(input, output, cache_read, cache_write)`` or a dict.

    Raises ValueError, TypeError or KeyError on anything else, and on negative or non-finite numbers.
    """
    if value is None or isinstance(value, Price):
        return value
    if isinstance(value, Mapping):
        parts = [value["input"], value["output"], value.get("cache_read"), value.get("cache_write")]
    elif isinstance(value, (list, tuple)) and len(value) in (2, 4):
        parts = list(value) + [None] * (4 - len(value))
    else:
        raise ValueError(f"expected {{input, output[, cache_read, cache_write]}} or a list of 2 or 4 numbers, got {value!r}")
    price = make_price(*parts)
    if any(not math.isfinite(x) or x < 0 for x in price):
        raise ValueError(f"prices must be finite and not negative: {value!r}")
    return price


def parse_prices(raw: Any, where: str) -> dict[str, Price]:
    """``{"provider:model": price}``; keys starting with ``_`` (metadata) are skipped."""
    if not isinstance(raw, dict):
        raise ConfigError(f"Cannot read prices from {where}: expected a JSON object of model -> price.")
    out: dict[str, Price] = {}
    for spec, value in raw.items():
        if str(spec).startswith("_"):
            continue
        try:
            out[spec] = as_price(value)  # type: ignore[assignment]
        except (ValueError, KeyError, TypeError) as e:
            raise ConfigError(f"Cannot read prices from {where}: bad entry for {spec!r}: {e}") from e
    return out


def load_price_file(path: str | Path) -> dict[str, Price]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"Cannot read prices file {str(path)!r}: {e}") from e
    return parse_prices(raw, repr(str(path)))


def load_prices(path: str | None = None) -> dict[str, Price]:
    """The bundled table, overridden or extended by the entries of *path* (``--prices FILE``)."""
    table = load_price_file(BUNDLED_PRICES) if BUNDLED_PRICES.is_file() else {}
    if path:
        table.update(load_price_file(path))
    return table


def price_for(prices: Mapping[str, Price], spec: str) -> Price | None:
    """The price of *spec*: an entry for the exact spec wins, else the base model's (no ``@effort``)."""
    if spec in prices:
        return prices[spec]
    try:
        return prices.get(parse_spec(spec).base)
    except ConfigError:  # not a provider:model spec at all: nothing to look up
        return None


def prices_for_models(table: Mapping[str, Price], models: Iterable[str]) -> dict[str, Price]:
    """The table entries that apply to *models*, keyed by spec as given (what bench.json stores)."""
    out: dict[str, Price] = {}
    for spec in models:
        price = price_for(table, spec)
        if price is not None:
            out[spec] = price
    return out

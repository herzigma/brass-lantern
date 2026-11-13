#!/usr/bin/env python
"""Regenerate brass_lantern/prices.json from OpenRouter's public model list (free, no API key).

    python scripts/update_prices.py            # fetch and write brass_lantern/prices.json
    python scripts/update_prices.py --from-file models.json --out somewhere.json

Prices are USD per 1M tokens (OpenRouter lists $/token). Only current Claude and OpenAI chat models are
kept, under the spec you would give Brass Lantern:

    anthropic/claude-sonnet-5.5 -> anthropic:claude-sonnet-5-5   (dots in the version become dashes)
    openai/gpt-6.1-sol          -> openai:gpt-6.1-sol

`:batch` (and any other `:variant`) entries are skipped, as are image, audio, codex, open-weight and
other non-chat OpenAI models. Where OpenRouter lists no cache price, the cache price is the input price
(no discount, no surcharge). Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "brass_lantern" / "prices.json"

CLAUDE = re.compile(r"^claude-(?:opus|sonnet|haiku|fable|mythos)-\d")
OPENAI = re.compile(r"^gpt-[56]")
OPENAI_SKIP = re.compile(r"image|audio|codex|oss|safeguard|realtime|transcribe|tts|search")


def spec_for(model_id: str) -> str | None:
    """The Brass Lantern spec for an OpenRouter model id, or None to skip it."""
    vendor, _, name = model_id.partition("/")
    if ":" in name:
        return None
    if vendor == "anthropic" and CLAUDE.match(name):
        return "anthropic:" + name.replace(".", "-")
    if vendor == "openai" and OPENAI.match(name) and not OPENAI_SKIP.search(name):
        return "openai:" + name
    return None


def per_million(value) -> float | None:
    """OpenRouter's $/token string as USD per 1M tokens (exact decimal arithmetic)."""
    if value in (None, ""):
        return None
    number = Decimal(str(value)) * 1_000_000
    return None if number < 0 else float(number.quantize(Decimal("0.000001")).normalize())


def build(models: list[dict]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for m in models:
        spec = spec_for(m.get("id", ""))
        pricing = m.get("pricing") or {}
        if not spec or "text" not in (m.get("architecture") or {}).get("output_modalities", ["text"]):
            continue
        inp, out = per_million(pricing.get("prompt")), per_million(pricing.get("completion"))
        if inp is None or out is None:
            continue
        table[spec] = {
            "input": inp,
            "output": out,
            "cache_read": per_million(pricing.get("input_cache_read")) if pricing.get("input_cache_read") else inp,
            "cache_write": per_million(pricing.get("input_cache_write")) if pricing.get("input_cache_write") else inp,
        }
    return dict(sorted(table.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output file (default: {DEFAULT_OUT})")
    ap.add_argument("--from-file", type=Path, help="read a saved OpenRouter model list instead of fetching it")
    args = ap.parse_args()
    if args.from_file:
        data = json.loads(args.from_file.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            data = json.load(resp)
    table = build(data["data"])
    if not table:
        print("error: no Claude or OpenAI chat models found in the model list; nothing written", file=sys.stderr)
        return 1
    out = {
        "_meta": {
            "source": URL,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unit": "USD per 1M tokens",
        },
        **table,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(table)} models to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

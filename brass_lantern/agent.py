"""The agent: turns Grue observations into prompts and model replies into commands."""

from __future__ import annotations

import re
from dataclasses import dataclass

from brass_lantern.grue_client import Observation
from brass_lantern.providers import Completion, Provider, ProviderError

MAX_COMMAND_CHARS = 120
MAX_NOTES_CHARS = 1500
MAX_REASONING_CHARS = 300  # reasoning kept in the compact assistant history

SYSTEM_PROMPT = """\
You are playing a classic text adventure (interactive fiction) through a text parser. \
Your goal is to score as many points as you can and, if possible, finish the game.

HOW IT WORKS
- Each turn you see what the game printed in response to your last command. You then give ONE new command.
- Commands are short imperative phrases: a verb plus optional nouns. Examples: look, inventory (i), \
examine lamp (x lamp), take all, take sword, drop sword, open door, unlock door with key, \
put lamp in case, read leaflet, push button, turn on lamp, ask troll about sword, go north.
- Directions: north, south, east, west, northeast, northwest, southeast, southwest, up, down, in, out \
(abbreviations n, s, e, w, ne, nw, se, sw, u, d). Typing a bare direction moves you.
- The parser is simple. Do not write sentences, do not combine actions ("take lamp and go north" will fail), \
and do not chat with the game. If it says it does not understand a word, try a simpler or different word.
- Never use quit, restart, restore or save: they end or disrupt the run.

STRATEGY
- Explore systematically. Examine interesting objects, read signs/books/notes, open containers, \
and take anything portable. Check your inventory now and then.
- Keep a mental map: note exits you have not tried yet and where items and locked things are.
- Points usually come from finding treasures, solving puzzles, and reaching new places. \
If a command fails, do not repeat it; try something different. Be careful around dangers.

RESPONSE FORMAT
- You may write brief reasoning first (a few sentences at most).
- Your reply MUST end with a final line of the form:
> COMMAND
  for example:
> open mailbox
- Exactly one command after the ">". Nothing after that line.\
"""

NOTES_ADDENDUM = """

NOTES
- You can only see the most recent turns of the game, so keep a short notes scratchpad. \
To update it, include a line starting with NOTES: (it may continue over a few following lines). \
It REPLACES your previous notes, so rewrite everything worth keeping: map, items, puzzles, goals. \
Keep it under about 1500 characters. Your current notes are shown to you every turn.
- Put NOTES before the final "> COMMAND" line."""

CHAR_WARNING = (
    "[The game is waiting for a SINGLE KEYPRESS, not a typed command. Reply with one key: a single "
    "character (for example y, n, 1, q) or a key name: return, space, escape, up, down, left, right, "
    "pageup, pagedown, tab, backspace, delete.]"
)


# ---------------------------------------------------------------------------
# Parsing model replies
# ---------------------------------------------------------------------------

_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"^(?:command|action)\s*:\s*(.+)$", re.IGNORECASE)
_NOTES_LINE = re.compile(r"^notes\s*:\s*(.*)$", re.IGNORECASE)
_SEPARATOR = re.compile(r"\s*;\s*|\s*,\s*(?:and\s+)?(?:then\s+)?|\s+(?:and\s+)?then\s+", re.IGNORECASE)
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


@dataclass
class ParsedCommand:
    command: str
    method: str  # "prompt" (> CMD) | "tag" (COMMAND: CMD) | "last_line" | "repair" | "empty"

    @property
    def ok(self) -> bool:
        """False when the reply had no explicit command marker (a format failure)."""
        return self.method in ("prompt", "tag")


def _core(line: str) -> str:
    """A line with markdown emphasis/backticks removed, for marker detection."""
    return line.replace("*", "").replace("`", "").translate(_QUOTE_MAP).strip()


def _first_command(s: str) -> str:
    """Keep only the first of several commands joined by then / , / ; (outside double quotes)."""
    in_quote = False
    for i, ch in enumerate(s):
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote and i > 0:
            m = _SEPARATOR.match(s, i)
            if m and m.end() > i:
                return s[:i]
    return s


def _strip_quotes(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def clean_command(raw: str) -> str:
    s = _strip_quotes(raw.translate(_QUOTE_MAP).replace("`", "").replace("*", ""))
    s = s.lstrip("> ").strip()
    s = _first_command(s)
    s = " ".join(s.split())
    trimmed = s.rstrip(".!").rstrip()
    s = _strip_quotes(trimmed or s)
    return s[:MAX_COMMAND_CHARS].rstrip()


_ECHOED_FOOTER = re.compile(r"^\s*\[Turn \d+", re.MULTILINE)


def strip_hallucinated_turns(text: str) -> str:
    """Cut a reply where the model starts inventing the next game turn.

    Some models keep writing after their command: a made-up game response, our status
    footer ("[Turn 21 of 150 | ...]"), then more commands. Everything from the first
    echoed footer on is not the model's move, so it is ignored when parsing.
    """
    m = _ECHOED_FOOTER.search(text)
    return text[: m.start()] if m else text


def parse_command(text: str) -> ParsedCommand:
    """Extract the game command from a model reply.

    Priority: last line starting with ``>``; else last ``COMMAND:`` line; else the last
    non-empty line. Markdown/quotes are stripped, whitespace collapsed, extra chained
    commands dropped, and the result truncated to 120 characters.
    """
    prompt_hit = tag_hit = last = None
    for line in strip_hallucinated_turns(_THINK.sub("", text or "")).splitlines():
        core = _core(line)
        if not core:
            continue
        last = core
        if core.startswith(">"):
            body = core.lstrip("> ").strip()
            if body and body.upper() != "COMMAND":  # ignore an echoed "> COMMAND" template
                prompt_hit = body
        else:
            m = _TAG.match(core)
            if m:
                tag_hit = m.group(1)
    if prompt_hit:
        return ParsedCommand(clean_command(prompt_hit), "prompt")
    if tag_hit:
        return ParsedCommand(clean_command(tag_hit), "tag")
    if last:
        return ParsedCommand(clean_command(last), "last_line")
    return ParsedCommand("", "empty")


def _notes_blocks(lines: list[str]):
    """Yield (start, end, text) for each NOTES: block (a NOTES: line plus continuation lines)."""
    i = 0
    while i < len(lines):
        m = _NOTES_LINE.match(_core(lines[i]))
        if not m:
            i += 1
            continue
        parts = [m.group(1).strip()]
        j = i + 1
        while j < len(lines):
            c = _core(lines[j])
            if not c or c.startswith(">") or _TAG.match(c) or _NOTES_LINE.match(c):
                break
            parts.append(c)
            j += 1
        yield i, j, "\n".join(p for p in parts if p)
        i = j


def extract_notes(text: str) -> str | None:
    """Return the last NOTES: block of a reply (capped), or None."""
    lines = _THINK.sub("", text or "").splitlines()
    blocks = [t for _, _, t in _notes_blocks(lines) if t]
    return blocks[-1][:MAX_NOTES_CHARS] if blocks else None


def compact_reply(text: str, command: str, parsed: ParsedCommand | None = None) -> str:
    """Short form of a reply for the history: trimmed reasoning plus the command."""
    lines = _THINK.sub("", text or "").splitlines()
    skip: set[int] = set()
    for start, end, _ in _notes_blocks(lines):
        skip.update(range(start, end))
    kept = []
    for i, line in enumerate(lines):
        core = _core(line)
        if i in skip or not core or core.startswith(">") or _TAG.match(core):
            continue
        kept.append(core)
    if parsed is not None and parsed.method == "last_line" and kept:
        kept.pop()  # that line *was* the command
    reasoning = " ".join(" ".join(kept).split())
    if len(reasoning) > MAX_REASONING_CHARS:
        reasoning = reasoning[:MAX_REASONING_CHARS].rstrip() + "..."
    return f"{reasoning}\n> {command}" if reasoning else f"> {command}"



REPAIR_PROMPT = (
    "Your reply did not end with a command line. Reply with ONLY the single game command you "
    "want to send, on one line starting with \">\", for example:\n> north"
)

_SENTENCE = re.compile(r"[.!?]\s+\S|[.!?]$")


def looks_like_prose(command: str) -> bool:
    """True when a fallback 'command' is really a sentence of reasoning, not a game command."""
    words = command.split()
    if not words:
        return True
    return len(words) > 8 or (len(words) > 3 and bool(_SENTENCE.search(command)))


def _raw_last_line(text: str) -> str:
    """The last non-empty line of a reply, before any command clean-up."""
    lines = [_core(l) for l in _THINK.sub("", text or "").splitlines()]
    lines = [l for l in lines if l]
    return lines[-1] if lines else ""


def _merge(a: Completion, b: Completion) -> Completion:
    """Combine the accounting of a reply and its format-repair follow-up."""
    cost = None if a.cost_usd is None and b.cost_usd is None else (a.cost_usd or 0) + (b.cost_usd or 0)
    return Completion(
        text=a.text + "\n\n[format repair]\n" + b.text,
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=cost,
        latency_s=a.latency_s + b.latency_s,
        finish_reason=b.finish_reason,
    )

# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    command: str  # what is sent to the game
    reply: str  # the full raw model reply
    completion: Completion
    parsed: ParsedCommand


@dataclass
class _Turn:
    user: str
    assistant: str


class Agent:
    """Holds the rolling context for one episode and asks the model for each move."""

    def __init__(
        self,
        provider: Provider,
        *,
        max_turns: int | None = None,
        history: int = 30,
        notes: bool = False,
    ):
        self.provider = provider
        self.max_turns = max_turns
        self.history = max(0, history)
        self.notes_enabled = notes
        self.notes = ""
        self.system_prompt = SYSTEM_PROMPT + (NOTES_ADDENDUM if notes else "")
        self._turns: list[_Turn] = []

    # -- prompt construction ----------------------------------------------

    def render_observation(self, obs: Observation) -> str:
        """Game text plus a compact status footer (and a char-mode warning if needed)."""
        turn = f"Turn {obs.turn}" + (f" of {self.max_turns}" if self.max_turns else "")
        bits = [turn]
        if obs.score is not None:
            bits.append(f"Score {obs.score}" + (f" of {obs.max_score}" if obs.max_score else ""))
        if obs.location:
            bits.append(f"Location: {obs.location}")
        out = [obs.text.strip() or "(the game printed nothing)", "", "[" + " | ".join(bits) + "]"]
        if obs.notice:
            out.append(f"[Server notice: {obs.notice}]")
        if obs.input_type == "char":
            out.append(CHAR_WARNING)
        return "\n".join(out)

    def build_messages(self, obs: Observation) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for t in self._turns[-self.history :] if self.history else []:
            messages.append({"role": "user", "content": t.user})
            messages.append({"role": "assistant", "content": t.assistant})
        current = self.render_observation(obs)
        if self.notes_enabled:
            current += "\n\n[Your notes]\n" + (self.notes or "(empty: write a NOTES: line to start keeping notes)")
        messages.append({"role": "user", "content": current})
        return messages

    # -- one move ---------------------------------------------------------

    async def act(self, obs: Observation) -> Decision:
        messages = self.build_messages(obs)
        completion = await self.provider.complete(self.system_prompt, messages)
        parsed = parse_command(completion.text)
        if not parsed.ok and looks_like_prose(_raw_last_line(completion.text)):
            # The model forgot the "> COMMAND" line and its last line is prose. Ask once for
            # just the command rather than sending a sentence to the parser. Still counted as
            # a format (parse) failure; the extra call's tokens and cost are added in.
            repair_messages = messages + [
                {"role": "assistant", "content": completion.text or "(empty reply)"},
                {"role": "user", "content": REPAIR_PROMPT},
            ]
            try:
                fix = await self.provider.complete(self.system_prompt, repair_messages)
            except ProviderError as e:
                e.partial = completion  # the first call was paid for; let the runner bill it
                raise
            fixed = parse_command(fix.text)
            completion = _merge(completion, fix)
            if fixed.command and not looks_like_prose(fixed.command):
                parsed = ParsedCommand(fixed.command, "repair")
        command = parsed.command or ("return" if obs.input_type == "char" else "look")
        if self.notes_enabled:
            new_notes = extract_notes(completion.text)
            if new_notes:
                self.notes = new_notes
        self._turns.append(
            _Turn(self.render_observation(obs), compact_reply(completion.text, command, parsed))
        )
        if self.history:
            self._turns = self._turns[-self.history :]
        else:
            self._turns = []
        return Decision(command=command, reply=completion.text, completion=completion, parsed=parsed)

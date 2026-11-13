"""The agent: turns Grue observations into prompts and model replies into commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from brass_lantern.grue_client import Observation
from brass_lantern.providers import Completion, Provider, ProviderError

MAX_COMMAND_CHARS = 120
MAX_NOTES_CHARS = 1500
MAX_REASONING_CHARS = 300  # reasoning kept in the compact assistant history
# With caching on, the history window start moves in chunks of this many turns instead of one at a
# time, so the message prefix stays byte-identical (and cacheable) for HISTORY_STEP turns at a stretch.
HISTORY_STEP = 10

FORMATS = ("text", "tools")

_PROMPT_BODY = """\
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
"""

# The answer-format section is the only part that differs between --format text and --format tools.
_RESPONSE_FORMAT_TEXT = """
RESPONSE FORMAT
- You may write brief reasoning first (a few sentences at most).
- Your reply MUST end with a final line of the form:
> COMMAND
  for example:
> open mailbox
- Exactly one command after the ">". Nothing after that line.\
"""

_RESPONSE_FORMAT_TOOLS = """
RESPONSE FORMAT
- You may think briefly first (a few sentences at most).
- Then call the `act` tool with exactly one game command, for example command: "open mailbox".
- Always finish by calling the `act` tool. Do not write the command only as text, and never \
write the game's replies or the next turn yourself: the game answers after you act.\
"""

SYSTEM_PROMPT = _PROMPT_BODY.rstrip("\n") + "\n" + _RESPONSE_FORMAT_TEXT
SYSTEM_PROMPT_TOOLS = _PROMPT_BODY.rstrip("\n") + "\n" + _RESPONSE_FORMAT_TOOLS

NOTES_ADDENDUM = """

NOTES
- You can only see the most recent turns of the game, so keep a short notes scratchpad. \
To update it, include a line starting with NOTES: (it may continue over a few following lines). \
It REPLACES your previous notes, so rewrite everything worth keeping: map, items, puzzles, goals. \
Keep it under about 1500 characters. Your current notes are shown to you every turn.
- Put NOTES before the final "> COMMAND" line."""

NOTES_ADDENDUM_TOOLS = """

NOTES
- You can only see the most recent turns of the game, so keep a short notes scratchpad. \
To update it, pass the optional `notes` argument of the `act` tool. \
It REPLACES your previous notes, so rewrite everything worth keeping: map, items, puzzles, goals. \
Keep it under about 1500 characters. Your current notes are shown to you every turn."""

ACT_TOOL_DESCRIPTION = (
    "Send exactly one command to the game. Use a short parser command such as 'open mailbox', "
    "'take lamp' or 'north' (or a single key when the game is waiting for a keypress)."
)


def act_tool(notes: bool = False) -> dict[str, Any]:
    """The provider-neutral ``act`` tool: one required ``command``, plus an optional ``notes``."""
    props: dict[str, Any] = {"command": {"type": "string", "description": "The single game command to send."}}
    if notes:
        props["notes"] = {
            "type": "string",
            "description": "Optional. Your updated notes; they REPLACE the previous notes (under ~1500 characters).",
        }
    return {
        "name": "act",
        "description": ACT_TOOL_DESCRIPTION,
        "parameters": {"type": "object", "properties": props, "required": ["command"]},
    }

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
    # How the command was obtained. Text format: "prompt" (> CMD) | "tag" (COMMAND: CMD) | "last_line" |
    # "repair" | "empty". Tools format: "tool" (native act call) | "repair" (act call after a repair
    # request) | "marker" (no tool call, but an explicit > CMD / COMMAND: line in the text) | "last_line" |
    # "empty".
    method: str

    @property
    def ok(self) -> bool:
        """False when the reply did not give the command the requested way (a format failure)."""
        return self.method in ("prompt", "tag", "tool")


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
            if ">" in body:  # "> examine barrel? Let's try just this.> barrel": the last one is the move
                body = body.rsplit(">", 1)[1].strip() or body
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

TOOL_REPAIR_PROMPT = (
    "You did not call the act tool. Call the `act` tool now with the single game command you want to "
    "send, for example command: \"north\"."
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
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        tool_calls=((a.tool_calls or []) + (b.tool_calls or [])) or None,
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


def _act_call(completion: Completion) -> tuple[str, str | None] | None:
    """(cleaned command, notes) from the first usable ``act`` tool call of a reply, else None."""
    for call in completion.tool_calls or []:
        args = call.get("arguments")
        if call.get("name") != "act" or not isinstance(args, dict):
            continue
        raw = args.get("command")
        command = clean_command(raw) if isinstance(raw, str) else ""
        if not command:
            continue
        notes = args.get("notes")
        return command, ((notes.strip()[:MAX_NOTES_CHARS] or None) if isinstance(notes, str) else None)
    return None


class Agent:
    """Holds the rolling context for one episode and asks the model for each move.

    ``format`` is how the model answers: "text" (a final ``> COMMAND`` line) or "tools" (a native
    ``act`` tool call). ``cache`` makes the history window advance in steps of HISTORY_STEP turns so
    provider prompt caches can hit; with it off the window slides one turn at a time, as it always did.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        max_turns: int | None = None,
        history: int = 30,
        notes: bool = False,
        format: str = "text",
        cache: bool = True,
    ):
        if format not in FORMATS:
            raise ValueError(f"format must be one of {', '.join(FORMATS)}, not {format!r}")
        self.provider = provider
        self.max_turns = max_turns
        self.history = max(0, history)
        self.notes_enabled = notes
        self.notes = ""
        self.format = format
        self.step = HISTORY_STEP if cache else 0  # 0 = slide one turn at a time
        if format == "tools":
            self.system_prompt = SYSTEM_PROMPT_TOOLS + (NOTES_ADDENDUM_TOOLS if notes else "")
            self.tools: list[dict[str, Any]] | None = [act_tool(notes)]
        else:
            self.system_prompt = SYSTEM_PROMPT + (NOTES_ADDENDUM if notes else "")
            self.tools = None
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
        for t in self._turns:  # act() keeps _turns trimmed to the window
            messages.append({"role": "user", "content": t.user})
            messages.append({"role": "assistant", "content": t.assistant})
        current = self.render_observation(obs)
        if self.notes_enabled:
            hint = "pass notes to the act tool" if self.format == "tools" else "write a NOTES: line"
            current += "\n\n[Your notes]\n" + (self.notes or f"(empty: {hint} to start keeping notes)")
        messages.append({"role": "user", "content": current})
        return messages

    # -- one move ---------------------------------------------------------

    async def _ask(self, messages: list[dict[str, str]]) -> Completion:
        if self.tools:  # only pass `tools` when used, so plain two-argument providers keep working
            return await self.provider.complete(self.system_prompt, messages, tools=self.tools)
        return await self.provider.complete(self.system_prompt, messages)

    async def _act_text(self, messages: list[dict[str, str]]) -> tuple[Completion, ParsedCommand]:
        completion = await self._ask(messages)
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
                fix = await self._ask(repair_messages)
            except ProviderError as e:
                e.partial = completion  # the first call was paid for; let the runner bill it
                raise
            fixed = parse_command(fix.text)
            completion = _merge(completion, fix)
            if fixed.command and not looks_like_prose(fixed.command):
                parsed = ParsedCommand(fixed.command, "repair")
        return completion, parsed

    async def _act_tools(self, messages: list[dict[str, str]]) -> tuple[Completion, ParsedCommand, str | None]:
        """One move in tools format: the command comes from the model's ``act`` tool call.

        No tool call: if the text still carries an explicit command line, use it (method "marker")
        without spending a repair call; otherwise ask once for the tool call (method "repair").
        Both count as format failures. Last resort is the last line of the text, then the default.
        """
        completion = await self._ask(messages)
        call = _act_call(completion)
        if call:
            return completion, ParsedCommand(call[0], "tool"), call[1]
        text_parse = parse_command(completion.text)
        if text_parse.method in ("prompt", "tag") and text_parse.command:
            return completion, ParsedCommand(text_parse.command, "marker"), None
        repair_messages = messages + [
            {"role": "assistant", "content": completion.text or "(no tool call)"},
            {"role": "user", "content": TOOL_REPAIR_PROMPT},
        ]
        try:
            fix = await self._ask(repair_messages)
        except ProviderError as e:
            e.partial = completion  # the first call was paid for; let the runner bill it
            raise
        merged = _merge(completion, fix)
        call = _act_call(fix)
        if call:
            return merged, ParsedCommand(call[0], "repair"), call[1]
        # Still no tool call: salvage whatever the text says (the repair reply first), else play the default.
        texts = (fix.text, completion.text)
        for text in texts:
            p = parse_command(text)
            if p.method in ("prompt", "tag") and p.command:
                return merged, ParsedCommand(p.command, "marker"), None
        for text in texts:
            p = parse_command(text)
            if p.method == "last_line" and p.command and not looks_like_prose(_raw_last_line(text)):
                return merged, p, None
        return merged, ParsedCommand("", "empty"), None

    def _trim_history(self) -> None:
        if not self.history:
            self._turns = []
        elif self.step:
            # Stepped window: keep at least `history` turns and drop the oldest `step` in one go once
            # there are history + step. Between drops the start of the window does not move, so the
            # rendered prefix is byte-identical from one call to the next (what prompt caches need).
            while len(self._turns) >= self.history + self.step:
                del self._turns[: self.step]
        else:
            self._turns = self._turns[-self.history :]  # sliding window (--cache off)

    async def act(self, obs: Observation) -> Decision:
        messages = self.build_messages(obs)
        new_notes: str | None = None
        if self.format == "tools":
            completion, parsed, new_notes = await self._act_tools(messages)
        else:
            completion, parsed = await self._act_text(messages)
            if self.notes_enabled:
                new_notes = extract_notes(completion.text)
        command = parsed.command or ("return" if obs.input_type == "char" else "look")
        if self.notes_enabled and new_notes:
            self.notes = new_notes
        # History stays provider-neutral plain text in both formats: the user turn is the rendered
        # observation (no notes: those only ever appear in the current message, so a past message never
        # changes) and the assistant turn is the trimmed reasoning plus a "> command" line. Native
        # tool_use/tool_result structures are deliberately not replayed: that would tie the history to
        # one provider's wire format, and tool_choice forces the next call to be an act call anyway.
        self._turns.append(
            _Turn(self.render_observation(obs), compact_reply(completion.text, command, parsed))
        )
        self._trim_history()
        return Decision(command=command, reply=completion.text, completion=completion, parsed=parsed)

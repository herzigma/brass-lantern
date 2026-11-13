"""Stepped history window (--cache) and the tool-calling answer format (--format tools)."""

import asyncio

import pytest

from brass_lantern.agent import (
    HISTORY_STEP,
    NOTES_ADDENDUM,
    NOTES_ADDENDUM_TOOLS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TOOLS,
    TOOL_REPAIR_PROMPT,
    Agent,
    act_tool,
)
from brass_lantern.grue_client import Observation
from brass_lantern.providers.base import Completion, Provider, ProviderError


class Recorder(Provider):
    """Replies with `> cmdN` and records every call (system, messages, tools)."""

    name = "recorder"

    def __init__(self):
        super().__init__("recorder")
        self.calls: list[tuple[str, list[dict], list | None]] = []

    async def complete(self, system, messages, tools=None):
        self.calls.append((system, [dict(m) for m in messages], tools))
        return Completion(text=f"> cmd{len(self.calls)}", input_tokens=1, output_tokens=1)


class ToolProvider(Provider):
    """Replays Completions (or plain strings, which become text-only replies) and records calls."""

    name = "toolfake"

    def __init__(self, replies):
        super().__init__("toolfake")
        self.replies = list(replies)
        self.calls: list[tuple[str, list[dict], list | None]] = []

    async def complete(self, system, messages, tools=None):
        self.calls.append((system, [dict(m) for m in messages], tools))
        r = self.replies.pop(0)
        return Completion(text=r, input_tokens=10, output_tokens=2, cost_usd=0.01) if isinstance(r, str) else r


def call(command=None, name="act", **extra):
    args = dict(extra)
    if command is not None:
        args["command"] = command
    return {"name": name, "arguments": args}


def reply(text="", *calls, **kw):
    return Completion(text=text, input_tokens=10, output_tokens=2, cost_usd=0.01, tool_calls=list(calls) or None, **kw)


def obs(turn=0, text="You are in a room.", **kw):
    return Observation(turn=turn, text=text, input_type="line", **kw)


def act(agent, o):
    return asyncio.run(agent.act(o))


# ---------------------------------------------------------------------------
# Stepped history window
# ---------------------------------------------------------------------------


def play(agent, n):
    """Run n turns against a Recorder; returns the messages sent on each call."""
    for i in range(n):
        act(agent, obs(turn=i, text=f"text {i}"))
    return [m for _, m, _ in agent.provider.calls]


def n_history_turns(messages):
    return (len(messages) - 1) // 2


def test_step_constant_is_ten():
    assert HISTORY_STEP == 10


def test_stepped_window_keeps_at_least_history_and_drops_a_chunk_at_history_plus_step():
    history = 3
    calls = play(Agent(Recorder(), history=history, cache=True), 40)
    sizes = [n_history_turns(m) for m in calls]
    # grows until history + step turns are stored, then drops `step` at once and starts over from `history`
    assert sizes[: history + HISTORY_STEP] == list(range(history + HISTORY_STEP))
    assert sizes[history + HISTORY_STEP] == history
    assert all(s >= history for s in sizes[history:]) and max(sizes) == history + HISTORY_STEP - 1
    # the window start (first message) moves only at those drops, `step` turns each time
    starts = [int(m[0]["content"].split()[1]) if n_history_turns(m) else None for m in calls]
    moves = [(i, b - a) for i, (a, b) in enumerate(zip(starts, starts[1:]), 1) if a is not None and b != a]
    assert moves and all(delta == HISTORY_STEP for _, delta in moves)
    assert [i for i, _ in moves] == [history + HISTORY_STEP, history + 2 * HISTORY_STEP, history + 3 * HISTORY_STEP]


def test_stepped_window_prefix_is_byte_identical_between_drops():
    calls = play(Agent(Recorder(), history=5, cache=True), 60)
    drops = 0
    for prev, nxt in zip(calls, calls[1:]):
        if nxt[0] == prev[0]:  # window start did not move: everything sent last time is still the prefix
            assert nxt[: len(prev)] == prev  # including last call's final user message (no notes involved)
        else:
            drops += 1
    assert drops == 5  # 60 turns, history 5, step 10: a drop at turns 15, 25, 35, 45, 55


def test_prefix_stability_holds_with_notes_enabled_because_notes_stay_out_of_history():
    p = ToolProvider([])  # text replies with a changing NOTES: block
    p.replies = [f"NOTES: n{i}\n> cmd{i}" for i in range(30)]
    a = Agent(p, history=3, notes=True, cache=True)
    for i in range(25):
        act(a, obs(turn=i, text=f"text {i}"))
    seen: dict[int, str] = {}  # history-user position (by turn text) -> the string sent for it
    for _, msgs, _ in p.calls:
        for m in msgs[:-1]:
            if m["role"] == "user":
                assert "[Your notes]" not in m["content"]
                assert seen.setdefault(m["content"].split("\n")[0], m["content"]) == m["content"]  # never rewritten


def test_cache_off_keeps_the_exact_sliding_window():
    history = 3
    calls = play(Agent(Recorder(), history=history, cache=False), 20)
    assert [n_history_turns(m) for m in calls] == [min(i, history) for i in range(20)]
    for i, m in enumerate(calls):  # the window start advances every single turn once it is full
        if i >= history:
            assert m[0]["content"].startswith(f"text {i - history}")


def test_history_zero_sends_only_the_current_turn_in_both_modes():
    for cache in (True, False):
        calls = play(Agent(Recorder(), history=0, cache=cache), 15)
        assert all(len(m) == 1 for m in calls)


def test_cache_defaults_on_and_format_defaults_to_text():
    a = Agent(Recorder())
    assert a.step == HISTORY_STEP and a.format == "text" and a.tools is None
    assert Agent(Recorder(), cache=False).step == 0
    with pytest.raises(ValueError, match="format"):
        Agent(Recorder(), format="xml")


def test_text_mode_never_passes_tools():
    p = Recorder()
    act(Agent(p), obs())
    assert p.calls[0][2] is None


# ---------------------------------------------------------------------------
# Tools format
# ---------------------------------------------------------------------------


def test_tools_prompt_keeps_the_guidance_but_swaps_the_output_instructions():
    assert "> COMMAND" not in SYSTEM_PROMPT_TOOLS and "`act`" in SYSTEM_PROMPT_TOOLS
    assert "> COMMAND" in SYSTEM_PROMPT
    for shared in ("HOW IT WORKS", "STRATEGY", "examine", "Never use quit"):
        assert shared in SYSTEM_PROMPT and shared in SYSTEM_PROMPT_TOOLS
    assert SYSTEM_PROMPT_TOOLS.split("RESPONSE FORMAT")[0] == SYSTEM_PROMPT.split("RESPONSE FORMAT")[0]
    assert "NOTES:" not in NOTES_ADDENDUM_TOOLS and "`notes`" in NOTES_ADDENDUM_TOOLS
    p = ToolProvider([reply("", call("look"))])
    act(Agent(p, format="tools"), obs())
    assert p.calls[0][0] == SYSTEM_PROMPT_TOOLS
    p = ToolProvider([reply("", call("look"))])
    act(Agent(p, format="tools", notes=True), obs())
    assert p.calls[0][0] == SYSTEM_PROMPT_TOOLS + NOTES_ADDENDUM_TOOLS
    assert SYSTEM_PROMPT + NOTES_ADDENDUM != SYSTEM_PROMPT_TOOLS + NOTES_ADDENDUM_TOOLS


def test_act_tool_schema():
    plain = act_tool()
    assert plain["name"] == "act" and plain["description"]
    assert plain["parameters"] == {
        "type": "object",
        "properties": {"command": {"type": "string", "description": plain["parameters"]["properties"]["command"]["description"]}},
        "required": ["command"],
    }
    with_notes = act_tool(notes=True)
    assert set(with_notes["parameters"]["properties"]) == {"command", "notes"}
    assert with_notes["parameters"]["required"] == ["command"]  # notes stay optional
    p = ToolProvider([reply("", call("look"))])
    act(Agent(p, format="tools", notes=True), obs())
    assert p.calls[0][2] == [with_notes]


def test_tool_call_gives_the_command():
    p = ToolProvider([reply("The door is shut.", call("open door"))])
    d = act(Agent(p, format="tools"), obs())
    assert d.command == "open door" and d.parsed.method == "tool" and d.parsed.ok
    assert len(p.calls) == 1 and p.calls[0][2][0]["name"] == "act"
    assert d.reply == "The door is shut." and d.completion.tool_calls == [call("open door")]


def test_tool_command_is_cleaned_and_capped_like_text_commands():
    p = ToolProvider([
        reply("", call('"Take lamp then go north."')),
        reply("", call("x" * 500)),
        reply("", call("  open   the\tmailbox ")),
    ])
    a = Agent(p, format="tools")
    assert act(a, obs()).command == "Take lamp"  # same clean-up as text mode (case is kept)
    d = act(a, obs())
    assert len(d.command) == 120
    assert act(a, obs()).command == "open the mailbox"


def test_first_act_call_wins_and_other_tools_are_ignored():
    p = ToolProvider([reply("", call("ignored", name="think"), call("north"), call("south"))])
    d = act(Agent(p, format="tools"), obs())
    assert d.command == "north" and len(p.calls) == 1


def test_history_stays_plain_text_with_reasoning_and_a_command_line():
    long_reasoning = "I should try the door. " * 40
    p = ToolProvider([reply(long_reasoning, call("open door")), reply("", call("look")), reply("", call("north"))])
    a = Agent(p, format="tools")
    act(a, obs(turn=0))
    act(a, obs(turn=1))
    act(a, obs(turn=2))
    msgs = p.calls[2][1]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in msgs)  # no tool_use structures
    first = msgs[1]["content"]
    assert first.startswith("I should try the door.") and first.endswith("\n> open door") and len(first) < 340
    assert msgs[3]["content"] == "> look"  # no reasoning: just the command line


def test_no_tool_call_asks_for_one_repair_call_and_counts_a_format_failure():
    p = ToolProvider([reply("I wonder what to do next, so many options here to weigh up carefully."), reply("", call("north"))])
    d = act(Agent(p, format="tools"), obs())
    assert d.command == "north" and d.parsed.method == "repair" and not d.parsed.ok
    assert len(p.calls) == 2
    repair_msgs = p.calls[1][1]
    assert repair_msgs[-2]["role"] == "assistant" and repair_msgs[-2]["content"].startswith("I wonder")
    assert repair_msgs[-1] == {"role": "user", "content": TOOL_REPAIR_PROMPT}
    assert p.calls[1][2] is not None  # the repair call still offers the tool
    assert d.completion.input_tokens == 20 and d.completion.cost_usd == pytest.approx(0.02)  # both calls billed
    assert d.completion.tool_calls == [call("north")]


def test_repair_without_a_tool_call_falls_back_to_text_then_to_the_default():
    prose = "Turns are limited, so I'll make steady progress toward the chasm."
    p = ToolProvider([
        reply(prose), reply("> south"),  # repair gives a command line in text
        reply(prose), reply(""),  # nothing usable at all
        reply("go west"), reply(""),  # a short last line is still a plausible command
        reply(""), reply(""),
    ])
    a = Agent(p, format="tools")
    d = act(a, obs())
    assert d.command == "south" and d.parsed.method == "marker" and not d.parsed.ok
    d = act(a, obs())
    assert d.command == "look" and d.parsed.method == "empty" and not d.parsed.ok
    d = act(a, obs())
    assert d.command == "go west" and d.parsed.method == "last_line" and not d.parsed.ok
    d = act(a, Observation(turn=3, text="Press a key", input_type="char"))
    assert d.command == "return"
    assert len(p.calls) == 8


@pytest.mark.parametrize(
    "bad_call",
    [
        call(name="act"),  # act without a command
        call("   "),  # blank command
        {"name": "act", "arguments": {"_raw": "go north please"}},  # undecodable arguments
        call(7),  # wrong type
        call("north", name="look"),  # some other tool
    ],
)
def test_unusable_tool_calls_count_as_no_tool_call(bad_call):
    p = ToolProvider([reply("", bad_call), reply("", call("north"))])
    d = act(Agent(p, format="tools"), obs())
    assert d.command == "north" and d.parsed.method == "repair" and not d.parsed.ok and len(p.calls) == 2
    p = ToolProvider([reply("", bad_call), reply("", bad_call)])
    d = act(Agent(p, format="tools"), obs())
    assert d.command == "look" and d.parsed.method == "empty" and len(p.calls) == 2


def test_explicit_command_line_without_a_tool_call_is_used_without_a_repair_call():
    p = ToolProvider(["Let me think.\n> go north"])
    d = act(Agent(p, format="tools"), obs())
    assert d.command == "go north" and d.parsed.method == "marker" and not d.parsed.ok
    assert len(p.calls) == 1


def test_repair_failure_attaches_the_first_completion():
    class Flaky(ToolProvider):
        async def complete(self, system, messages, tools=None):
            self.calls.append((system, messages, tools))
            if len(self.calls) == 2:
                raise ProviderError("repair failed")
            return Completion(text="No command line here, just a long ramble about the situation.", input_tokens=7, cost_usd=0.5)

    with pytest.raises(ProviderError) as ei:
        act(Agent(Flaky([]), format="tools"), obs())
    assert ei.value.partial is not None and ei.value.partial.input_tokens == 7 and ei.value.partial.cost_usd == 0.5


def test_notes_come_from_the_tool_argument_and_text_notes_lines_are_ignored():
    p = ToolProvider([
        reply("NOTES: from the text\n", call("open mailbox")),  # ignored: no notes argument
        reply("", call("look", notes="key is in the mailbox")),
        reply("", call("north")),  # no notes argument: the old notes stay
        reply("", call("look", notes="   ")),  # blank notes do not wipe them
        reply("", call("look", notes="x" * 4000)),
        reply("", call("look")),
    ])
    a = Agent(p, format="tools", notes=True)
    act(a, obs(turn=0))
    assert a.notes == "" and "[Your notes]\n(empty: pass notes to the act tool" in p.calls[0][1][-1]["content"]
    act(a, obs(turn=1))
    assert a.notes == "key is in the mailbox"
    act(a, obs(turn=2))
    assert "[Your notes]\nkey is in the mailbox" in p.calls[2][1][-1]["content"]
    act(a, obs(turn=3))
    assert a.notes == "key is in the mailbox"
    act(a, obs(turn=4))
    assert len(a.notes) == 1500
    act(a, obs(turn=5))
    history_users = [m["content"] for m in p.calls[5][1][:-1] if m["role"] == "user"]
    assert history_users and all("[Your notes]" not in h for h in history_users)


def test_notes_argument_is_ignored_when_notes_are_disabled():
    p = ToolProvider([reply("", call("look", notes="secret"))])
    a = Agent(p, format="tools")
    act(a, obs())
    assert a.notes == ""


def test_tools_mode_uses_the_stepped_window_too():
    p = ToolProvider([reply("", call(f"cmd{i}")) for i in range(30)])
    a = Agent(p, history=3, format="tools", cache=True)
    for i in range(25):
        act(a, obs(turn=i, text=f"text {i}"))
    sizes = [n_history_turns(m) for _, m, _ in p.calls]
    assert sizes[:13] == list(range(13)) and sizes[13] == 3

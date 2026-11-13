import asyncio

from brass_lantern.agent import CHAR_WARNING, NOTES_ADDENDUM, SYSTEM_PROMPT, Agent
from brass_lantern.grue_client import Observation
from brass_lantern.providers.base import Completion, Provider


class FakeProvider(Provider):
    """Replies from a list and records what it was sent."""

    name = "fake"

    def __init__(self, replies):
        super().__init__("fake")
        self.replies = list(replies)
        self.calls: list[tuple[str, list[dict]]] = []

    async def complete(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        return Completion(text=self.replies.pop(0) if self.replies else "> look", input_tokens=1, output_tokens=1)


def obs(turn=0, text="You are in a room.", **kw):
    return Observation(turn=turn, text=text, input_type="line", **kw)


def act(agent, o):
    return asyncio.run(agent.act(o))


def test_system_prompt_mentions_format_and_basics():
    assert "> COMMAND" in SYSTEM_PROMPT
    assert "examine" in SYSTEM_PROMPT and "north" in SYSTEM_PROMPT
    p = FakeProvider(["> look"])
    act(Agent(p), obs())
    assert p.calls[0][0] == SYSTEM_PROMPT  # no notes text unless enabled
    p2 = FakeProvider(["> look"])
    act(Agent(p2, notes=True), obs())
    assert p2.calls[0][0] == SYSTEM_PROMPT + NOTES_ADDENDUM


def test_status_footer():
    a = Agent(FakeProvider([]), max_turns=150)
    text = a.render_observation(obs(turn=5, text="Hello.", score=10, max_score=350, location="Cellar"))
    assert text.startswith("Hello.")
    assert "[Turn 5 of 150 | Score 10 of 350 | Location: Cellar]" in text
    bare = a.render_observation(obs(turn=0, text=""))
    assert "(the game printed nothing)" in bare and "[Turn 0 of 150]" in bare


def test_char_mode_warning_and_notice():
    a = Agent(FakeProvider([]))
    o = Observation(turn=3, text="Press any key.", input_type="char", notice="Game expects a single keypress")
    text = a.render_observation(o)
    assert CHAR_WARNING in text and "SINGLE KEYPRESS" in text and "return" in text and "space" in text
    assert "Server notice: Game expects a single keypress" in text
    assert CHAR_WARNING not in a.render_observation(obs())


def test_history_trimming_keeps_last_n_turns_verbatim():
    p = FakeProvider([f"> cmd{i}" for i in range(10)])
    a = Agent(p, history=3)
    for i in range(6):
        act(a, obs(turn=i, text=f"text {i}"))
    msgs = p.calls[-1][1]
    # 3 prior turns (user+assistant each) + the current user message
    assert len(msgs) == 7
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3 + ["user"]
    assert msgs[0]["content"].startswith("text 2")  # turns 0 and 1 are gone
    assert msgs[1]["content"] == "> cmd2"
    assert msgs[-1]["content"].startswith("text 5")


def test_history_zero_sends_only_current_turn():
    p = FakeProvider(["> a", "> b"])
    a = Agent(p, history=0)
    act(a, obs(turn=0, text="one"))
    act(a, obs(turn=1, text="two"))
    assert len(p.calls[1][1]) == 1


def test_assistant_history_is_compact():
    long_reply = "Reasoning. " + "word " * 300 + "\nNOTES: hidden\n> open door"
    p = FakeProvider([long_reply, "> look"])
    a = Agent(p)
    d = act(a, obs(turn=0))
    assert d.reply == long_reply  # full reply is preserved on the Decision
    act(a, obs(turn=1))
    assistant = p.calls[1][1][1]["content"]
    assert assistant.endswith("> open door") and len(assistant) < 340 and "hidden" not in assistant


def test_notes_roundtrip_replace_and_cap():
    p = FakeProvider(["NOTES: key is in the mailbox\n> open mailbox", "> look", "NOTES: " + "x" * 4000 + "\n> look"])
    a = Agent(p, notes=True)
    act(a, obs(turn=0))
    assert "[Your notes]\n(empty" in p.calls[0][1][-1]["content"]
    act(a, obs(turn=1))
    assert "[Your notes]\nkey is in the mailbox" in p.calls[1][1][-1]["content"]
    act(a, obs(turn=2))
    assert len(a.notes) == 1500
    # notes are not baked into the verbatim history
    p.replies.append("> look")
    act(a, obs(turn=3))
    history_users = [m["content"] for m in p.calls[3][1][:-1] if m["role"] == "user"]
    assert all("[Your notes]" not in h for h in history_users)


def test_notes_ignored_when_disabled():
    p = FakeProvider(["NOTES: stuff\n> look", "> look"])
    a = Agent(p)
    act(a, obs(turn=0))
    act(a, obs(turn=1))
    assert a.notes == "" and "[Your notes]" not in p.calls[1][1][-1]["content"]


def test_decision_command_and_fallbacks():
    p = FakeProvider([
        "I will go.\n> go north",
        "just text",
        "Turns are limited, so I'll make steady progress toward the chasm.", "> south",
        "", "",
        "", "",
        "> take lamp then north",
    ])
    a = Agent(p)
    d = act(a, obs())
    assert d.command == "go north" and d.parsed.ok
    d = act(a, obs())
    assert d.command == "just text" and not d.parsed.ok and d.parsed.method == "last_line"
    d = act(a, obs())  # prose without a marker -> one format-repair call
    assert d.command == "south" and d.parsed.method == "repair" and not d.parsed.ok
    assert "[format repair]" in d.reply
    d = act(a, obs())  # empty reply (and empty repair) -> harmless "look"
    assert d.command == "look" and d.parsed.method == "empty"
    d = act(a, Observation(turn=1, text="Press a key", input_type="char"))  # empty in char mode -> return
    assert d.command == "return"
    d = act(a, obs())
    assert d.command == "take lamp"

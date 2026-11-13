import pytest

from brass_lantern.agent import clean_command, compact_reply, extract_notes, parse_command


@pytest.mark.parametrize(
    "reply, command, method",
    [
        ("> open mailbox", "open mailbox", "prompt"),
        ("I should look inside.\n> open mailbox", "open mailbox", "prompt"),
        ("Thinking...\n\n> take lamp\n", "take lamp", "prompt"),
        (">open mailbox", "open mailbox", "prompt"),
        ("**> open mailbox**", "open mailbox", "prompt"),
        ("`> open mailbox`", "open mailbox", "prompt"),
        ("```\n> open mailbox\n```", "open mailbox", "prompt"),
        ("> go north.", "go north", "prompt"),
        ('> "open mailbox"', "open mailbox", "prompt"),
        ("> 'open mailbox'", "open mailbox", "prompt"),
        ("> “open mailbox”", "open mailbox", "prompt"),
        ("> open   the\tmailbox  ", "open the mailbox", "prompt"),
        ("> > open mailbox", "open mailbox", "prompt"),
        # last > line wins, including over blockquoted reasoning
        ("> I wonder what is here\nmaybe the door\n> open door", "open door", "prompt"),
        ("> look\n> north", "north", "prompt"),
        # echoed template is ignored
        ("> COMMAND\n> look", "look", "prompt"),
        # prompt beats tag
        ("COMMAND: look\n> north", "north", "prompt"),
        # tag forms
        ("COMMAND: open door", "open door", "tag"),
        ("Command: open door", "open door", "tag"),
        ("**Command:** open door", "open door", "tag"),
        ("Some thoughts\ncommand: examine desk\nthanks", "examine desk", "tag"),
        ("ACTION: take key", "take key", "tag"),
        ("COMMAND: > look", "look", "tag"),
        # fallback to last non-empty line
        ("open mailbox", "open mailbox", "last_line"),
        ("I will try this.\n\nexamine desk\n\n", "examine desk", "last_line"),
        ("**open mailbox**", "open mailbox", "last_line"),
        # multiple commands: keep the first
        ("> take lamp then go north", "take lamp", "prompt"),
        ("> take lamp, then go north", "take lamp", "prompt"),
        ("> take lamp, go north", "take lamp", "prompt"),
        ("> take lamp; go north", "take lamp", "prompt"),
        ("> take lamp and then go north", "take lamp", "prompt"),
        ("> take lamp THEN north", "take lamp", "prompt"),
        # but "and" alone and commas inside quotes are kept
        ("> take lamp and sword", "take lamp and sword", "prompt"),
        ('> say "hello, world"', 'say "hello, world"', "prompt"),
        ("> put coin in slot, then pull lever", "put coin in slot", "prompt"),
        # words that merely contain "then"
        ("> examine authentic mat", "examine authentic mat", "prompt"),
        # <think> blocks are ignored
        ("<think>\n> wrong\n</think>\n> right", "right", "prompt"),
        ("<think>long ponder</think>go east", "go east", "last_line"),
        # char-mode keys
        ("> \\space", "\\space", "prompt"),
        ("> return", "return", "prompt"),
        ("> .", ".", "prompt"),
    ],
)
def test_parse_command(reply, command, method):
    parsed = parse_command(reply)
    assert (parsed.command, parsed.method) == (command, method)
    assert parsed.ok == (method in ("prompt", "tag"))


def test_parse_empty():
    for text in ("", "   \n\n", None, "<think>only thoughts</think>"):
        p = parse_command(text)
        assert p.command == "" and p.method == "empty" and not p.ok


def test_truncates_to_120_chars():
    p = parse_command("> " + "x" * 300)
    assert len(p.command) == 120


def test_clean_command_trailing_punctuation():
    assert clean_command("open door!") == "open door"
    assert clean_command("...") == "..."


def test_extract_notes_basic_and_last_wins():
    assert extract_notes("NOTES: key under mat\n> look") == "key under mat"
    assert extract_notes("NOTES: old\nthinking\nNOTES: new\n> look") == "new"
    assert extract_notes("no notes here\n> look") is None
    assert extract_notes("NOTES:\n> look") is None


def test_extract_notes_multiline_and_markdown():
    text = "**NOTES:** map\n- north: cave\n- east: lake\n\nreasoning\n> look"
    assert extract_notes(text) == "map\n- north: cave\n- east: lake"


def test_extract_notes_capped():
    assert len(extract_notes("NOTES: " + "a" * 5000 + "\n> look")) == 1500


def test_compact_reply_trims_and_drops_notes():
    reply = "First thought. " + "blah " * 200 + "\nNOTES: secret map\n> open door"
    out = compact_reply(reply, "open door", parse_command(reply))
    reasoning, last = out.split("\n")
    assert last == "> open door"
    assert len(reasoning) <= 303 and reasoning.endswith("...")
    assert "secret map" not in out


def test_compact_reply_last_line_fallback_not_duplicated():
    reply = "Let me think.\nopen door"
    p = parse_command(reply)
    assert compact_reply(reply, p.command, p) == "Let me think.\n> open door"


def test_compact_reply_command_only():
    assert compact_reply("> look", "look", parse_command("> look")) == "> look"


def test_hallucinated_continuation_is_ignored():
    reply = (
        "I'll open it.\n> open tin\nYou open the tin.\n\n"
        "[Turn 21 of 150 | Score 2 of 10 | Location: Kitchen]\n\n> take everything"
    )
    p = parse_command(reply)
    assert p.command == "open tin" and p.method == "prompt"


def test_reply_starting_with_echoed_footer_has_no_command():
    p = parse_command("[Turn 3 of 50 | Score 0]\n> north")
    assert p.method == "empty"


def test_second_prompt_marker_on_one_line_wins():
    p = parse_command("> examine barrel again? The parser might not understand that.> barrel")
    assert p.command == "barrel" and p.method == "prompt"

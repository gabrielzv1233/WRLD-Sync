import pytest

from gap_hints import parse_lyrics_gap_hints


def test_plain_gap_is_removed_from_alignment_text():
    parsed = parse_lyrics_gap_hints("Line one\n[...]\nLine two")

    assert parsed.text == "Line one\nLine two"
    assert parsed.lines == ("Line one", "Line two")
    assert len(parsed.gaps) == 1
    assert parsed.gaps[0].position == 1
    assert parsed.gaps[0].strength == 1
    assert parsed.gaps[0].interlude == "auto"


def test_interlude_overrides():
    forced = parse_lyrics_gap_hints("A\n+[...]\nB").gaps[0]
    forbidden = parse_lyrics_gap_hints("A\n-[...]\nB").gaps[0]

    assert forced.interlude == "force"
    assert forbidden.interlude == "forbid"


def test_numeric_and_repeated_strengths_accumulate():
    numeric = parse_lyrics_gap_hints("A\n2+[...]\nB").gaps[0]
    repeated = parse_lyrics_gap_hints("A\n+[...]+[...]\nB").gaps[0]
    separate = parse_lyrics_gap_hints("A\n+[...]\n+[...]\nB").gaps[0]

    assert numeric.strength == 2
    assert repeated.strength == 2
    assert separate.strength == 2
    assert numeric.interlude == repeated.interlude == separate.interlude == "force"


def test_auto_markers_do_not_override_explicit_policy():
    parsed = parse_lyrics_gap_hints("A\n[...]\n2-[...]\nB")

    assert parsed.gaps[0].strength == 3
    assert parsed.gaps[0].interlude == "forbid"


def test_marker_like_text_inside_a_lyric_is_not_control_syntax():
    parsed = parse_lyrics_gap_hints("I waited [...] forever\nNext line")

    assert parsed.text == "I waited [...] forever\nNext line"
    assert parsed.gaps == ()


def test_leading_and_trailing_gaps_are_supported():
    parsed = parse_lyrics_gap_hints("+[...]\nFirst\nLast\n-[...]")

    assert [(gap.position, gap.interlude) for gap in parsed.gaps] == [
        (0, "force"),
        (2, "forbid"),
    ]


def test_conflicting_explicit_controls_are_rejected():
    with pytest.raises(ValueError, match="Conflicting gap hint controls"):
        parse_lyrics_gap_hints("A\n+[...]-[...]\nB")

    with pytest.raises(ValueError, match="Conflicting gap hint controls"):
        parse_lyrics_gap_hints("A\n+[...]\n-[...]\nB")

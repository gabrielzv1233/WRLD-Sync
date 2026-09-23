import pytest

from gap_hints import build_alignment_chunks, parse_lyrics_gap_hints


def test_plain_gap_is_removed_from_alignment_text():
    parsed = parse_lyrics_gap_hints("Line one\n[...]\nLine two")

    assert parsed.text == "Line one\nLine two"
    assert parsed.lines == ("Line one", "Line two")
    assert len(parsed.gaps) == 1
    assert parsed.gaps[0].position == 1
    assert parsed.gaps[0].seconds == 0.25
    assert parsed.gaps[0].interlude == "auto"


def test_interlude_overrides_keep_default_quarter_second():
    forced = parse_lyrics_gap_hints("A\n+[...]\nB").gaps[0]
    forbidden = parse_lyrics_gap_hints("A\n-[...]\nB").gaps[0]

    assert forced.seconds == 0.25
    assert forbidden.seconds == 0.25
    assert forced.interlude == "force"
    assert forbidden.interlude == "forbid"


def test_integer_and_decimal_seconds_are_literal():
    integer = parse_lyrics_gap_hints("A\n2+[...]\nB").gaps[0]
    decimal = parse_lyrics_gap_hints("A\n11.5[...]\nB").gaps[0]
    leading_decimal = parse_lyrics_gap_hints("A\n.25-[...]\nB").gaps[0]
    zero_prefixed = parse_lyrics_gap_hints("A\n0.25[...]\nB").gaps[0]

    assert integer.seconds == 2.0
    assert integer.interlude == "force"
    assert decimal.seconds == 11.5
    assert leading_decimal.seconds == 0.25
    assert leading_decimal.interlude == "forbid"
    assert zero_prefixed.seconds == 0.25


def test_zero_seconds_is_a_pure_alignment_boundary():
    parsed = parse_lyrics_gap_hints("A\n0[...]\nB")

    assert parsed.gaps[0].seconds == 0.0
    assert parsed.gaps[0].interlude == "auto"


def test_consecutive_markers_keep_larger_minimum_and_explicit_policy():
    parsed = parse_lyrics_gap_hints("A\n.5[...]\n2-[...]\nB")

    assert parsed.gaps[0].seconds == 2.0
    assert parsed.gaps[0].interlude == "forbid"


def test_auto_marker_does_not_override_explicit_policy():
    parsed = parse_lyrics_gap_hints("A\n2-[...]\n5[...]\nB")

    assert parsed.gaps[0].seconds == 5.0
    assert parsed.gaps[0].interlude == "forbid"


def test_marker_like_text_inside_a_lyric_is_not_control_syntax():
    parsed = parse_lyrics_gap_hints("I waited [...] forever\nNext line")

    assert parsed.text == "I waited [...] forever\nNext line"
    assert parsed.gaps == ()


def test_old_repeated_token_syntax_is_rejected_instead_of_becoming_lyrics():
    with pytest.raises(ValueError, match="Invalid gap hint syntax"):
        parse_lyrics_gap_hints("A\n+[...]+[...]\nB")


def test_leading_and_trailing_gaps_are_supported():
    parsed = parse_lyrics_gap_hints("+[...]\nFirst\nLast\n11.5-[...]")

    assert [(gap.position, gap.seconds, gap.interlude) for gap in parsed.gaps] == [
        (0, 0.25, "force"),
        (2, 11.5, "forbid"),
    ]


def test_conflicting_explicit_controls_are_rejected():
    with pytest.raises(ValueError, match="Conflicting gap hint controls"):
        parse_lyrics_gap_hints("A\n1+[...]\n2-[...]\nB")


def test_alignment_chunks_split_at_gap_boundaries():
    parsed = parse_lyrics_gap_hints(
        "A one\nA two\n[...]\nB one\n11.5+[...]\nC one\nC two"
    )
    chunks = build_alignment_chunks(parsed)

    assert [chunk.lines for chunk in chunks] == [
        ("A one", "A two"),
        ("B one",),
        ("C one", "C two"),
    ]
    assert chunks[0].before_gap is None
    assert chunks[1].before_gap.position == 2
    assert chunks[1].before_gap.seconds == 0.25
    assert chunks[2].before_gap.position == 3
    assert chunks[2].before_gap.seconds == 11.5
    assert chunks[2].before_gap.interlude == "force"


def test_leading_gap_attaches_to_first_chunk_and_trailing_gap_does_not_make_empty_chunk():
    parsed = parse_lyrics_gap_hints("3+[...]\nFirst\nSecond\n4-[...]")
    chunks = build_alignment_chunks(parsed)

    assert len(chunks) == 1
    assert chunks[0].lines == ("First", "Second")
    assert chunks[0].before_gap is not None
    assert chunks[0].before_gap.seconds == 3.0
    assert chunks[0].before_gap.interlude == "force"
    assert parsed.gaps[-1].position == 2
    assert parsed.gaps[-1].seconds == 4.0
    assert parsed.gaps[-1].interlude == "forbid"


def test_anchor_search_uses_first_matching_phrase_after_barrier():
    from gap_hints import find_alignment_anchor

    words = [
        {"word": "this", "start": 1.0, "end": 1.2},
        {"word": "is", "start": 1.3, "end": 1.4},
        {"word": "next", "start": 1.5, "end": 1.7},
        {"word": "noise", "start": 5.0, "end": 5.2},
        {"word": "this", "start": 8.0, "end": 8.2},
        {"word": "is", "start": 8.3, "end": 8.4},
        {"word": "next", "start": 8.5, "end": 8.7},
    ]

    assert find_alignment_anchor("this is next", words, start_at=0.0) == 1.0
    assert find_alignment_anchor("this is next", words, start_at=4.0) == 8.0


def test_anchor_search_does_not_anchor_on_unrelated_prefix_words():
    from gap_hints import find_alignment_anchor

    words = [
        {"word": "blah", "start": 0.0, "end": 0.2},
        {"word": "blah", "start": 0.3, "end": 0.5},
        {"word": "this", "start": 2.0, "end": 2.2},
        {"word": "is", "start": 2.3, "end": 2.4},
        {"word": "the", "start": 2.5, "end": 2.6},
        {"word": "next", "start": 2.7, "end": 2.9},
        {"word": "lyric", "start": 3.0, "end": 3.2},
    ]

    assert find_alignment_anchor("this is the next lyric", words) == 2.0


def test_anchor_search_can_tolerate_small_first_word_asr_error():
    from gap_hints import find_alignment_anchor

    words = [
        {"word": "gunna", "start": 12.0, "end": 12.2},
        {"word": "make", "start": 12.3, "end": 12.5},
        {"word": "it", "start": 12.6, "end": 12.7},
        {"word": "back", "start": 12.8, "end": 13.0},
    ]

    assert find_alignment_anchor("gonna make it back", words, start_at=10.0) == 12.0

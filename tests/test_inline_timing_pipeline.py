import pytest

from app import (
    _assert_timing_hints_consumed,
    _merge_virtual_timing_rows,
    _virtualize_inline_timing,
)
from gap_hints import parse_lyrics_gap_hints


def timed_row(line, start, end, words):
    return {
        "line": line, "start": start, "end": end,
        "words": [
            {"word": text, "start": word_start, "end": word_end}
            for text, word_start, word_end in words
        ],
    }


def test_hennessy_forward_inline_hint_is_removed_but_preserves_one_original_row():
    parsed = parse_lyrics_gap_hints(
        "First lyric\nSecond lyric\n"
        "Ah, 2-[...]pour up, pour up, I take Hennessey to pour up +{(Woah-oh)}"
    )
    virtual, indices, backward_at = _virtualize_inline_timing(parsed)

    assert indices == [0, 1, 2, 2]
    assert virtual.lines[2] == "Ah,"
    assert virtual.lines[3] == "pour up, pour up, I take Hennessey to pour up (Woah-oh)"
    assert [(gap.position, gap.seconds, gap.interlude) for gap in virtual.gaps] == [
        (3, 2.0, "forbid")
    ]
    assert backward_at == {}
    assert "2-[...]" not in virtual.text

    rows = [
        timed_row("First lyric", 1.0, 2.0, [("First", 1.0, 1.3), ("lyric", 1.4, 2.0)]),
        timed_row("Second lyric", 11.0, 12.0, [("Second", 11.0, 11.3), ("lyric", 11.4, 12.0)]),
        timed_row("Ah,", 15.130, 15.359, [("Ah,", 15.130, 15.359)]),
        timed_row(virtual.lines[3], 17.680, 23.759, [
            ("pour", 17.680, 18.0), ("up,", 18.1, 18.3),
            ("pour", 18.4, 18.6), ("up,", 18.7, 18.9),
            ("I", 19.0, 19.1), ("take", 19.2, 19.3),
            ("Hennessey", 19.4, 19.8), ("to", 19.9, 20.0),
            ("pour", 20.1, 20.5), ("up", 20.6, 20.8),
            ("(Woah-oh)", 23.7, 23.759),
        ]),
    ]
    result = _merge_virtual_timing_rows(rows, parsed, indices, backward_at)

    assert len(result) == 3
    assert result[2]["line"] == parsed.lines[2]
    assert result[2]["start"] == 15.130
    assert result[2]["end"] == 23.759
    assert [w["word"] for w in result[2]["words"]][0:3] == ["Ah,", "pour", "up,"]
    assert all("2-[...]" not in w["word"] for row in result for w in row["words"])
    _assert_timing_hints_consumed(result, parsed)


def test_backward_hint_only_adjusts_following_segment_and_later_lines():
    parsed = parse_lyrics_gap_hints("Ah, -2-[...]pour up\nNext line")
    virtual, indices, backward_at = _virtualize_inline_timing(parsed)
    assert virtual.lines == ("Ah,", "pour up", "Next line")
    assert indices == [0, 0, 1]
    assert backward_at == {1: -2.0}

    rows = [
        timed_row("Ah,", 3.5, 3.7, [("Ah,", 3.5, 3.7)]),
        timed_row("pour up", 6.0, 7.0, [("pour", 6.0, 6.4), ("up", 6.5, 7.0)]),
        timed_row("Next line", 7.5, 8.0, [("Next", 7.5, 7.7), ("line", 7.8, 8.0)]),
    ]
    result = _merge_virtual_timing_rows(rows, parsed, indices, backward_at)
    assert [(w["word"], w["start"], w["end"]) for w in result[0]["words"]] == [
        ("Ah,", 3.5, 3.7), ("pour", 4.0, 4.4), ("up", 4.5, 5.0)
    ]
    assert result[1]["start"] == 5.5
    assert result[1]["end"] == 6.0
    _assert_timing_hints_consumed(result, parsed)


def test_sync_output_guard_rejects_marker_as_timed_word():
    parsed = parse_lyrics_gap_hints("Ah, 2-[...]pour up")
    bad = [timed_row("Ah, 2-[...]pour up", 15.13, 17.329, [
        ("Ah,", 15.13, 15.359),
        ("2-[...]", 15.371, 15.419),
        ("pour", 15.68, 17.0),
        ("up", 17.1, 17.329),
    ])]
    with pytest.raises(ValueError, match="timing marker reached the alignment output"):
        _assert_timing_hints_consumed(bad, parsed)


def test_negative_correction_cannot_move_a_word_before_zero():
    parsed = parse_lyrics_gap_hints("Ah, -2-[...]pour")
    virtual, indices, backward_at = _virtualize_inline_timing(parsed)
    rows = [
        timed_row("Ah,", 0.1, 0.2, [("Ah,", 0.1, 0.2)]),
        timed_row("pour", 1.0, 1.3, [("pour", 1.0, 1.3)]),
    ]
    with pytest.raises(ValueError, match="before 00:00"):
        _merge_virtual_timing_rows(rows, parsed, indices, backward_at)

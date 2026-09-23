from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal
import re

GapInterludeMode = Literal["auto", "force", "forbid"]

_MARKER_TOKEN_RE = re.compile(
    r"(?P<seconds>(?:\d+(?:\.\d+)?|\.\d+))?"
    r"(?P<mode>[+-]?)"
    r"(?P<marker>\[\.\.\.\])"
)
_CONTROLISH_RE = re.compile(r"[0-9.+\-\s\[\]]+")


@dataclass(frozen=True, slots=True)
class GapHint:
    """A manual timing barrier between lyric lines.

    position is the number of lyric lines before this gap. A value of 0 means
    the gap is before the first lyric; len(lyrics) means it is after the last.
    seconds is a literal minimum forward distance, not the final gap duration.
    """

    position: int
    seconds: float = 0.25
    interlude: GapInterludeMode = "auto"
    source: str = "[...]"


@dataclass(frozen=True, slots=True)
class ParsedLyrics:
    text: str
    lines: tuple[str, ...]
    gaps: tuple[GapHint, ...]


@dataclass(frozen=True, slots=True)
class AlignmentChunk:
    """A contiguous lyric section that can be forced-aligned independently."""

    start_line: int
    lines: tuple[str, ...]
    before_gap: GapHint | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def build_alignment_chunks(parsed: ParsedLyrics) -> tuple[AlignmentChunk, ...]:
    """Split parsed lyrics at gap hints while preserving each boundary policy.

    A leading marker becomes before_gap on the first chunk. A trailing marker
    is intentionally not turned into an empty chunk because there are no lyrics
    after it to align; it is still retained in ParsedLyrics.gaps for output
    policy and metadata.
    """
    if not parsed.lines:
        return ()

    gaps = {gap.position: gap for gap in parsed.gaps}
    internal_boundaries = sorted(
        position for position in gaps
        if 0 < position < len(parsed.lines)
    )

    chunks: list[AlignmentChunk] = []
    start = 0
    before_gap = gaps.get(0)
    for boundary in internal_boundaries:
        chunks.append(AlignmentChunk(
            start_line=start,
            lines=parsed.lines[start:boundary],
            before_gap=before_gap,
        ))
        start = boundary
        before_gap = gaps[boundary]

    chunks.append(AlignmentChunk(
        start_line=start,
        lines=parsed.lines[start:],
        before_gap=before_gap,
    ))
    return tuple(chunk for chunk in chunks if chunk.lines)


def _parse_marker_line(line: str) -> tuple[float, GapInterludeMode] | None:
    """Parse one manual gap-control line."""
    stripped = line.strip()
    if not stripped:
        return None

    match = _MARKER_TOKEN_RE.fullmatch(stripped)
    if not match:
        # A line that consists only of marker-ish punctuation/numbers is almost
        # certainly a malformed control. Fail visibly instead of sending it to
        # the aligner as fake lyric text.
        if "[...]" in stripped and _CONTROLISH_RE.fullmatch(stripped):
            raise ValueError(
                f"Invalid gap hint syntax: {stripped}. "
                "Use forms such as [...], 1.5[...], 1.5+[...], or 1.5-[...]."
            )
        return None

    seconds = float(match.group("seconds") or "0.25")
    marker_mode = match.group("mode")
    interlude: GapInterludeMode = (
        "force" if marker_mode == "+"
        else "forbid" if marker_mode == "-"
        else "auto"
    )
    return seconds, interlude


def _merge_gap(existing: GapHint | None, *, position: int, seconds: float,
               interlude: GapInterludeMode, source: str) -> GapHint:
    if existing is None:
        return GapHint(
            position=position,
            seconds=seconds,
            interlude=interlude,
            source=source,
        )

    explicit = {mode for mode in (existing.interlude, interlude) if mode != "auto"}
    if "force" in explicit and "forbid" in explicit:
        raise ValueError(f"Conflicting gap hint controls at lyric position {position}.")
    merged_mode: GapInterludeMode = next(iter(explicit), "auto")

    # Consecutive controls refer to the same boundary. The larger minimum is
    # the stricter constraint, so durations are not added together.
    return GapHint(
        position=position,
        seconds=max(existing.seconds, seconds),
        interlude=merged_mode,
        source=f"{existing.source}\n{source}",
    )


def parse_lyrics_gap_hints(lyrics: str) -> ParsedLyrics:
    """Strip manual gap controls from lyrics and return their timing metadata.

    Syntax:
      [...]       0.25s minimum gap, automatic interlude decision
      -[...]      0.25s minimum gap, never emit an interlude
      +[...]      0.25s minimum gap, force an interlude
      2[...]      2s minimum gap
      11.5+[...]  11.5s minimum gap, force an interlude
      .5-[...]    0.5s minimum gap, never emit an interlude

    Consecutive marker lines at the same position keep the larger minimum.
    Plain lyric lines containing "[...]" inline are left untouched.
    """
    lyric_lines: list[str] = []
    gaps_by_position: dict[int, GapHint] = {}

    for raw_line in str(lyrics or "").splitlines():
        parsed = _parse_marker_line(raw_line)
        if parsed is not None:
            seconds, interlude = parsed
            position = len(lyric_lines)
            gaps_by_position[position] = _merge_gap(
                gaps_by_position.get(position),
                position=position,
                seconds=seconds,
                interlude=interlude,
                source=raw_line.strip(),
            )
            continue

        line = raw_line.strip()
        if line:
            lyric_lines.append(line)

    return ParsedLyrics(
        text="\n".join(lyric_lines),
        lines=tuple(lyric_lines),
        gaps=tuple(gaps_by_position[position] for position in sorted(gaps_by_position)),
    )


def _normalize_anchor_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(value or "").lower())


def find_alignment_anchor(
    target_text: str,
    transcript_words: list[dict],
    *,
    start_at: float = 0.0,
    max_target_words: int = 10,
) -> float | None:
    """Return the earliest likely timestamp for target_text after start_at.

    The first target word must itself be a close match. That prevents a fuzzy
    window containing the right phrase later from anchoring on unrelated words
    that happen to come before it.
    """
    target = [
        _normalize_anchor_word(token)
        for token in re.findall(r"\S+", str(target_text or ""))
    ]
    target = [token for token in target if token][:max(1, int(max_target_words))]
    if not target:
        return None

    candidates: list[tuple[str, float, float]] = []
    for raw in transcript_words:
        token = _normalize_anchor_word(raw.get("word") or raw.get("text") or "")
        if not token:
            continue
        try:
            start = float(raw.get("start", 0.0) or 0.0)
            end = float(raw.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        if end >= start_at:
            candidates.append((token, start, end))

    if not candidates:
        return None

    if len(target) == 1:
        for token, start, _ in candidates:
            if start >= start_at and token == target[0]:
                return start
        return None

    threshold = 0.86 if len(target) == 2 else 0.78 if len(target) == 3 else 0.70
    min_window = max(1, len(target) - 2)
    max_window = len(target) + 2

    for index, (first_token, start, _) in enumerate(candidates):
        if start < start_at:
            continue
        first_score = SequenceMatcher(
            None, target[0], first_token, autojunk=False
        ).ratio()
        if first_score < 0.78:
            continue

        best_score = 0.0
        for size in range(min_window, max_window + 1):
            window = [item[0] for item in candidates[index:index + size]]
            if len(window) < min_window:
                continue
            score = SequenceMatcher(None, target, window, autojunk=False).ratio()
            best_score = max(best_score, score)
        if best_score >= threshold:
            return start

    return None

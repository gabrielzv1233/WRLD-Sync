from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

GapInterludeMode = Literal["auto", "force", "forbid"]

_MARKER_TOKEN_RE = re.compile(r"(?P<count>\d+)?(?P<mode>[+-]?)(?P<marker>\[\.\.\.\])")


@dataclass(frozen=True, slots=True)
class GapHint:
    """A manual timing barrier between lyric lines.

    position is the number of lyric lines before this gap. A value of 0 means
    the gap is before the first lyric; len(lyrics) means it is after the last.
    """

    position: int
    strength: int = 1
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


def _parse_marker_line(line: str) -> tuple[int, GapInterludeMode] | None:
    """Parse a line made entirely of one or more gap-control tokens."""
    stripped = line.strip()
    if not stripped:
        return None

    matches = list(_MARKER_TOKEN_RE.finditer(stripped))
    if not matches:
        return None

    cursor = 0
    total_strength = 0
    explicit_modes: set[GapInterludeMode] = set()
    for match in matches:
        if stripped[cursor:match.start()].strip():
            return None
        cursor = match.end()

        count = int(match.group("count") or "1")
        if count < 1:
            raise ValueError("Gap hint strength must be at least 1.")
        total_strength += count

        marker_mode = match.group("mode")
        if marker_mode == "+":
            explicit_modes.add("force")
        elif marker_mode == "-":
            explicit_modes.add("forbid")

    if stripped[cursor:].strip():
        return None
    if "force" in explicit_modes and "forbid" in explicit_modes:
        raise ValueError(f"Conflicting gap hint controls on one line: {stripped}")

    interlude: GapInterludeMode = next(iter(explicit_modes), "auto")
    return total_strength, interlude


def _merge_gap(existing: GapHint | None, *, position: int, strength: int,
               interlude: GapInterludeMode, source: str) -> GapHint:
    if existing is None:
        return GapHint(position=position, strength=strength, interlude=interlude, source=source)

    explicit = {mode for mode in (existing.interlude, interlude) if mode != "auto"}
    if "force" in explicit and "forbid" in explicit:
        raise ValueError(f"Conflicting gap hint controls at lyric position {position}.")
    merged_mode: GapInterludeMode = next(iter(explicit), "auto")
    return GapHint(
        position=position,
        strength=existing.strength + strength,
        interlude=merged_mode,
        source=f"{existing.source}\n{source}",
    )


def parse_lyrics_gap_hints(lyrics: str) -> ParsedLyrics:
    """Strip manual gap controls from lyrics and return their timing metadata.

    Syntax:
      [...]       expected gap, automatic interlude decision
      -[...]      expected gap, never emit an interlude
      +[...]      expected gap, force an interlude
      2+[...]     strength 2 forced gap
      +[...]+[...] equivalent to strength 2 forced gap

    Consecutive marker lines at the same position accumulate strength. Plain
    lyric lines containing "[...]" are left untouched unless the whole line is
    composed only of valid gap-control tokens.
    """
    lyric_lines: list[str] = []
    gaps_by_position: dict[int, GapHint] = {}

    for raw_line in str(lyrics or "").splitlines():
        parsed = _parse_marker_line(raw_line)
        if parsed is not None:
            strength, interlude = parsed
            position = len(lyric_lines)
            gaps_by_position[position] = _merge_gap(
                gaps_by_position.get(position),
                position=position,
                strength=strength,
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

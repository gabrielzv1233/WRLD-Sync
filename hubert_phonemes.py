"""English lyric -> HuBERT FA phoneme bridge.

Display lyrics are never rewritten.  This module builds hidden pronunciation
candidates, chooses a conservative default, keeps word<->phoneme ranges, and
can collapse aligned phoneme intervals back to the original words.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

_WORD_EDGE_RE = re.compile(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$")
_VARIANT_SUFFIX_RE = re.compile(r"\(\d+\)$")
_STRESS_RE = re.compile(r"([A-Z]+)([012])?$")

# Hidden pronunciation-only aliases. One visible lyric token may map to the
# phonemes of multiple spoken words while remaining one display word.
SLANG_PHONES: dict[str, tuple[str, ...]] = {
    # Pronunciation candidates, not semantic expansions. These intentionally
    # model the sounds a singer is likely to produce while leaving display text
    # untouched. Neural G2P is still retained as an alternate candidate.
    "tryna": ("t", "r", "ay", "n", "ax"),
    "finna": ("f", "ih", "n", "ax"),
    "gonna": ("g", "ah", "n", "ax"),
    "wanna": ("w", "aa", "n", "ax"),
    "gotta": ("g", "aa", "t", "ax"),
    "ain't": ("ey", "n", "t"),
    "'cause": ("k", "ax", "z"),
    "cuz": ("k", "ax", "z"),
    "cos": ("k", "ax", "z"),
    "ima": ("ay", "m", "ax"),
    "imma": ("ay", "m", "ax"),
    "shawty": ("sh", "ao", "r", "t", "iy"),
    "lil": ("l", "ih", "l"),
    "ayy": ("ey",),
    "ayyy": ("ey",),
    "yeah": ("y", "ae"),
    "woah": ("w", "ow"),
    "uh": ("ax",),
    "uhh": ("ax",),
}

# Alternate lexical spellings are lower-priority candidates only.
SLANG_ALIASES = {
    "'cause": "cause",
    "cuz": "cause",
    "cos": "cause",
    "shawty": "shorty",
    "woah": "whoa",
}

# Conservative pronunciation-only rewrites commonly found in lyric sheets.
# These never replace the text shown to the user.
LYRIC_ALIASES = {
    "aint": "ain't",
    "bout": "about",
    "cmon": "come on",
    "coulda": "could have",
    "dat": "that",
    "dem": "them",
    "dis": "this",
    "em": "them",
    "errbody": "everybody",
    "gimme": "give me",
    "gon": "gonna",
    "ion": "i don't",
    "jus": "just",
    "kinda": "kind of",
    "lemme": "let me",
    "lotta": "lot of",
    "musta": "must have",
    "ol": "old",
    "outta": "out of",
    "shoulda": "should have",
    "thang": "thing",
    "tho": "though",
    "til": "until",
    "wit": "with",
    "woulda": "would have",
    "wrld": "world",
    "xtc": "ecstasy",
    "yall": "y'all",
}

# CMUdict ARPAbet -> the lowercase English inventory used by HuBERT FA's
# ds_cmudict-07b dictionary. Unstressed AH is schwa (ax); stressed AH remains ah.
_CMU_CONSONANTS = {
    "B": "b", "CH": "ch", "D": "d", "DH": "dh", "F": "f", "G": "g",
    "HH": "hh", "JH": "jh", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t",
    "TH": "th", "V": "v", "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}
_CMU_VOWELS = {
    "AA": "aa", "AE": "ae", "AO": "ao", "AW": "aw", "AY": "ay",
    "EH": "eh", "ER": "er", "EY": "ey", "IH": "ih", "IY": "iy",
    "OW": "ow", "OY": "oy", "UH": "uh", "UW": "uw",
}
_HUBERT_VOWELS = set(_CMU_VOWELS.values()) | {"ah", "ax"}
_HUBERT_GLIDES = {"w", "y"}
_CONTEXT_SENSITIVE_WORDS = {
    "read", "lead", "live", "wind", "bass", "close", "does", "use",
    "tear", "bow", "row", "minute", "object", "record", "present",
    "produce", "refuse", "content", "project", "desert", "invalid",
}


@dataclass(frozen=True)
class PronunciationCandidate:
    phones: tuple[str, ...]
    source: str
    note: str = ""


@dataclass
class LyricWord:
    index: int
    text: str
    lookup: str
    start_char: int
    end_char: int
    candidates: list[PronunciationCandidate] = field(default_factory=list)
    chosen: int = 0
    phoneme_start: int = 0
    phoneme_end: int = 0

    @property
    def phones(self) -> tuple[str, ...]:
        if not self.candidates:
            return ()
        return self.candidates[self.chosen].phones


@dataclass
class PhonemePlan:
    raw_text: str
    words: list[LyricWord]
    phones: list[str]

    @property
    def lab_text(self) -> str:
        return " ".join(self.phones)

    @property
    def word_phone_ranges(self) -> dict[int, tuple[int, int]]:
        return {word.index: (word.phoneme_start, word.phoneme_end) for word in self.words}


@dataclass(frozen=True)
class WordInterval:
    word_index: int
    text: str
    start: float
    end: float
    phones: tuple[str, ...]


def normalize_lookup(text: str) -> str:
    text = str(text or "").replace("’", "'").replace("‘", "'").strip().lower()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = _WORD_EDGE_RE.sub("", text)
    return text


def cmu_phone_to_hubert(phone: str) -> str | None:
    match = _STRESS_RE.fullmatch(str(phone or "").upper())
    if not match:
        return None
    base, stress = match.groups()
    if base == "AH":
        return "ax" if stress in (None, "0") else "ah"
    if base in _CMU_VOWELS:
        return _CMU_VOWELS[base]
    return _CMU_CONSONANTS.get(base)


def cmu_sequence_to_hubert(phones: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for phone in phones:
        mapped = cmu_phone_to_hubert(phone)
        if mapped:
            out.append(mapped)
    return tuple(out)


def load_hubert_dictionary(path: str | Path | None) -> dict[str, list[tuple[str, ...]]]:
    """Load HuBERT FA ds_cmudict format: word<TAB>phone phone ..."""
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    result: dict[str, list[tuple[str, ...]]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or "\t" not in raw:
                continue
            key, value = raw.split("\t", 1)
            key = _VARIANT_SUFFIX_RE.sub("", key.strip().lower())
            phones = tuple(x for x in value.split() if x)
            if key and phones and phones not in result.setdefault(key, []):
                result[key].append(phones)
    return result


@lru_cache(maxsize=1)
def _cmudict_lexicon() -> dict[str, list[list[str]]]:
    # ``cmudict`` is the direct maintained data package. Keep ``pronouncing`` as
    # a compatibility fallback because older WRLD Sync environments may already
    # have it installed.
    try:
        import cmudict
        return cmudict.dict()
    except Exception:
        return {}


def _cmudict_candidates(word: str) -> list[tuple[str, ...]]:
    lexicon = _cmudict_lexicon()
    raw = lexicon.get(word, []) if lexicon else []
    if raw:
        return [p for p in (cmu_sequence_to_hubert(item) for item in raw) if p]
    try:
        import pronouncing
        legacy = pronouncing.phones_for_word(word)
    except Exception:
        return []
    return [p for p in (cmu_sequence_to_hubert(item.split()) for item in legacy) if p]


@lru_cache(maxsize=1)
def _g2p_engine():
    from g2p_en import G2p
    return G2p()


def _g2p_context_groups(words: Sequence[LyricWord]) -> list[tuple[str, ...]]:
    """Run g2p_en once over the line so POS-dependent homographs see context."""
    if not words:
        return []
    text = " ".join(word.lookup or "uh" for word in words)
    try:
        raw = _g2p_engine()(text)
    except Exception:
        return []
    groups: list[list[str]] = [[]]
    for item in raw:
        if item == " ":
            groups.append([])
            continue
        mapped = cmu_phone_to_hubert(item)
        if mapped:
            groups[-1].append(mapped)
    compact = [tuple(group) for group in groups]
    if len(compact) != len(words):
        return []
    return compact


def _alias_phones(alias: str, dictionary: dict[str, list[tuple[str, ...]]]) -> tuple[str, ...]:
    out: list[str] = []
    for part in alias.split():
        candidates = dictionary.get(part) or _cmudict_candidates(part)
        if candidates:
            out.extend(candidates[0])
            continue
        predicted = _g2p_predict(part)
        if not predicted:
            return ()
        out.extend(predicted)
    return tuple(out)


def _append_candidate(word: LyricWord, phones: Sequence[str], source: str, note: str = "") -> None:
    phones = tuple(str(x) for x in phones if x)
    if not phones:
        return
    if any(existing.phones == phones for existing in word.candidates):
        return
    word.candidates.append(PronunciationCandidate(phones=phones, source=source, note=note))


def _lexical_candidates(
    lookup: str,
    dictionary: dict[str, list[tuple[str, ...]]],
) -> list[tuple[str, ...]]:
    """Return dictionary pronunciations for one hidden lookup spelling."""
    return dictionary.get(lookup, []) or _cmudict_candidates(lookup)


def _colloquial_lookup(lookup: str) -> str:
    """Return a conservative standard spelling for a common lyric elision."""
    compact = re.sub(r"[^a-z0-9]", "", lookup)
    if len(compact) > 3 and compact.endswith("in"):
        return f"{compact}g"
    return ""


def _reduced_repetition_lookups(lookup: str) -> list[str]:
    """Offer bounded spellings for expressive forms such as ``yeaaah``."""
    compact = re.sub(r"[^a-z0-9]", "", lookup)
    variants: list[str] = []
    for maximum in (2, 1):
        reduced = re.sub(rf"([a-z])\1{{{maximum},}}", lambda match: match.group(1) * maximum, compact)
        if reduced and reduced != compact and reduced not in variants:
            variants.append(reduced)
    return variants


@lru_cache(maxsize=8192)
def _g2p_predict(lookup: str) -> tuple[str, ...]:
    """Run g2p-en's bundled per-word network without optional NLTK data."""
    lookup = re.sub(r"[^a-z]", "", normalize_lookup(lookup))
    if not lookup:
        return ()
    try:
        return cmu_sequence_to_hubert(_g2p_engine().predict(lookup))
    except Exception:
        return ()


def _token_g2p_phones(lookup: str) -> tuple[str, ...]:
    """Predict each readable component of one OOV display token locally."""
    out: list[str] = []
    components = re.findall(r"[a-z]+|[0-9]", normalize_lookup(lookup))
    digit_names = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    for component in components:
        spoken = digit_names.get(component, component)
        phones = _g2p_predict(spoken)
        if not phones:
            return ()
        out.extend(phones)
    return tuple(out)


def _lookup_with_rewrites(
    lookup: str,
    dictionary: dict[str, list[tuple[str, ...]]],
) -> tuple[tuple[str, ...], str, str] | None:
    """Resolve one hidden spelling through bounded, deterministic rewrites."""
    compact = re.sub(r"[^a-z0-9]", "", lookup)
    attempts: list[tuple[str, str]] = []
    if compact:
        attempts.append((compact, "alphanumeric-compact"))
    colloquial = _colloquial_lookup(lookup)
    if colloquial:
        attempts.append((colloquial, "colloquial-ing"))
    for reduced in _reduced_repetition_lookups(lookup):
        attempts.append((reduced, "expressive-repeat"))
    alias = LYRIC_ALIASES.get(compact)
    if alias:
        phones = _alias_phones(alias, dictionary)
        if phones:
            return phones, "lyric-alias", alias

    seen: set[str] = set()
    for spelling, source in attempts:
        if spelling in seen:
            continue
        seen.add(spelling)
        candidates = _lexical_candidates(spelling, dictionary)
        if candidates:
            return candidates[0], source, spelling
        slang = SLANG_PHONES.get(spelling)
        if slang:
            return slang, source, spelling
    return None


def _add_unresolved_token_fallbacks(
    word: LyricWord,
    dictionary: dict[str, list[tuple[str, ...]]],
) -> None:
    """Retry punctuation-heavy OOVs without changing their display token.

    Hyphens and underscores are first treated as hidden word boundaries, so a
    token such as ``ring-ring`` receives the concatenated pronunciation of two
    ``ring`` lookups.  If that cannot be resolved, punctuation is removed and
    the compact spelling is tried.  In either case the result remains one
    LyricWord and therefore one timed TTML term.
    """
    if word.candidates:
        return

    initialism = re.sub(r"[^A-Za-z]", "", word.text)
    compact_lookup = re.sub(r"[^a-z0-9]", "", word.lookup)
    if 1 < len(initialism) <= 4 and initialism.isupper() and compact_lookup not in LYRIC_ALIASES:
        phones = _alias_phones(" ".join(initialism.lower()), dictionary)
        if phones:
            _append_candidate(
                word,
                phones,
                "initialism",
                f"hidden letter-by-letter pronunciation: {' '.join(initialism)}",
            )

    if word.candidates:
        return

    parts = [part for part in re.split(r"[-_]+", word.lookup) if part]
    if len(parts) > 1:
        phones: list[str] = []
        for part in parts:
            resolved = _lookup_with_rewrites(part, dictionary)
            if not resolved:
                phones = []
                break
            phones.extend(resolved[0])
        if phones:
            _append_candidate(
                word,
                phones,
                "separator-split",
                f"hidden pronunciation lookup: {' '.join(parts)}",
            )

    if word.candidates:
        return
    resolved = _lookup_with_rewrites(word.lookup, dictionary)
    if resolved:
        phones, source, spelling = resolved
        _append_candidate(word, phones, source, f"hidden pronunciation lookup: {spelling}")


def _add_singing_variants(word: LyricWord, max_variants: int = 5) -> None:
    """Add a deliberately small singing-specific insertion/deletion budget.

    Re-articulated held vowels/glides get one- and two-repeat alternatives. A
    single unstressed schwa may also be elided. This is intentionally bounded so
    confidence still means something and the aligner cannot invent an arbitrary
    pronunciation to explain any audio.
    """
    if not word.candidates or max_variants <= 0:
        return
    canonical = word.candidates[0].phones
    eligible = [(idx, phone) for idx, phone in enumerate(canonical) if phone in _HUBERT_VOWELS or phone in _HUBERT_GLIDES]
    made = 0
    for repeat_count in (1, 2):
        for idx, phone in eligible:
            variant = canonical[: idx + 1] + (phone,) * repeat_count + canonical[idx + 1 :]
            _append_candidate(word, variant, "singing-variance", f"{repeat_count} repeated {phone} event{'s' if repeat_count != 1 else ''}")
            made += 1
            if made >= max_variants:
                return
    for idx, phone in enumerate(canonical):
        if phone == "ax" and len(canonical) > 2:
            _append_candidate(word, canonical[:idx] + canonical[idx + 1:], "singing-variance", "one elided unstressed vowel")
            break


def build_phoneme_plan(
    raw_text: str,
    *,
    dictionary_path: str | Path | None = None,
    include_singing_variants: bool = True,
    chooser: Callable[[LyricWord], int] | None = None,
) -> PhonemePlan:
    """Build hidden HuBERT FA input while preserving exact visible lyric tokens.

    Candidate order is conservative: HuBERT's own dictionary, CMUdict, curated
    lyric/slang pronunciation, then context-aware g2p_en / spelling alternates.
    A caller that can score audio
    candidates can provide ``chooser``; otherwise the first curated candidate is
    used, except an OOV will naturally fall through to neural G2P.
    """
    dictionary = load_hubert_dictionary(dictionary_path)
    words: list[LyricWord] = []
    for index, match in enumerate(re.finditer(r"\S+", str(raw_text or ""))):
        text = match.group(0)
        words.append(LyricWord(index, text, normalize_lookup(text), match.start(), match.end()))

    # First use deterministic curated sources. Neural G2P is comparatively
    # expensive, so only run the context model when a word is unknown, slang,
    # or one of the common English homographs whose pronunciation can depend on
    # surrounding words.
    for word in words:
        lookup = word.lookup
        for phones in dictionary.get(lookup, []):
            _append_candidate(word, phones, "hubert-dictionary")
        for phones in _cmudict_candidates(lookup):
            _append_candidate(word, phones, "cmudict")
        if lookup in SLANG_PHONES:
            _append_candidate(word, SLANG_PHONES[lookup], "slang-pronunciation", "curated lyric/slang pronunciation")
        compact = re.sub(r"[^a-z0-9]", "", lookup)
        lyric_alias = LYRIC_ALIASES.get(compact)
        if lyric_alias:
            _append_candidate(
                word,
                _alias_phones(lyric_alias, dictionary),
                "lyric-alias",
                lyric_alias,
            )
        _add_unresolved_token_fallbacks(word, dictionary)

    needs_context = any(
        not word.candidates or word.lookup in SLANG_PHONES or word.lookup in _CONTEXT_SENSITIVE_WORDS
        for word in words
    )
    context_groups = _g2p_context_groups(words) if needs_context else []

    for word in words:
        lookup = word.lookup
        if context_groups:
            _append_candidate(word, context_groups[word.index], "g2p-context", "POS-aware context / neural OOV fallback")
        alias = SLANG_ALIASES.get(lookup)
        if alias:
            _append_candidate(word, _alias_phones(alias, dictionary), "slang-alias", alias)
        if not word.candidates:
            _append_candidate(
                word,
                _token_g2p_phones(lookup),
                "g2p-token",
                "local per-token OOV fallback (no external language model)",
            )
        if include_singing_variants:
            _add_singing_variants(word)
        if not word.candidates:
            raise ValueError(f"Could not produce a pronunciation for lyric word {word.text!r}")
        if chooser is not None:
            chosen = int(chooser(word))
            if 0 <= chosen < len(word.candidates):
                word.chosen = chosen

    flattened: list[str] = []
    for word in words:
        word.phoneme_start = len(flattened)
        flattened.extend(word.phones)
        word.phoneme_end = len(flattened)
    return PhonemePlan(raw_text=str(raw_text or ""), words=words, phones=flattened)


def collapse_phoneme_intervals(
    plan: PhonemePlan,
    intervals: Sequence[tuple[str, float, float] | dict],
) -> list[WordInterval]:
    """Collapse HuBERT phoneme timestamps back to the ORIGINAL display words."""
    normalized: list[tuple[str, float, float]] = []
    for item in intervals:
        if isinstance(item, dict):
            phone = str(item.get("phone") or item.get("text") or item.get("label") or "")
            start = float(item.get("start", item.get("begin", 0.0)))
            end = float(item.get("end", start))
        else:
            phone, start, end = item
            phone, start, end = str(phone), float(start), float(end)
        # HuBERT outputs can include detected silence/breath labels. They are not
        # part of word pronunciation ranges, so callers should remove them before
        # exact reconstruction rather than letting them shift every later word.
        normalized.append((phone, start, end))
    if len(normalized) != len(plan.phones):
        raise ValueError(f"Expected {len(plan.phones)} lexical phoneme intervals, got {len(normalized)}")

    result: list[WordInterval] = []
    for word in plan.words:
        chunk = normalized[word.phoneme_start:word.phoneme_end]
        if not chunk:
            continue
        result.append(WordInterval(
            word_index=word.index,
            text=word.text,
            start=chunk[0][1],
            end=max(x[2] for x in chunk),
            phones=word.phones,
        ))
    return result

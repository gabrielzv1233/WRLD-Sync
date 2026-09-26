# Manual lyric gap hints

Gap hints are optional timing controls in the raw lyrics used by **Sync** when you know there is a real break between two lyric sections. They can occupy their own line or be appended to the end of the lyric immediately before the gap.

They are control metadata, not lyrics. WRLD Sync removes them before forced alignment, and they do not appear as lyric text in Preview or generated TTML.

## Syntax

```text
{seconds}{mode}[...]
```

Both `seconds` and `mode` are optional.

| Syntax | Meaning |
| --- | --- |
| `[...]` | Minimum 0.25 second gap, automatic interlude decision |
| `.25[...]` | Same as `[...]` |
| `0.25[...]` | Same as `[...]` |
| `2[...]` | Minimum 2 second gap |
| `11.5[...]` | Minimum 11.5 second gap |
| `+[...]` | Minimum 0.25 second gap and force an interlude |
| `11.5+[...]` | Minimum 11.5 second gap and force an interlude |
| `11.5-[...]` | Minimum 11.5 second gap and never create an interlude |
| `0[...]` | Alignment boundary with no minimum time |

Decimals can be written with or without a leading zero, so `.5[...]` and `0.5[...]` are equivalent.

## What the number means

The number is the **minimum amount of time WRLD Sync must move forward after the previous aligned lyric ends before the next lyric is allowed to begin**.

It is not the final duration of the gap.

For example:

```text
I know I'm not right
10[...]
But I'm not wrong
```

If the first line ends at `1:12.000`, the next lyric cannot be placed before `1:22.000`.

After that boundary, WRLD Sync searches its timestamped transcription for the first words of the next lyric. If the real vocal begins at `1:28.400`, it can still align at `1:28.400`. The `10` does not force the lyric to start exactly ten seconds later.

Bare `[...]` is shorthand for `0.25[...]`.

## Interlude mode

The optional sign controls only interlude output:

- no sign, such as `4[...]`: normal automatic interlude detection
- `+`, such as `4+[...]`: force the real detected gap to be emitted as an interlude
- `-`, such as `4-[...]`: never emit that gap as an interlude

The sign does not change the minimum number of seconds.

## Example

```text
First verse line
Last line before the instrumental
11.5+[...]
First line after the instrumental
Next line
```

For this boundary, WRLD Sync will not accept the next lyric during the first 11.5 seconds after the previous lyric ends. It then looks for the actual next lyric, aligns the following section from there, and marks the real detected gap as an interlude because `+` was used.

## Multiple hints at the same boundary

If several gap-hint lines appear consecutively, they all describe the same boundary. WRLD Sync keeps the **largest minimum**, rather than adding the numbers.

```text
Line one
0.5[...]
3-[...]
Line two
```

This becomes a 3 second minimum with interludes forbidden.

An automatic hint does not cancel an explicit `+` or `-` policy. Conflicting explicit policies at the same boundary are rejected.

The old repeated-token form such as `+[...]+[...]` is not part of the literal-seconds syntax. Write the desired duration directly instead.

## Where hints are recognized

A hint can occupy its own line:

```text
Last line before the gap
2-[...]
First line after the gap
```

or it can be appended directly to the end of the lyric immediately before the gap:

```text
Last line before the gap 2-[...]
First line after the gap
```

Both forms describe the same boundary. The inline control is stripped before alignment, so the lyric becomes just `Last line before the gap`.

Marker-looking text in the **middle** of a lyric is still left alone because treating it as a boundary would be ambiguous:

```text
I waited [...] forever
```

That line remains normal lyric text.

## Leading hints and opening instrumentals

A leading hint can delay the first lyric from the beginning of the audio:

```text
[...]
First lyric
```

The number still acts as the minimum forward offset before the first lyric may be accepted.

Opening Instrumental output is intentionally conservative:

- a leading automatic hint such as `[...]` or `1[...]` emits an opening Instrumental only when the actual resolved first lyric starts at least **3 seconds** into the song
- a shorter automatic opening gap emits no Instrumental metadata, so `[...]` can fix a small early offset without creating a visible interlude
- `1+[...]` (or `+[...]`) forces an opening Instrumental even when the resolved opening is shorter than 3 seconds
- `-[...]` always forbids an opening Instrumental
- disabling interlude detection suppresses automatic opening Instrumentals, but an explicit `+` still forces one

A trailing hint is retained as metadata, but with no following lyric it does not create another alignment section.



## Inline alignment and background-role controls

These controls separate two independent decisions: whether text is sent to the
alignment model, and whether its output is forced to Apple `x-bg`.

| Syntax | Sent to aligner? | Background role |
| --- | --- | --- |
| `-\{text}` | No | Automatic, using the normal parenthesis rules |
| `+\{text}` | No | Forced `x-bg` |
| `+{text}` | Yes | Forced `x-bg` |
| `-{text}` | No | Forced `x-bg` (legacy alias for `+\{text}`) |

The braces, sign, and backslash are editor controls only. They do not appear in
Preview, generated TTML/LRC, copied output, or proposals.

### Skip alignment but keep automatic background detection

```text
Main lyric -\{(background ad-lib)}
```

The model receives only `Main lyric`. The parenthesized fragment is restored
afterward and the existing parenthesis logic can render it as background. Plain
text such as `-\{spoken note}` is also withheld from alignment but is not
forced to background.

Because a backslash form is never aligned, WRLD Sync does not invent precise
word timing for it. The restored fragment spans the containing lyric line.

### Skip alignment and force background

```text
Main lyric +\{background words without parentheses}
```

The model receives only `Main lyric`, while the restored text is always emitted
as background/`x-bg`, regardless of parentheses or the inline-parenthetical
toggle.

### Align normally and force background

```text
Main +{oddly written background lyric} lyric
```

The model receives `Main oddly written background lyric lyric`. Those words keep
their real aligned word timestamps, but WRLD Sync forces that controlled range
to `x-bg`. This is useful when parentheses would look wrong in the displayed
lyrics but the vocal is still semantically background.

A non-aligned control must share its line with some normal alignable lyric text.
A line made only from `+\{...}` or `-\{...}` is rejected because there is no
anchor from which to obtain the line timing. A line made entirely from
`+{...}` is valid because that text is sent to the aligner.

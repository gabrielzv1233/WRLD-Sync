this file lists changes I have made whilest you (codex) have been inactive
> one thing I actually want u to add is the project storing already processed lyrics in a database, than storing its processing settings in it aswell, using the song ID as the database ID/key

## 2026-08-29 Auto-follow root-cause / hard shutdown fix

- Fixed the actual reason Preview auto-scroll could do nothing in Chromium/Opera GX. The horizontal-overflow guard applied `overflow-x:hidden` to `.lyrics-body`; per CSS overflow-axis rules that implicitly computed its `overflow-y` to `auto`, making `.lyrics-body` a second hidden vertical scroller. All camera code was correctly moving `#lyricsSection`, but `#lyricsSection` had no scroll range. Descendant lyric containers now use `overflow-x:clip` / visible Y overflow so `#lyricsSection` is the one and only lyric scroller.
- Verified the corrected layout in Chromium with a 50-line synthetic lyric set: `#lyricsSection` now owns the full overflow, a mid-song target produces a real nonzero camera destination, and the animated controller lands on that destination.
- Reworked the follow controller to continuously reconcile the active/nearest lyric while playback is running instead of depending only on media/line-change events. This also repairs camera position after layout reflow.
- Preview tab restoration no longer holds auto-follow indefinitely on the restored scroll position. It restores the exact saved position first, waits only 260 ms, then smoothly resumes following the current lyric if auto-follow is enabled.
- Camera targeting keeps the active lyric or overlap group centered within the usable viewport whenever physically possible. Two active lines use their midpoint, larger overlap groups use the median row center, and destinations are hard-clamped at the content edges.
- Added a natural lower scroll bound based on the actual lyric content so the camera does not intentionally chase artificial bottom padding merely to mathematically center the final lyric.
- Tightened follow motion to about 2.25 px/ms with a 170 ms minimum and 850 ms maximum. The easing now has an intentionally tiny ~4.5% acceleration phase, near-constant middle travel, and a longer 24% deceleration phase for a fast but polished settle.
- Resume auto-scroll, Play, Seek, viewport resize, Preview return, and newly rendered lyrics all explicitly start/restart the follow controller. Resume/seek use the timed lyric closest to current playback time and always animate instead of teleporting.
- Fixed launcher shutdown more aggressively. `launch.py` no longer blocks indefinitely inside `Popen.wait()`; it installs explicit SIGINT/SIGBREAK handling, polls the server process, gives Uvicorn only a short graceful CTRL_BREAK/SIGINT window, then uses a whole-process-tree hard kill on Windows (`taskkill /T /F`) if Whisper/CTranslate2/ffmpeg/native work is stuck.
- `wait_for_server()` now aborts immediately when shutdown is requested, so Ctrl+C also works during server startup.
- Simplified `start.bat` to prefer directly installed Python before the uv wrapper, reducing another layer that can interfere with console signals, and removed the unconditional final `pause` that previously made a successful shutdown look like the app was still hanging.

## 2026-08-29 Deterministic Preview auto-follow controller

- Replaced the remaining event-only Preview auto-scroll path with a continuously reconciled follow controller. While Preview is visible and auto-follow is enabled, every karaoke frame computes the current intended lyric target, but an in-flight animation is only restarted when that destination actually changes. This removes the failure mode where highlighting updated first and the later scroll transition was never triggered.
- Auto-follow now centers against the genuinely visible lyric viewport rather than raw `lyricsSection.clientHeight`: it accounts for the sticky Preview/Lyrics/TTML tabs at the top and the floating player controls overlay at the bottom.
- Follow destinations are hard-clamped to the real `0..scrollHeight-clientHeight` range. First/last lyrics therefore settle at the closest physically reachable position instead of trying to manufacture whitespace just to achieve mathematical centering.
- Explicitly resuming auto-follow, starting playback, or seeking now re-anchors by smoothly scrolling to the lyric interval closest to the current playback time. These actions never snap/teleport unless the destination is already within roughly one pixel.
- Normal playback keeps the latest started lyric through tiny between-line gaps, while explicit re-anchors use the closest timed lyric. Instrumental gaps continue to target their interlude marker when one is actually rendered.
- Overlapping active lyrics remain a group target: two active rows use their exact midpoint, while larger simultaneous groups use the median row center.
- Replaced symmetric smoothstep travel with an asymmetric near-constant-speed profile: a very short 8% acceleration phase, a long cruise phase, and a softer 26% deceleration phase. Duration is distance-based at roughly 1.8 px/ms with a 210 ms minimum for short lyric hops and an 1100 ms cap only for extreme jumps. This makes nearby moves visibly animated without making long seeks crawl.
- Manual wheel/touch input still cancels the current animation and disables auto-follow. Clicking `Resume auto-scroll` clears any held camera state and performs the new animated closest-lyric re-anchor immediately.
- Preserved the earlier Preview-tab scroll-position behavior: returning from Lyrics/TTML restores the exact saved Preview position and temporarily holds it until the active lyric group genuinely changes. Explicit Resume/play/seek overrides that hold immediately.

## 2026-08-29 Overlap renderer / group auto-follow fix

- Reworked Preview overlap rendering so simultaneous lyric lines no longer all behave like the single primary row. Every timed-overlap line stays fully active and keeps its word animation, while only the newest/pinned line receives the full primary emphasis and persistent timestamp/delete actions. Older still-active lines use a smaller concurrent emphasis so multiple highlighted rows do not visually fight or shift the layout.
- Added overlap-group-aware auto-follow. While two active lyric rows overlap, the scroller targets the midpoint between their row centers; for larger overlap stacks it uses the median row center so a wrapped/tall line cannot drag the viewport away from the active group.
- Auto-follow now responds to overlap-group membership changes as well as new lyric starts. When a second line joins it smoothly centers the active group, and when an older sustained line ends it smoothly recenters on the remaining active lyric instead of staying stranded between the old pair.
- Preserved the existing 420-820 ms custom smoothstep follow animation, manual-scroll cancellation, Preview-tab scroll persistence, and interlude targeting.

## 2026-08-29 Settings header removal / overlapping lyric playback

- Removed `settings-modal-head` completely, including the settings icon, title, subtitle, and modal X. The settings sections now begin immediately at the top of the centered modal. Settings still closes from its toolbar toggle, backdrop, or Escape key.
- Added a persisted `Allow overlapping lyrics` setting, disabled by default so Preview behavior is unchanged unless explicitly enabled.
- With overlap mode enabled, Preview treats every lyric line whose real `[start, end)` range contains the current playback time as active simultaneously instead of forcing the newest-started line to visually replace an older line that is still singing. This covers sustained final words crossing into the next line and separate foreground/background lines that overlap.
- Per-word animation now runs on every simultaneously active line while overlap mode is enabled, so an older sustained word can continue filling while the next line begins. Inline `x-bg` spans continue to animate independently inside their containing line.
- Overlap mode never fabricates or stretches timestamps. It only exposes overlaps already produced by Whisper, imported TTML, or manual TTML/timing edits.
- Auto-follow still tracks the newest-started lyric line. Ending an older overlapping line updates its visual state without triggering a redundant second scroll to the same newer line.
- Interlude detection/rendering becomes overlap-aware when overlap mode is enabled: an instrumental gap cannot begin until every already-started overlapping vocal range has ended.
- TTML section bounds are now safe for overlapping lines regardless of the toggle: a parent lyric `<div>` keeps the maximum end time of all child `<p>` lines, so a shorter newer line can never truncate an older sustained line. Instrumental TTML sections likewise start after the latest overlapping vocal end.
- The overlap preference is stored in localStorage, queue settings, and catalog/local processed-lyrics SQLite settings so it follows cached processing state.

## 2026-08-29 Settings UI redesign

- Changed Task Queue dismissal so clicks elsewhere on the page no longer close it. The queue now closes only from its X button or by clicking the Task Queue toolbar button again, which acts as a true open/close toggle.
- Rebuilt Settings as a centered modal instead of the old upper-right popout. It now uses a dedicated dim/blur backdrop, centered responsive sizing, a fixed title/footer, a scrollable settings body, and animated open/close transitions.
- Reorganized the contents into clear API access, Whisper, Compute device, and Apple TTML sections with explanatory secondary text instead of the previous flat collection of generic bordered rows.
- Replaced browser-default-looking model/engine selects with custom select shells, consistent dark surfaces, focus rings, hover states, spacing, and custom chevrons.
- Rebuilt the API token control as one integrated input with lock/reveal affordances and a dedicated primary Save token button. Saving the token no longer closes the entire Settings modal.
- Rebuilt Auto/GPU/CPU selection as a three-way segmented device control with icons, subtitles, and a distinct animated selected state.
- Replaced plain TTML checkboxes with custom animated switch controls, full-row click targets, and concise descriptions of each behavior.
- Added modal interaction polish: backdrop-click dismissal, Escape dismissal, focus restoration to the previously focused control, Tab focus containment, responsive one-column behavior, and body scroll locking while Settings is open.
- Improved status presentation for model/device saves and token saves, including clearing stale error coloring after a successful model setting update.


## 2026-08-29 Preview scroll / inline background toggle correction

- Corrected the catalog search width from the accidental 1040px maximum back to 520px, which is the intended 2x size rather than the previous 4x expansion.
- Preview, Lyrics, and TTML now keep independent scroll positions. Leaving Preview records its exact `lyricsSection.scrollTop`; switching back restores that exact position instead of auto-centering the currently playing line just because the tab changed.
- Auto-follow no longer scrolls the shared lyric container while Lyrics or TTML is visible, so playback continuing in the background cannot overwrite those tab positions or the saved Preview position. Normal Preview auto-follow resumes on the next real lyric/interlude transition.
- Added a persisted `Inline (…) as background vocals` TTML setting. When enabled, embedded parenthetical phrases such as `I've been through this a thousand (thousand, thousand, thousand)` can render/export as Apple `ttm:role="x-bg"`; when disabled, that embedded phrase stays ordinary timed words in the same lyric line.
- Parenthetical-only lyric lines remain eligible background vocals regardless of the inline toggle, while explicit `x-bg` imported from TTML is always preserved.
- Background classification now records its source (`explicit`, `parenthetical-line`, or `parenthetical-inline`) so the inline setting can be toggled live without rerunning Whisper. The setting is stored in localStorage, queue processing settings, song/local-track SQLite processing records, and proposal TTML generation.

## 2026-08-29 Preview layout / auto-follow fix

- Fixed Preview `kl-actions` positioning by moving lyric rows to a two-column text/actions layout. The active-row emphasis now scales only the lyric content, not the action controls, so timestamp/delete controls no longer get shifted or clipped by the lyric container's horizontal overflow guard.
- Restored and hardened Preview auto-follow. Playback immediately re-anchors the current lyric when starting, returning to the Preview tab re-anchors the current lyric, and deliberate wheel/touch movement pauses follow without the old stale-scroll flag that could disable it on a later programmatic scroll. Scrolling inside Lyrics/TTML no longer disables Preview auto-follow.
- Tightened Preview vertical rhythm by reducing the line gap and row padding while keeping a small amount of separation between lyrics.
- Increased desktop Preview lyric size from 1.12rem to 1.17rem and slightly increased the mobile size, while keeping active-line emphasis subtle and applied only to the lyric content.

## 2026-08-29 local audio / sidebar pass

- Redesigned Local Audio as its own full-width landing view. The top-bar and landing-page Open Audio actions now switch to a dedicated screen with two paths: browse/upload a local audio file or paste a direct HTTP(S) audio URL. After loading, the app returns to the normal player in a local-file-specific state.
- Local files now read embedded title, artist, duration, and cover artwork with Mutagen. The player keeps the duration badge, hides the era badge, and forces the category badge to `Local file`.
- Added decoded-audio identity for local files: WRLD Sync hashes canonical decoded PCM audio with SHA-256, so filename, ID3 tags, embedded artwork, and other container metadata do not change the local track key.
- Added separate SQLite tables for local audio: `local_tracks` stores the audio hash plus local metadata/path/artwork, while `local_processed_lyrics` stores timed lyrics/TTML/settings keyed by that audio hash. Catalog `processed_lyrics` remains keyed by Juice WRLD API song ID.
- Local queue tasks now carry the audio hash, which prevents completed/live tasks for one local file from being applied to a different local file and allows successful Auto/Sync results to persist automatically to the local-file table. Manual Preview/TTML saves now persist for local tracks too.
- Added local URL downloading with redirect support and a 2 GB safety limit. Remote audio is downloaded to the local server before hashing/tag extraction/Whisper processing.
- Replaced the catalog search glyph with Google Material Symbols Rounded `search` at weight 300 / grade 0. This is intentionally the one non-Feather header icon exception. The empty-state Search action uses the same glyph.
- Removed the `logo-mark` element entirely and doubled the desktop search field's maximum width from 520px to 1040px.
- Updated the empty-state Random and Open Audio icons to exactly mirror their top-bar controls.
- Reworked the collapsed results-panel reveal control into a borderless edge handle. A muted 8x120px bar stays 8px from the left edge while collapsed; moving the pointer into the left eighth of the viewport reveals its chevron, hover turns it fully white, opening the results pane quickly fades/slides it underneath, and collapsing the pane returns it with a slight overshoot before settling. The control has no pointer events whenever it should not be available.
- Softened and enlarged the center glow on `emptyState` to remove visible radial color banding and make the falloff smoother.
- Fixed launcher Ctrl+C shutdown hangs by starting Uvicorn in its own process group and using a bounded interrupt -> terminate -> kill shutdown sequence, so native Whisper/executor threads cannot leave the launcher waiting indefinitely.

## 2026-08-29 editor / workflow pass

- Reworked Preview auto-follow scrolling to use a custom requestAnimationFrame animation instead of native `scrollIntoView`. Every next-line/interlude transition now has a 420 ms minimum duration (up to 820 ms for larger jumps) and a smoothstep easing curve with zero initial/final velocity, so nearby lines cannot visually snap into place; manual wheel/touch input cancels the in-flight animation immediately.
- Renamed the `Synced` tab to `Preview` and scoped its keyboard shortcuts to that tab only. Up/Down now move to previous/next lyric, Left/Right seek -/+ 1 second, Ctrl+Up/Down pin previous/next lyric, Shift+Left jumps/replays from the current line start, Shift+Right jumps to its end, and Space toggles playback. Removed C, V, X, Ctrl+Space, and the global Shift auto-scroll shortcut so typing/copy/paste are never intercepted.
- Removed the lyrics offset control UI and its global +/-0.1 second controls.
- Added a first-class `TTML` tab with an editable Apple TTML source view. `Save TTML` parses the edited document, applies it back to Preview, and persists it to the local processed-lyrics database; `Refresh from Preview` regenerates TTML from the current timed lines.
- Changed Auto into a true full Whisper transcription run. Auto now ignores preloaded raw lyrics, transcribes the audio with the selected Whisper model/engine, streams Faster-Whisper segments live when available, fills the raw Lyrics tab, and keeps the transcription timestamps for Preview/TTML.
- Sync is now explicitly alignment-only: it uses the current raw Lyrics textarea and aligns those lyrics to the song without doing a free transcription pass.
- Integrated manual Preview/TTML editing with the SQLite processed-lyrics cache. Queue results still save automatically, while line text/timestamp edits, line deletion, and TTML saves persist back to the current song-ID record. Stored per-song TTML/interlude settings are restored when cached lyrics are loaded.
- Added persistent player volume using localStorage.
- Fixed first-run volume restoration so a missing localStorage value no longer gets coerced to `0` and mutes the player.
- Exact imported TTML is now preserved as the stored source instead of being immediately regenerated by the Preview autosave path.
- The now-playing title is muted while paused/stopped and smoothly fades to full white only while that loaded track is actively playing.
- Fixed horizontal overflow in the lyric body/Preview and constrained long lyric text so it wraps instead of creating a horizontal page scroller.
- Search is now stored in the page URL as `?q=...` using `history.replaceState`; reloading or sharing the page restores and reruns the same catalog search.
- Hardened the catalog search field against Opera GX/Chromium password-manager misclassification with search-specific naming/autocomplete hints plus a readonly-until-focus/pointer interaction guard.
- Reworked the header so the navigation cluster is centered relative to the full header while the polished WRLD Sync logo stays pinned left. Control order is now Filter, Random, Load File, Search, Task Queue, Settings, Database Manager; header dividers were removed.
- Replaced the Auto symbol with the Feather `zap` SVG and repaired the Random button to use the complete Feather `shuffle` shape.
- Turned Settings into an organized floating popout instead of a full-width strip.
- Reworked the collapsed-sidebar reveal into an enlarged arrow-only edge control that mostly hides off-canvas until hovered/focused, then eases outward with an animated translucent circular background.
- Removed the landing-page kicker and changed its description to: `Search the Juice WRLD catalog, choose something at random, or bring your own audio. Whisper models run locally on your machine.`
- Changed the Database Manager page-title chip radius to 8px.

# Implemented changes

- Switched generated/proposed synced lyrics to Apple-style TTML with explicit line `begin` and `end` times.
- Added optional real per-word TTML timing using stable-ts word timestamps.
- Added automatic interlude/instrumental gap markers for gaps of at least 2 seconds.
- Kept LRC import and export for compatibility and other submissions/use cases.
- Added TTML import support alongside LRC.
- Added 1 ms separation between adjacent line timings so single-performer lines do not overlap.
- Added Fast Auto, enabled by default, which uses a single stable-ts forced-alignment pass instead of the old verify + align double pass.
- Enabled stable-ts `fast_mode` and a larger alignment token step in Fast Auto.
- Added faster-whisper/CTranslate2 as the default inference engine with PyTorch Whisper retained as a compatibility toggle.
- Verification transcription no longer computes unused word timestamps/silence post-processing.
- Preserved word timing when globally nudging or manually shifting synced lines.
- Fixed lyric-line text editing so Space/Enter shortcuts no longer intercept typing inside the inline editor.
- Added live partial transcription for Faster-Whisper: decoded segments are pushed through the queue SSE stream and appear in the lyric UI while the model is still running.
- Added Apple Music-style per-word playback rendering using the real TTML/Whisper word timestamps, with smooth left-to-right word brightening, subtle word growth, active-line enlargement, dim/blurred future lines, and smooth auto-follow.
- Refined Whisper/TTML timestamp cleanup so model-produced line and word timings preserve natural silence gaps instead of forcing adjacent words/lines to touch. The old 1 ms non-overlap workaround is now limited to legacy LRC parsing/export compatibility.
- Fast Auto now uses a less aggressive 2-second `nonspeech_skip` threshold; strict alignment retains the stable-ts 5-second default-style threshold so short musical/speech pauses are not skipped as aggressively.
- Added an Apple Music-style ambient lyric background using the song artwork with blur, saturation, darkening, and subtle motion behind the lyric view.
- Added Apple Music-style interlude playback rendering with animated three-dot waiting indicators that progress left-to-right across detected instrumental gaps.
- Updated per-word playback highlighting so real silence between words remains visually unfilled instead of stretching one word's highlight until the next begins.
- Split CUDA capability detection by inference backend: Faster-Whisper checks CTranslate2's CUDA device availability, while the PyTorch engine checks `torch.cuda`.
- Fixed the launcher leaving an existing CPU-only PyTorch wheel installed on NVIDIA systems. It now force-reinstalls from the appropriate CUDA wheel index when needed, verifies the resulting CUDA build/runtime, and skips CUDA installation when no NVIDIA GPU is detected.
- Cleaned terminal output by suppressing raw stable-ts/tqdm stderr progress, rendering model work with Rich progress bars, decoding song/file names for readable status output, and disabling noisy Uvicorn request access logs.
- Added a local SQLite processed-lyrics cache at `data/wrld_sync.sqlite3`, keyed by song ID. Successful processed results store the source lyrics, timed lines/words, generated TTML, source/task type, processing timestamps, engine/model/device choices, Fast Auto state, word/interlude settings, and alignment parameters.
- Loading a song now checks the local processed-lyrics database first and restores its cached synced result/settings when present, avoiding unnecessary reprocessing.
- Added Apple TTML background-vocal support: parenthesized/ad-lib word runs are preserved as background words and exported as nested `ttm:role="x-bg"` spans; imported Apple TTML `x-bg` spans are also recognized and rendered with inset/subdued background-vocal styling.
- Interlude TTML now uses explicit `itunes:song-part="Instrumental"` sections for detected gaps instead of manufacturing empty lyric paragraphs.
- On Windows, when the repaired CUDA PyTorch wheel contains cuBLAS/cuDNN runtime DLLs, the backend exposes its `torch/lib` directory to CTranslate2 so Faster-Whisper can reuse those CUDA 12 libraries instead of falling back because the DLLs are not discoverable.

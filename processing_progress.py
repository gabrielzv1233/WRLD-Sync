"""Measured model progress and stage ETA, independent of inference libraries."""
from collections import deque
import json
import math
from pathlib import Path
import statistics
import threading
import time


def clock_label(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '—'
    seconds = max(0, math.ceil(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02}:{seconds:02}' if hours else f'{minutes:02}:{seconds:02}'


class RuntimeHistory:
    """Keep recent successful seconds-per-audio-second samples by model/device/stage."""
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            self.samples = {
                k: [float(x) for x in v if isinstance(x, (float, int)) and math.isfinite(x) and x > 0][-8:]
                for k, v in data.items() if isinstance(v, list)
            }
        except (OSError, ValueError, TypeError, AttributeError):
            self.samples = {}

    def estimate(self, key, duration):
        with self.lock:
            samples = self.samples.get(key, [])
            return statistics.median(samples) * duration if samples and duration > 0 else None

    def record(self, key, duration, elapsed):
        if duration <= 0 or elapsed <= 0 or not math.isfinite(duration + elapsed):
            return
        with self.lock:
            self.samples[key] = [*self.samples.get(key, []), elapsed / duration][-8:]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.path.with_suffix('.tmp')
                temp.write_text(json.dumps(self.samples), encoding='utf-8')
                temp.replace(self.path)
            except OSError:
                pass  # Progress history must never prevent inference.


class ProcessingProgress:
    def __init__(self, history, key, duration, cancel=lambda: False, clock=time.perf_counter):
        self.history, self.key, self.duration = history, key, duration
        self.cancel, self.clock = cancel, clock
        self.lock = threading.RLock()
        self.phase = None
        self.high_water = 0.0
        self.finished = False

    def check_cancel(self):
        if self.cancel():
            raise InterruptedError('Cancelled')

    def begin(self, phase, label, start, end, *, total=None, unit='seconds'):
        self.check_cancel()
        with self.lock:
            if self.phase is not None:
                self._record()
            self.phase, self.label = phase, label
            self.start, self.end = start, end
            self.total = total if total and math.isfinite(total) and total > 0 else None
            self.unit, self.done = unit, 0.0
            self.started = self.clock()
            self.samples = deque([(self.started, 0.0)], maxlen=12)
            self.rate = None
            self.estimate = self.history.estimate(f'{self.key}:{phase}', self.duration)
            self.high_water = max(self.high_water, start)

    def update(self, done, total):
        self.check_cancel()
        if not math.isfinite(done) or not math.isfinite(total) or total <= 0:
            return
        with self.lock:
            self.total = total
            done = min(total, max(self.done, done))
            if done > self.done:
                now = self.clock()
                self.samples.append((now, done))
                first_time, first_done = self.samples[0]
                if now - first_time >= 1:
                    measured = (done - first_done) / (now - first_time)
                    self.rate = measured if self.rate is None else .3 * measured + .7 * self.rate
                self.done = done

    def _record(self):
        if not self.cancel():
            self.history.record(f'{self.key}:{self.phase}', self.duration, self.clock() - self.started)

    def finish(self):
        self.check_cancel()
        with self.lock:
            self._record()
            self.finished = True

    def snapshot(self):
        with self.lock:
            elapsed = max(0.0, self.clock() - self.started)
            measured = self.total is not None
            fraction = self.done / self.total if measured else None
            speed, eta = self.rate, None
            estimated = False
            if measured and speed and self.done < self.total:
                # If the decoder stalls, extend the estimate instead of counting down to zero.
                last_time = self.samples[-1][0]
                interval = max(1.0, (last_time - self.samples[0][0]) / max(1, len(self.samples) - 1))
                speed *= min(1.0, interval / max(interval, self.clock() - last_time))
                eta = (self.total - self.done) / speed
            elif not measured and self.estimate:
                estimated = True
                fraction = min(.95, elapsed / self.estimate)
                if elapsed < self.estimate:
                    eta = self.estimate - elapsed
            elif measured and not speed and self.estimate and elapsed < self.estimate:
                eta, estimated = self.estimate - elapsed, True
            # 100% is reserved for returned results, including timestamp postprocessing.
            if self.finished:
                fraction, eta = 1.0, 0.0
            phase_pct = (100 if self.finished else min(99.0, fraction * 100)) if fraction is not None else None
            overall = self.end if self.finished else self.start + (self.end - self.start) * min(.99, fraction or 0)
            self.high_water = max(self.high_water, overall)
            cancelling = self.cancel()
            return {
                'stage': self.phase, 'step': 'aligning' if 'align' in self.phase else 'transcribing',
                'pct': self.high_water, 'indeterminate': fraction is None,
                'msg': 'Cancelling…' if cancelling else self.label,
                'progress': {
                    'done': self.done if measured else None, 'total': self.total,
                    'unit': self.unit, 'elapsed_seconds': elapsed,
                    'eta_seconds': None if cancelling else eta,
                    'eta': clock_label(None if cancelling else eta),
                    'speed': speed, 'estimated': estimated, 'phase_pct': phase_pct,
                    'eta_scope': 'stage',
                },
            }

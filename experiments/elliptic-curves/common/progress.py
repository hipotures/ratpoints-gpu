"""Small dependency-free progress reporter for the elliptic-curve experiments."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@dataclass
class Progress:
    """Rate-limited progress output on stderr (or another supplied stream)."""

    stage: str
    total: int | None = None
    stream: TextIO = sys.stderr
    interval: float = 0.35

    def __post_init__(self) -> None:
        self.started = time.monotonic()
        self.last_update = 0.0
        self.last_completed = 0
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._write(self._line(0, {}), final=False, force_newline=True)

    def _line(self, completed: int, counters: dict[str, int]) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = completed / elapsed
        pieces = [f"[{_duration(elapsed)}]", self.stage]
        if self.total is not None and self.total > 0:
            fraction = min(max(completed / self.total, 0.0), 1.0)
            pieces.extend([f"{100.0 * fraction:5.1f}%", f"{completed:,}/{self.total:,}"])
            if completed > 0 and rate > 0 and completed < self.total:
                pieces.append(f"ETA {_duration((self.total - completed) / rate)}")
        else:
            pieces.append(f"{completed:,}")
        if completed:
            pieces.append(f"{rate:,.0f}/s")
        pieces.extend(f"{key}={value:,}" for key, value in counters.items())
        return "  ".join(pieces)

    def _write(self, text: str, *, final: bool, force_newline: bool = False) -> None:
        if self.tty and not force_newline:
            suffix = "\n" if final else ""
            self.stream.write("\r\x1b[K" + text + suffix)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def update(self, completed: int, *, force: bool = False, **counters: int) -> None:
        now = time.monotonic()
        if not force and now - self.last_update < self.interval:
            return
        self.last_update = now
        self.last_completed = completed
        self._write(self._line(completed, counters), final=False)

    def done(self, completed: int | None = None, **counters: int) -> None:
        if completed is None:
            completed = self.total if self.total is not None else self.last_completed
        self.last_completed = completed
        self._write(self._line(completed, counters), final=True)

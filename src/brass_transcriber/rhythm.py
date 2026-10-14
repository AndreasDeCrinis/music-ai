from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, model_validator


class TempoSection(BaseModel):
    quarter: float = Field(ge=0)
    bpm: int = Field(ge=20, le=400)


class Rhythm(BaseModel):
    tempo: float
    detected_tempo: float | None
    beats: list[float]
    time_signature: str = "4/4"
    grid: str = "1/16"
    pickup: float = 0
    warnings: list[str] = Field(default_factory=list)
    mode: str = "constant"
    downbeats: list[float] = Field(default_factory=list)
    grid_times: list[float] = Field(default_factory=list)
    grid_positions: list[float] = Field(default_factory=list)
    tracker: str = "librosa"
    tracker_device: str = "cpu"
    tracker_runtime: float = 0
    tempo_sections: list[TempoSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_map(self):
        if self.tempo_sections and (
            self.tempo_sections[0].quarter != 0
            or any(
                b.quarter <= a.quarter for a, b in zip(self.tempo_sections, self.tempo_sections[1:])
            )
        ):
            raise ValueError("Tempo sections must start at zero and increase strictly")
        if self.mode == "beats":
            if len(self.grid_times) != len(self.grid_positions) or len(self.grid_times) < 2:
                raise ValueError("Beat timing requires matching time/position arrays")
            if any(b <= a for a, b in zip(self.grid_times, self.grid_times[1:])) or any(
                b <= a for a, b in zip(self.grid_positions, self.grid_positions[1:])
            ):
                raise ValueError("Beat timing anchors must be strictly increasing")
        elif self.mode != "constant":
            raise ValueError("Timing mode must be constant or beats")
        return self


def grid_quarters(grid: str) -> float:
    fraction = Fraction(grid)
    if fraction not in {Fraction(1, d) for d in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)}:
        raise ValueError("Use a binary or common triplet grid, e.g. 1/16 or 1/12")
    value = fraction * 4
    if value <= 0 or value > 4 or value.denominator > 96:
        raise ValueError(
            "Quantize grid must be positive, at most one whole note, denominator <= 96"
        )
    return float(value)


def analyze(
    path: Path, tempo: float | None, meter: str | None, grid: str, start: float, pickup: float = 0
) -> Rhythm:
    grid_quarters(grid)
    numerator, denominator = (int(s) for s in (meter or "4/4").split("/"))
    if numerator < 1 or denominator not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("Invalid time signature")
    if not 0 <= pickup < numerator * 4 / denominator:
        raise ValueError("Pickup is measured in quarter notes and must be shorter than a bar")
    y, sr = sf.read(path, dtype="float32")
    detected, frames = librosa.beat.beat_track(y=y, sr=sr)
    detected = float(np.asarray(detected).reshape(-1)[0])
    warnings = [
        "Downbeat/meter inference unavailable; using manual meter or provisional 4/4.",
        "Quantization uses a global tempo; rubato needs manual correction.",
    ]
    if detected <= 0 and tempo is None:
        warnings.append("No reliable tempo detected; provisional 120 BPM used.")
    return Rhythm(
        tempo=tempo or (round(detected) if detected > 0 else 120),
        detected_tempo=detected or None,
        beats=(librosa.frames_to_time(frames, sr=sr) + start).tolist(),
        time_signature=meter or "4/4",
        grid=grid,
        pickup=pickup,
        warnings=warnings,
    )


def quantize(start: float, end: float, rhythm: Rhythm) -> tuple[float, float]:
    step = grid_quarters(rhythm.grid)
    onset = max(0, round(seconds_to_quarters(start, rhythm) / step) * step)
    offset = round(seconds_to_quarters(end, rhythm) / step) * step
    return onset, max(onset + step, offset)


def interpolate(value: float, x: list[float], y: list[float]) -> float:
    if value < x[0]:
        index = 0
    elif value > x[-1]:
        index = len(x) - 2
    else:
        return float(np.interp(value, x, y))
    return y[index] + (value - x[index]) * (y[index + 1] - y[index]) / (x[index + 1] - x[index])


def seconds_to_quarters(seconds: float, rhythm: Rhythm) -> float:
    if rhythm.mode == "beats":
        return interpolate(seconds, rhythm.grid_times, rhythm.grid_positions)
    return seconds * rhythm.tempo / 60


def quarters_to_seconds(quarters: float, rhythm: Rhythm) -> float:
    if rhythm.mode == "beats":
        return interpolate(quarters, rhythm.grid_positions, rhythm.grid_times)
    return quarters * 60 / rhythm.tempo


def tempo_map(rhythm: Rhythm) -> list[tuple[float, float]]:
    if rhythm.tempo_sections:
        return [(s.quarter, s.bpm) for s in rhythm.tempo_sections]
    if rhythm.mode != "beats":
        return [(0, rhythm.tempo)]
    result = []
    for i in range(len(rhythm.grid_times) - 1):
        bpm = (
            60
            * (rhythm.grid_positions[i + 1] - rhythm.grid_positions[i])
            / (rhythm.grid_times[i + 1] - rhythm.grid_times[i])
        )
        if not result or abs(bpm - result[-1][1]) > 0.05:
            result.append((rhythm.grid_positions[i], bpm))
    return result


def regularize_tempo(
    rhythm: Rhythm, minimum_bars: int = 2, change_bpm: float = 6, change_fraction: float = 0.06
) -> Rhythm:
    """Use shared integer-BPM sections; retain source beats for note placement.

    Small timing errors affect quantization against the original beat grid, but
    playback runs steadily within each section. Sustained large tempo changes
    remain at measure boundaries. All instruments use this same conductor map.
    """
    from scipy.ndimage import median_filter

    if minimum_bars < 1 or change_bpm <= 0 or change_fraction <= 0:
        raise ValueError("Tempo regularization thresholds must be positive")
    if rhythm.mode != "beats":
        bpm = max(20, min(400, round(rhythm.tempo)))
        return Rhythm.model_validate(
            {**rhythm.model_dump(), "tempo": bpm, "tempo_sections": [dict(quarter=0, bpm=bpm)]}
        )
    numerator, denominator = map(int, rhythm.time_signature.split("/"))
    bar = numerator * 4 / denominator
    # Bar 0 ends at pickup; subsequent boundaries are complete measures.
    boundaries = [0.0]
    first = rhythm.pickup or bar
    boundaries.extend(np.arange(first, rhythm.grid_positions[-1] + 1e-7, bar).tolist())
    if len(boundaries) < 2:
        boundaries.append(float(rhythm.grid_positions[-1]))
    times = np.array([quarters_to_seconds(q, rhythm) for q in boundaries])
    raw = 60 * np.diff(boundaries) / np.diff(times)
    values = median_filter(raw, size=3, mode="nearest")
    n = len(values)
    threshold = max(change_bpm, float(np.median(values)) * change_fraction)
    penalty = 4 * threshold**2
    sums = np.r_[0, np.cumsum(values)]
    squares = np.r_[0, np.cumsum(values**2)]
    costs = np.full(n + 1, np.inf)
    costs[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    for end in range(1, n + 1):
        for begin in range(end):
            if end - begin < minimum_bars and not (begin == 0 and end == n):
                continue
            error = max(
                0, squares[end] - squares[begin] - (sums[end] - sums[begin]) ** 2 / (end - begin)
            )
            candidate = costs[begin] + error + penalty
            if candidate < costs[end]:
                costs[end], previous[end] = candidate, begin
    groups = []
    end = n
    while end:
        begin = int(previous[end])
        groups.insert(0, (begin, end))
        end = begin
    # Remove statistically detectable but musically small changes.
    merged = []
    for begin, end in groups:
        if merged:
            a, b = merged[-1]
            left, right = float(np.median(raw[a:b])), float(np.median(raw[begin:end]))
            if abs(right - left) < max(change_bpm, min(left, right) * change_fraction):
                merged[-1] = (a, end)
                continue
        merged.append((begin, end))
    sections = [
        dict(quarter=float(boundaries[a]), bpm=max(20, min(400, round(float(np.median(raw[a:b]))))))
        for a, b in merged
    ]
    warnings = [w for w in rhythm.warnings if "Steady section tempos" not in w]
    warnings.append(
        "Steady section tempos intentionally smooth small performance timing variations; source beats are retained for note placement."
    )
    return Rhythm.model_validate(
        {
            **rhythm.model_dump(),
            "tempo": sections[0]["bpm"],
            "tempo_sections": sections,
            "warnings": warnings,
        }
    )

from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_map(self):
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
        tempo=tempo or (detected if detected > 0 else 120),
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

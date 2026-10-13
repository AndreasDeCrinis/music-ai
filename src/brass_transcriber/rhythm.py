from fractions import Fraction
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel


class Rhythm(BaseModel):
    tempo: float
    detected_tempo: float | None
    beats: list[float]
    time_signature: str = "4/4"
    grid: str = "1/16"
    pickup: float = 0
    warnings: list[str] = []


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
    onset = round(start * rhythm.tempo / 60 / step) * step
    offset = round(end * rhythm.tempo / 60 / step) * step
    return onset, max(onset + step, offset)

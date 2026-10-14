"""Verified local neural beats with a metrical continuity decoder."""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from .rhythm import Rhythm, analyze, grid_quarters


class BeatAnalysis(BaseModel):
    beats: list[float]
    downbeats: list[float]
    beat_numbers: list[int]
    runtime: float
    device: str = "cpu"
    model: str = "Beat This! final0 + madmom DBN (configured meter)"


def neural_beats(
    audio: Path, checkpoint: Path, meter: str, prior_tempo: float, artifacts: Path
) -> BeatAnalysis:
    import torch
    from beat_this.inference import Audio2Frames
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing neural beat checkpoint: {checkpoint}; run brass-transcribe download-model --variant beats"
        )
    artifacts.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(audio, dtype="float32")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    started = time.perf_counter()
    model = Audio2Frames(checkpoint_path=str(checkpoint), device="cpu", float16=False)
    beat_logits, downbeat_logits = model(y, sr)
    bp = beat_logits.double().sigmoid().cpu().numpy()
    dp = downbeat_logits.double().sigmoid().cpu().numpy()
    np.savez_compressed(artifacts / "beat_activations.npz", beat=bp, downbeat=dp)
    numerator, denominator = map(int, meter.split("/"))
    compound = denominator == 8 and numerator >= 6 and numerator % 3 == 0
    beats_per_bar = numerator // 3 if compound else numerator
    beat_unit = 1.5 if compound else 4 / denominator
    prior = prior_tempo / beat_unit
    decoder = DBNDownBeatTrackingProcessor(
        beats_per_bar=[beats_per_bar],
        fps=50,
        min_bpm=max(45, prior * 0.72),
        max_bpm=min(240, prior * 1.32),
        transition_lambda=100,
        correct=True,
    )
    observations = np.stack([np.maximum(bp - dp, 1e-5), np.maximum(dp, 1e-5)], axis=1)
    # DBN also models non-beat probability; leave a positive remainder when
    # saturated logits or numerical floors would make the two columns sum to 1.
    observations /= np.maximum(1, observations.sum(axis=1, keepdims=True) / 0.999999)
    decoded = decoder(observations)
    if len(decoded) < 8:
        raise ValueError("Too few neural beats for reliable beat-aligned quantization")
    result = BeatAnalysis(
        beats=decoded[:, 0].tolist(),
        downbeats=decoded[decoded[:, 1] == 1, 0].tolist(),
        beat_numbers=decoded[:, 1].astype(int).tolist(),
        runtime=time.perf_counter() - started,
    )
    (artifacts / "beat_predictions.json").write_text(result.model_dump_json(indent=2))
    return result


def beat_rhythm(
    prediction: BeatAnalysis,
    meter: str = "4/4",
    grid: str = "1/16",
    start: float = 0,
    pickup: float = 0,
) -> Rhythm:
    grid_quarters(grid)
    numerator, denominator = map(int, meter.split("/"))
    compound = denominator == 8 and numerator >= 6 and numerator % 3 == 0
    unit = 1.5 if compound else 4 / denominator
    bar_length = numerator * 4 / denominator
    beats = np.asarray(prediction.beats)
    period = float(np.median(np.diff(beats)))
    bpm = 60 * unit / period
    # Use a strong first downbeat to establish bar phase. Weak opening beat
    # detections are not allowed to compress the first beat into an extreme tempo.
    earliest_anchor = max(0.5 * period, 0.5 * bar_length / unit * period - start)
    first_downbeat = next((d for d in prediction.downbeats if d >= earliest_anchor), None)
    if first_downbeat is None:
        raise ValueError("No suitable downbeat anchor detected")
    down_index = int(np.argmin(abs(beats - first_downbeat)))
    down_absolute = first_downbeat + start
    down_quarter = max(bar_length, round(down_absolute / period * unit / bar_length) * bar_length)
    times = [0.0, float(down_absolute)]
    quarters = [0.0, float(down_quarter)]
    for index in range(down_index + 1, len(beats)):
        times.append(float(beats[index] + start))
        quarters.append(float(down_quarter + (index - down_index) * unit))
    return Rhythm(
        tempo=bpm,
        detected_tempo=bpm,
        beats=(beats + start).tolist(),
        downbeats=[d + start for d in prediction.downbeats],
        time_signature=meter,
        grid=grid,
        pickup=pickup,
        mode="beats",
        grid_times=times,
        grid_positions=quarters,
        tracker=prediction.model,
        tracker_device=prediction.device,
        tracker_runtime=prediction.runtime,
        warnings=[
            "Beat/downbeat positions are model estimates; meter is configured, not proven.",
            "Opening lead-in uses a linear map to the first strong downbeat; verify pickup/first bar.",
        ],
    )


def analyze_quality(
    audio: Path,
    cache: Path,
    artifacts: Path,
    meter: str = "4/4",
    grid: str = "1/16",
    start: float = 0,
    pickup: float = 0,
    tempo_override: float | None = None,
) -> Rhythm:
    baseline = analyze(audio, tempo_override, meter, grid, start, pickup)
    if tempo_override is not None:
        baseline.warnings.append("Manual tempo override uses constant-tempo quantization.")
        return baseline
    prediction = neural_beats(
        audio, cache / "beat_this/final0.ckpt", meter, baseline.tempo, artifacts
    )
    return beat_rhythm(prediction, meter, grid, start, pickup)

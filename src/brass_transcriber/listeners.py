"""Second listener: Basic Pitch on audio with drums and bass removed by Hybrid Demucs.

YourMT3 hears dense band mixes as a whole and misses quiet upper lines (for example
clarinets above a brass tutti). Removing the rhythm section and listening with an
instrument-agnostic model recovers part of that register. Notes remain hypotheses;
consensus decides which are kept.
"""

import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .backend import resolve_device
from .models import NoteEvent, TranscriptionResult

log = logging.getLogger(__name__)
SEPARATION_WEIGHTS = Path("torch/torchaudio/models/hdemucs_high_trained.pt")
SEPARATION_URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"
SEPARATION_SHA256 = "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
# Upper register above typical brass melody lines; lower notes need YourMT3 agreement.
MIN_PITCH, MAX_PITCH = 77, 96
MIN_DURATION = 0.15


def remove_rhythm_section(
    audio: Path,
    output: Path,
    cache: Path,
    segment: float = 10,
    overlap: float = 1,
    device: str = "auto",
) -> Path:
    """Write 16 kHz mono audio without the separated drums and bass stems."""
    import librosa
    import torch
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

    if not (cache / SEPARATION_WEIGHTS).is_file():
        raise FileNotFoundError(
            "Missing separation weights; run brass-transcribe download-model --variant separation"
        )
    torch.hub.set_dir(str((cache / "torch").resolve()))
    device = resolve_device(device)
    model = HDEMUCS_HIGH_MUSDB_PLUS.get_model().eval().to(device)
    rate = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate
    y, sr = sf.read(audio, dtype="float32")
    mix = librosa.resample(y, orig_sr=sr, target_sr=rate)
    mean, std = float(mix.mean()), float(mix.std()) + 1e-8
    size, hop = int(segment * rate), int((segment - overlap) * rate)
    fade = int(overlap * rate)
    kept = np.zeros_like(mix)
    weight = np.zeros_like(mix)
    keep = [model.sources.index(s) for s in ("other", "vocals")]
    with torch.inference_mode():
        for begin in range(0, len(mix), hop):
            chunk = mix[begin : begin + size]
            x = torch.from_numpy((chunk - mean) / std)[None, None].repeat(1, 2, 1).to(device)
            stems = model(x)[0].cpu().numpy() * std + mean
            part = stems[keep].sum(0).mean(0)
            window = np.ones(len(chunk), dtype="float32")
            if begin:
                window[:fade] = np.linspace(0, 1, min(fade, len(chunk)), dtype="float32")
            if begin + size < len(mix):
                window[-fade:] = np.linspace(1, 0, fade, dtype="float32")
            kept[begin : begin + len(chunk)] += part * window
            weight[begin : begin + len(chunk)] += window
            if begin + size >= len(mix):
                break
    kept /= np.maximum(weight, 1e-6)
    result = librosa.resample(kept, orig_sr=rate, target_sr=sr).astype("float32")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result, sr, subtype="FLOAT")
    return output


def octave_shadow(note: tuple, events: list[tuple]) -> bool:
    """A weak note overlapping a much louder note one octave below is likely an overtone."""
    start, end, pitch, amplitude = note[:4]
    return any(
        other[2] == pitch - 12
        and min(end, other[1]) - max(start, other[0]) > 0.5 * (end - start)
        and other[3] > 1.5 * amplitude
        for other in events
    )


def basic_pitch_notes(audio: Path, start: float = 0, pass_id: str = "bp_") -> TranscriptionResult:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    started = time.perf_counter()
    _, _, events = predict(
        str(audio), ICASSP_2022_MODEL_PATH, onset_threshold=0.4, frame_threshold=0.2
    )
    notes = [
        NoteEvent(
            id=f"{pass_id}{i}",
            pitch=int(e[2]),
            start=float(e[0]) + start,
            end=float(e[1]) + start,
            velocity=int(np.clip(round(float(e[3]) * 127), 1, 127)),
            program=0,
            provenance=[f"{pass_id}{i}"],
        )
        for i, e in enumerate(events)
        if MIN_PITCH <= e[2] <= MAX_PITCH
        and e[1] - e[0] >= MIN_DURATION
        and not octave_shadow(e, events)
    ]
    return TranscriptionResult(
        notes=notes,
        model="Basic Pitch (ICASSP 2022, ONNX) on Hybrid Demucs drums/bass-removed audio",
        device="cpu",
        runtime=time.perf_counter() - started,
        warnings=[
            f"Upper-register listener only (MIDI {MIN_PITCH}-{MAX_PITCH}); no instrument labels.",
            "Octave-overtone filtering is heuristic.",
        ],
    )

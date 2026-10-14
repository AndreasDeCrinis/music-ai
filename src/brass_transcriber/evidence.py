"""Inspectable harmonic-template audio evidence, independent of model program labels.

These scores are relative salience heuristics, not calibrated note probabilities.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from .models import NoteEvent


@dataclass
class PitchEvidence:
    activations: np.ndarray
    times: np.ndarray
    pitches: np.ndarray
    start: float = 0

    def support(self, note: NoteEvent) -> float:
        indices = np.where(self.pitches == note.pitch)[0]
        if not len(indices) or note.drum:
            return 0
        left = max(0, int(np.searchsorted(self.times, note.start - self.start)))
        right = min(
            len(self.times), max(left + 1, int(np.searchsorted(self.times, note.end - self.start)))
        )
        if left >= right:
            return 0
        relative = self.activations[indices[0], left:right] / (
            np.max(self.activations[:, left:right], axis=0) + 1e-8
        )
        return float(np.clip(np.median(relative), 0, 1))

    def onset_novelty(self, note: NoteEvent) -> float:
        index = np.where(self.pitches == note.pitch)[0]
        if not len(index):
            return 0
        time = note.start - self.start
        before = (self.times >= time - 0.1) & (self.times < time - 0.02)
        after = (self.times >= time) & (self.times < time + 0.08)
        if not before.any() or not after.any():
            return 1
        row = self.activations[index[0]]
        a, b = float(np.mean(row[before])), float(np.mean(row[after]))
        return float(max(0, (b - a) / (max(a, b) + 1e-8)))


def analyze_evidence(audio: Path, artifact: Path, start: float = 0) -> PitchEvidence:
    y, sr = sf.read(audio, dtype="float32")
    n_fft, hop = 4096, 160
    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).astype("float32")
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    keep = (frequencies >= 25) & (frequencies <= 6500)
    frequencies = frequencies[keep]
    spectrum = magnitude[keep]
    # Whitening reduces the domination of low-frequency instrument/percussion energy.
    whitening = (
        1
        / np.maximum(np.mean(spectrum, axis=1), np.percentile(np.mean(spectrum, axis=1), 20)) ** 0.5
    )
    spectrum *= whitening[:, None]
    pitches = np.arange(20, 101)
    templates = np.zeros((len(frequencies), len(pitches)), dtype="float32")
    for column, pitch in enumerate(pitches):
        fundamental = librosa.midi_to_hz(pitch)
        for harmonic in range(1, 13):
            center = fundamental * harmonic
            if center > 6500:
                break
            width = max(sr / n_fft, center * 0.006)
            templates[:, column] += (
                np.exp(-0.5 * ((frequencies - center) / width) ** 2) / harmonic**0.8
            )
    templates *= whitening[:, None]
    templates /= np.linalg.norm(templates, axis=0, keepdims=True) + 1e-8
    with threadpool_limits(limits=8):
        gram = templates.T @ templates
        projection = templates.T @ spectrum
        activations = np.maximum(projection, 1e-8)
        for _ in range(45):
            activations *= projection / (gram @ activations + 1e-4)
    times = librosa.frames_to_time(np.arange(activations.shape[1]), sr=sr, hop_length=hop)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, pitches=pitches, times=times, activations=activations)
    return PitchEvidence(activations, times, pitches, start)

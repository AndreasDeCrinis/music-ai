"""Measure pass combinations of a finished quality run against a reference score.

The reference is a MIDI export of a score whose audio was transcribed (for example
a MuseScore rendering). Selection and part assignment are replayed from the cached
passes, so no inference runs here. Notes count as correct with matching pitch and
an onset within 50 ms; same-pitch notes with onsets within 80 ms count once on both
sides, so unison doublings neither reward nor penalize a configuration.
"""

import json
from pathlib import Path

import numpy as np
import pretty_midi

from .consensus import build_consensus
from .evidence import PitchEvidence
from .models import EnsembleConfig, TranscriptionResult
from .quality import PASS_SPECS, assign_selected
from .rhythm import Rhythm

ONSET_TOLERANCE = 0.05
UNISON_WINDOW = 0.08


def reference_notes(path: Path) -> list[tuple[float, float, int]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        (n.start, n.end, n.pitch)
        for track in midi.instruments
        if not track.is_drum
        for n in track.notes
        if n.end > n.start
    ]


def merge_unisons(notes: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    merged: list[tuple[float, float, int]] = []
    last: dict[int, int] = {}
    for start, end, pitch in sorted(notes):
        index = last.get(pitch)
        if index is not None and start - merged[index][0] < UNISON_WINDOW:
            merged[index] = (merged[index][0], max(merged[index][1], end), pitch)
            continue
        last[pitch] = len(merged)
        merged.append((start, end, pitch))
    return merged


def note_scores(reference, estimate, offset: float = 0) -> tuple[float, float, float]:
    import mir_eval

    def arrays(notes):
        if not notes:
            return np.zeros((0, 2)), np.zeros(0)
        intervals = np.array([[s, max(e, s + 0.01)] for s, e, _ in notes])
        return intervals, np.array([440 * 2 ** ((p - 69) / 12) for _, _, p in notes])

    ref_intervals, ref_pitches = arrays(reference)
    est_intervals, est_pitches = arrays([(s - offset, e - offset, p) for s, e, p in estimate])
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=ONSET_TOLERANCE,
        offset_ratio=None,
    )
    return precision, recall, f1


def estimate_offset(reference, estimate) -> float:
    """Global latency between reference and transcription (rendering lead-in)."""
    grid = np.arange(-0.3, 0.301, 0.01)
    scores = [note_scores(reference, estimate, o)[2] for o in grid]
    return float(grid[int(np.argmax(scores))])


def load_run(run: Path):
    intermediate = run / "intermediate"
    settings = json.loads((intermediate / "run_settings.json").read_text())
    data = np.load(intermediate / "pitch_evidence.npz")
    evidence = PitchEvidence(
        data["activations"], data["times"], data["pitches"], settings["excerpt_start"]
    )
    results = {
        v.name: TranscriptionResult.model_validate_json((v / "model_notes.json").read_text())
        for v in sorted((intermediate / "variants").iterdir())
        if (v / "model_notes.json").exists()
    }
    parts = EnsembleConfig.model_validate(
        {**(settings.get("ensemble") or {}), "parts": settings["parts"]}
    ).parts
    return results, evidence, parts, Rhythm.model_validate(settings["rhythm"])


def replay(results, evidence, parts, rhythm, min_agreement: int, beam_size: int):
    trusted = frozenset({"separated"} & set(results))
    _, selected, decisions = build_consensus(
        results, "ps", evidence, trusted, min_agreement=min_agreement
    )
    assignments, _ = assign_selected(selected, decisions, parts, beam_size, rhythm)
    return [
        (a.note.start, a.note.end, a.note.pitch) for a in assignments if a.part and not a.note.drum
    ]


def configurations(available: list[str]) -> list[tuple[str, list[str], int]]:
    """Nested pass sets in PASS_SPECS order with agreement levels, then leave-one-out."""
    order = ["baseline"] + [spec[0] for spec in PASS_SPECS[1:]]
    extras = [p for p in order if p in available]
    configs = []
    for count in range(len(extras) + 1):
        chosen = extras[:count]
        for agreement in range(2, max(2, min(4, len(chosen) + 1)) + 1):
            configs.append((f"{count + 1} passes", chosen, agreement))
    if len(extras) > 2:
        for left_out in extras:
            configs.append((f"all without {left_out}", [p for p in extras if p != left_out], 2))
    return configs


def benchmark_run(run: Path, reference: Path, beam_size: int = 64) -> dict:
    results, evidence, parts, rhythm = load_run(run)
    if "ps" not in results:
        raise ValueError("Benchmark needs a quality run with the primary ps pass")
    ref = merge_unisons(reference_notes(reference))
    full = replay(results, evidence, parts, rhythm, 2, beam_size)
    offset = estimate_offset(ref, merge_unisons(full))
    rows = []
    for label, chosen, agreement in configurations(list(results)):
        subset = {k: results[k] for k in ["ps", *chosen, "separated"] if k in results}
        notes = replay(subset, evidence, parts, rhythm, agreement, beam_size)
        unique = merge_unisons(notes)
        precision, recall, f1 = note_scores(ref, unique, offset)
        rows.append(
            dict(
                label=label,
                passes=["ps", *chosen],
                agreement=agreement,
                notes=len(notes),
                duplicates=len(notes) - len(unique),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
            )
        )
    return dict(reference=str(reference), reference_notes=len(ref), offset=offset, rows=rows)


def format_table(report: dict) -> str:
    lines = [
        f"reference notes (unisons merged): {report['reference_notes']}   offset {report['offset']:+.2f}s",
        f"{'configuration':28s} {'agree':>5s} {'notes':>6s} {'dupes':>5s} {'prec':>6s} {'recall':>6s} {'F1':>6s}",
    ]
    for r in report["rows"]:
        lines.append(
            f"{r['label']:28s} {r['agreement']:5d} {r['notes']:6d} {r['duplicates']:5d} "
            f"{r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}"
        )
    best = max(report["rows"], key=lambda r: r["f1"])
    lines.append(
        f"best F1: {best['label']} (agreement {best['agreement']}): {' + '.join(best['passes'])}"
    )
    return "\n".join(lines)

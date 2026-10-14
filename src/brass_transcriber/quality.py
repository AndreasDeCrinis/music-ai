"""Slow, local, auditable precision pipeline. Raw predictions are never overwritten."""

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .backend import YourMT3Backend, prepare_ps_checkpoint
from .consensus import build_consensus
from .evidence import analyze_evidence
from .listeners import SEPARATION_WEIGHTS, basic_pitch_notes, remove_rhythm_section
from .models import (
    Assignment,
    EnsembleConfig,
    EnsemblePreset,
    TranscriptionResult,
    load_ensemble,
    write_midi,
)
from .phrases import add_phrases, assign_phrases, place_unassigned
from .rhythm import Rhythm, quantize, quarters_to_seconds
from .rhythm_quality import analyze_quality

log = logging.getLogger(__name__)


@dataclass
class QualityRun:
    result: TranscriptionResult
    assignments: list[Assignment]
    rhythm: Rhythm
    ensemble: EnsembleConfig
    duration: float
    start: float
    summary: dict


def improve_run(
    source: Path,
    output: Path,
    cache: Path,
    preset: EnsemblePreset | None = None,
    passes: int = 4,
    beam_size: int = 64,
    tempo_override: float | None = None,
    listener: bool = True,
    device: str = "auto",
) -> QualityRun:
    if source.resolve() == output.resolve():
        raise ValueError(
            "Improvement requires a separate --output directory to preserve the baseline"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Improved output directory is not empty")
    if passes not in {2, 3, 4}:
        raise ValueError("Use 2, 3 or 4 passes (cached baseline + PS + optional pitch-shift views)")
    source_intermediate = source / "intermediate"
    settings = json.loads((source_intermediate / "run_settings.json").read_text())
    baseline_path = source_intermediate / "model_notes.json"
    if (source_intermediate / "variants/baseline/model_notes.json").exists():
        baseline_path = source_intermediate / "variants/baseline/model_notes.json"
    baseline = TranscriptionResult.model_validate_json(baseline_path.read_text())
    if preset is not None:
        ensemble = load_ensemble(preset=preset)
    else:
        ensemble = EnsembleConfig.model_validate(
            {**(settings.get("ensemble") or {}), "parts": settings["parts"]}
        )
    duration, start = settings["audio_duration"], settings["excerpt_start"]
    original_rhythm = Rhythm.model_validate(settings["rhythm"])
    # Fail before creating a partial run when required model weights are unavailable.
    prepare_ps_checkpoint(cache)
    if not (cache / "beat_this/final0.ckpt").exists():
        raise FileNotFoundError("Run brass-transcribe download-model --variant beats first")
    if listener and not (cache / SEPARATION_WEIGHTS).exists():
        raise FileNotFoundError(
            "Run brass-transcribe download-model --variant separation first, or use --no-listener"
        )
    intermediate = output / "intermediate"
    intermediate.mkdir(parents=True)
    (output / "diagnostics").mkdir()
    shutil.copytree(source_intermediate, intermediate / "baseline")
    audio = intermediate / "normalized.wav"
    shutil.copy2(source_intermediate / "normalized.wav", audio)
    variants = intermediate / "variants"
    variants.mkdir()
    cached = variants / "baseline"
    cached.mkdir()
    shutil.copy2(baseline_path, cached / "model_notes.json")
    started = time.perf_counter()
    log.info("Quality: neural beat/downbeat tracking with metrical continuity")
    rhythm = analyze_quality(
        audio,
        cache,
        intermediate / "timing",
        original_rhythm.time_signature,
        original_rhythm.grid,
        start,
        original_rhythm.pickup,
        tempo_override,
        device,
    )
    rhythm = Rhythm.model_validate({**rhythm.model_dump(), "notation_mode": "phrases"})
    log.info("Quality: harmonic-template audio evidence")
    evidence = analyze_evidence(audio, intermediate / "pitch_evidence.npz", start)
    results = {"baseline": baseline}
    y, sr = sf.read(audio, dtype="float32")
    specifications = [("ps", 0, 45, 5), ("ps_up", 1, 33, 3), ("ps_down", -1, 39, 4)]
    for pass_id, shift, chunk, overlap in specifications[: passes - 1]:
        directory = variants / pass_id
        directory.mkdir()
        log.info("Quality: transcription pass %s (pitch shift %+d)", pass_id, shift)
        transformed = audio
        if shift:
            transformed = directory / "input.wav"
            sf.write(
                transformed,
                librosa.effects.pitch_shift(y, sr=sr, n_steps=shift).astype("float32"),
                sr,
                subtype="FLOAT",
            )
        backend = YourMT3Backend(
            cache,
            directory / "model_chunks",
            start,
            chunk,
            overlap,
            device=device,
            variant="ps",
            pass_id=pass_id + "_",
        )
        prediction = backend.transcribe(transformed)
        # Untouched raw MIDI remains in model_chunks. Normalized JSON records the
        # inverse augmentation so concert pitches can be compared between passes.
        if shift:
            prediction.notes = [
                n.model_copy(update={"pitch": n.pitch - shift}) if not n.drum else n
                for n in prediction.notes
                if n.drum or 0 <= n.pitch - shift <= 127
            ]
        (directory / "model_notes.json").write_text(prediction.model_dump_json(indent=2))
        (directory / "pass_settings.json").write_text(
            json.dumps(
                dict(pitch_shift=shift, chunk_seconds=chunk, overlap_seconds=overlap), indent=2
            )
        )
        results[pass_id] = prediction
    trusted = frozenset()
    if listener:
        log.info("Quality: upper-register listener on drums/bass-removed audio")
        directory = variants / "separated"
        directory.mkdir()
        separated = remove_rhythm_section(audio, directory / "input.wav", cache, device=device)
        prediction = basic_pitch_notes(separated, start, "separated_")
        (directory / "model_notes.json").write_text(prediction.model_dump_json(indent=2))
        results["separated"] = prediction
        trusted = frozenset({"separated"})
    log.info("Quality: combine hypotheses, retaining disagreement and provenance")
    result, selected, decisions = build_consensus(results, "ps", evidence, trusted)
    (intermediate / "quality_decisions.json").write_text(json.dumps(decisions, indent=2))
    (intermediate / "model_notes.json").write_text(result.model_dump_json(indent=2))
    (intermediate / "quality_notes.json").write_text(
        json.dumps([n.model_dump() for n in selected], indent=2)
    )
    write_midi(result.notes, intermediate / "raw_transcription.mid", rhythm.tempo)
    write_midi(selected, intermediate / "selected_transcription.mid", rhythm.tempo)
    log.info("Quality: track melodic voices and assign complete phrases")
    dependent = {d["id"] for d in decisions if d["listener_dependent"]}
    assignments, phrases = assign_phrases(
        [n for n in selected if n.id not in dependent], ensemble.parts, beam_size, rhythm=rhythm
    )
    assignments = place_unassigned(assignments, ensemble.parts)
    assignments = add_phrases(
        assignments, [n for n in selected if n.id in dependent], ensemble.parts
    )
    (intermediate / "phrases.json").write_text(json.dumps(phrases, indent=2))
    selected_ids = {n.id for n in selected}
    rejected = {d["id"]: d for d in decisions if d["action"] == "reject"}
    assignments.extend(
        Assignment(
            note=n, part=None, confidence=0, reason="quality rejection: " + rejected[n.id]["reason"]
        )
        for n in result.notes
        if n.id not in selected_ids
    )
    (intermediate / "assigned_voices.json").write_text(
        json.dumps([a.model_dump() for a in assignments], indent=2)
    )
    summary = dict(
        pipeline="local precision v1",
        passes={
            k: dict(
                model=r.model,
                notes=len(r.notes),
                inference_runtime=r.runtime,
                cached=k == "baseline",
            )
            for k, r in results.items()
        },
        primary="ps",
        hypotheses=len(result.notes),
        selected_notes=len(selected),
        rejected_notes=len(rejected),
        phrase_count=phrases["phrase_count"],
        mean_phrase_notes=phrases["mean_phrase_notes"],
        new_runtime_seconds=time.perf_counter() - started,
        raw_transcription_kind="unfiltered multi-pass hypotheses; original MIDI retained per variant",
        limitations=[
            "No hand-annotated reference score: improved structural consistency is not measured ground-truth precision.",
            "Pitch-shift views can omit valid notes; unshifted PS predictions are retained as the primary anchor.",
            "Harmonic salience and pass agreement are heuristic evidence, not calibrated probabilities.",
        ],
    )
    summary["comparison"] = compare_quality(source, result, assignments, rhythm, decisions)
    (output / "diagnostics/quality_report.json").write_text(json.dumps(summary, indent=2))
    write_review_plot(source, assignments, rhythm, output / "diagnostics/opening_comparison.png")
    return QualityRun(result, assignments, rhythm, ensemble, duration, start, summary)


def fragmentation(assignments: list[Assignment]) -> dict:
    isolated = 0
    fragments = 0
    parts = {a.part for a in assignments if a.part and not a.note.drum}
    for part in parts:
        notes = sorted((a.note for a in assignments if a.part == part), key=lambda n: n.start)
        runs = []
        for n in notes:
            if not runs or n.start - max(x.end for x in runs[-1]) > 1:
                runs.append([n])
            else:
                runs[-1].append(n)
        fragments += len(runs)
        isolated += sum(len(r) == 1 for r in runs)
    return dict(phrases_separated_by_one_second=fragments, isolated_single_note_fragments=isolated)


def compare_quality(
    source: Path,
    result: TranscriptionResult,
    assignments: list[Assignment],
    rhythm: Rhythm,
    decisions: list[dict],
) -> dict:
    old_settings = json.loads((source / "intermediate/run_settings.json").read_text())
    old_rhythm = Rhythm.model_validate(old_settings["rhythm"])
    old_assignments = [
        Assignment.model_validate(a)
        for a in json.loads((source / "intermediate/assigned_voices.json").read_text())
    ]
    before_errors = []
    after_errors = []
    for a in assignments:
        if not a.part:
            continue
        onset, _ = quantize(a.note.start, a.note.end, rhythm)
        after_errors.append(abs(quarters_to_seconds(onset, rhythm) - a.note.start))
    for a in old_assignments:
        if not a.part:
            continue
        onset, _ = quantize(a.note.start, a.note.end, old_rhythm)
        before_errors.append(abs(quarters_to_seconds(onset, old_rhythm) - a.note.start))
    comparison = dict(
        baseline=fragmentation(old_assignments),
        improved=fragmentation(assignments),
        baseline_onset_rounding_median_seconds=float(np.median(before_errors))
        if before_errors
        else 0,
        improved_onset_rounding_median_seconds=float(np.median(after_errors))
        if after_errors
        else 0,
        note_event_accuracy="not measured without annotated reference",
    )
    if rhythm.mode == "beats":
        deviations = [
            abs(quarters_to_seconds(q, old_rhythm) - t)
            for q, t in zip(rhythm.grid_positions, rhythm.grid_times, strict=True)
        ]
        comparison["baseline_deviation_from_neural_beat_anchors_max_seconds"] = max(deviations)
        comparison["improved_deviation_from_neural_beat_anchors_max_seconds"] = 0
    # Preserve a concrete audit of the passage the user identified, using the
    # baseline's second-measure audio window rather than silently renumbering it.
    numerator, denominator = map(int, old_rhythm.time_signature.split("/"))
    bar = numerator * 4 / denominator
    window = (quarters_to_seconds(bar, old_rhythm), quarters_to_seconds(2 * bar, old_rhythm))
    comparison["baritone_second_measure_review"] = dict(
        audio_window=list(window),
        baseline=[
            a.model_dump()
            for a in old_assignments
            if a.part == "baritone" and window[0] <= a.note.start < window[1]
        ],
        improved=[
            a.model_dump()
            for a in assignments
            if a.part == "baritone" and window[0] <= a.note.start < window[1]
        ],
        related_hypotheses=[
            d for d in decisions if not d["percussion"] and window[0] <= d["start"] < window[1]
        ],
    )
    return comparison


def write_review_plot(source: Path, assignments: list[Assignment], rhythm: Rhythm, path: Path):
    """A short, readable comparison around the user-reported baritone passage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    old = [
        Assignment.model_validate(a)
        for a in json.loads((source / "intermediate/assigned_voices.json").read_text())
    ]
    settings = json.loads((source / "intermediate/run_settings.json").read_text())
    baseline_rhythm = Rhythm.model_validate(settings["rhythm"])
    numerator, denominator = map(int, baseline_rhythm.time_signature.split("/"))
    bar = numerator * 4 / denominator
    left, right = (quarters_to_seconds(q, baseline_rhythm) for q in [bar, 2 * bar])
    end = max(8, right + 3)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
    colors = {"baritone": "#c62828", "other": "#246da8", "unassigned": "#aaa"}
    for ax, items, timing, title in zip(
        axes,
        [old, assignments],
        [baseline_rhythm, rhythm],
        ["Original part assignment", "Multi-pass phrase assignment"],
        strict=True,
    ):
        ax.axvspan(left, right, color="#f3d879", alpha=0.25)
        for a in items:
            n = a.note
            if n.drum or n.start >= end or n.end < 0:
                continue
            category = "baritone" if a.part == "baritone" else "other" if a.part else "unassigned"
            onset, offset = quantize(n.start, n.end, timing)
            t0, t1 = (quarters_to_seconds(q, timing) for q in (onset, offset))
            ax.broken_barh(
                [(t0, max(0.01, t1 - t0))],
                (n.pitch - 0.35, 0.7),
                facecolors=colors[category],
                alpha=0.9 if a.part else 0.4,
            )
            if category == "baritone":
                ax.text(t0, n.pitch + 0.6, f"{n.pitch}", fontsize=8, color=colors[category])
        ax.set(title=title, ylabel="Concert MIDI pitch", ylim=(40, 85))
        ax.grid(alpha=0.15)
    axes[1].set(xlim=(0, end), xlabel="Original recording time (seconds)")
    axes[0].legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="upper right")
    fig.suptitle(
        "Opening passage — shaded region is the original second bar\n"
        "Changed assignments are suggestions; correct notes need a reference",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import Assignment, NoteEvent, PartConfig, TranscriptionResult
from .rhythm import Rhythm


def report(
    result: TranscriptionResult,
    assignments: list[Assignment],
    parts: list[PartConfig],
    rhythm: Rhythm,
    duration: float,
    start: float,
    output: Path,
    warnings: list[str],
) -> dict:
    tracks = defaultdict(list)
    for n in result.notes:
        tracks[(n.track, n.program, n.drum)].append(n.pitch)
    counts = Counter(a.part for a in assignments)
    collisions = []
    for p in parts:
        events = sorted((a.note for a in assignments if a.part == p.id), key=lambda n: n.start)
        active = []
        for n in events:
            active = [prev for prev in active if prev.end > n.start + 0.01]
            if len(active) >= p.max_simultaneous:
                collisions.append(n.id)
            active.append(n)
    data = dict(
        audio_duration=duration,
        excerpt_start=start,
        selected_model=result.model,
        inference_device=result.device,
        inference_runtime=result.runtime,
        rhythm=rhythm.model_dump(),
        extracted_notes=len(result.notes),
        original_tracks=[
            dict(
                track=k[0],
                program=k[1],
                percussion=k[2],
                pitch_range=[min(v), max(v)],
                note_count=len(v),
            )
            for k, v in tracks.items()
        ],
        part_note_counts={p.id: counts[p.id] for p in parts},
        unassigned_notes=[a.note.id for a in assignments if a.part is None]
        if assignments
        else [n.id for n in result.notes],
        potential_collisions=collisions,
        low_confidence_assignments=[
            a.note.id for a in assignments if a.part and a.confidence < 0.6
        ],
        warnings=result.warnings
        + rhythm.warnings
        + warnings
        + [f"{p.name}: no supported notes assigned" for p in parts if not counts[p.id]],
        assignment_status="All part assignments are provisional; confidence is heuristic.",
    )
    settings_path = output / "intermediate/run_settings.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        data["ensemble"] = settings.get("ensemble")
        if settings.get("quality"):
            data["quality"] = settings["quality"]
        if settings.get("refinement"):
            data["refinement"] = settings["refinement"]
    data["percussion_notes"] = sum(n.drum for n in result.notes)
    data["assigned_percussion_notes"] = sum(a.note.drum and a.part is not None for a in assignments)
    (output / "diagnostics/transcription_report.json").write_text(json.dumps(data, indent=2))
    return data


def piano_roll(notes: list[NoteEvent], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for n in notes:
        ax.broken_barh(
            [(n.start, n.end - n.start)],
            (n.pitch - 0.4, 0.8),
            facecolors="gray" if n.drum else plt.cm.tab20(n.track % 20),
        )
    ax.set(
        xlabel="Original recording time (seconds)",
        ylabel="Sounding MIDI pitch",
        title="Unquantized model output",
    )
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

"""Refine cached predictions without loading transcription models."""

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import Assignment, NoteEvent, PartConfig, TranscriptionResult
from .phrases import assign_phrases
from .rhythm import Rhythm, seconds_to_quarters


@dataclass
class RefinementRun:
    result: TranscriptionResult
    assignments: list[Assignment]
    parts: list[PartConfig]
    rhythm: Rhythm
    duration: float
    start: float
    source: Path
    phrases: dict
    before: list[Assignment]


def refine_cached(source: Path, output: Path, beam_size: int = 64) -> RefinementRun:
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a separate output directory outside the source run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refined output directory is not empty")
    inter = source / "intermediate"
    settings = json.loads((inter / "run_settings.json").read_text())
    result = TranscriptionResult.model_validate_json((inter / "model_notes.json").read_text())
    before = [
        Assignment.model_validate(a)
        for a in json.loads((inter / "assigned_voices.json").read_text())
    ]
    selection = inter / "quality_notes.json"
    notes = (
        [NoteEvent.model_validate(n) for n in json.loads(selection.read_text())]
        if selection.exists()
        else result.notes
    )
    model_by_id = {n.id: n for n in result.notes}
    if any(n.id not in model_by_id or n != model_by_id[n.id] for n in notes):
        raise ValueError("Selected notes differ from cached model predictions")
    parts = [PartConfig.model_validate(p) for p in settings["parts"]]
    rhythm = Rhythm.model_validate({**settings["rhythm"], "notation_mode": "phrases"})
    assignments, phrases = assign_phrases(
        notes,
        parts,
        beam_size,
        rhythm=rhythm,
        persistent=True,
        prior_parts={a.note.id: a.part for a in before if a.part},
    )
    selected = {a.note.id for a in assignments}
    old_by_id = {a.note.id: a for a in before}
    assignments.extend(
        Assignment(
            note=n,
            part=None,
            confidence=0,
            reason=old_by_id[n.id].reason
            if n.id in old_by_id and old_by_id[n.id].part is None
            else "cached quality rejection preserved",
        )
        for n in result.notes
        if n.id not in selected
    )
    shutil.copytree(inter, output / "intermediate", dirs_exist_ok=True)
    (output / "diagnostics").mkdir(parents=True, exist_ok=True)
    settings["rhythm"] = rhythm.model_dump()
    settings["refinement"] = dict(
        pipeline="persistent voices and joint rhythm v1",
        source=str(source.resolve()),
        beam_size=beam_size,
        raw_predictions_unchanged=True,
        selection_policy="Refine previously scored notes; retain unassigned candidates for review",
    )
    (output / "intermediate/run_settings.json").write_text(json.dumps(settings, indent=2))
    (output / "intermediate/assigned_voices.json").write_text(
        json.dumps([a.model_dump() for a in assignments], indent=2)
    )
    (output / "intermediate/phrases.json").write_text(json.dumps(phrases, indent=2))
    return RefinementRun(
        result,
        assignments,
        parts,
        rhythm,
        settings["audio_duration"],
        settings["excerpt_start"],
        source,
        phrases,
        before,
    )


def linked_switches(phrases, assignments):
    parts = {a.note.id: a.part for a in assignments}
    choices = {}
    for p in phrases["phrases"]:
        votes = Counter(parts.get(id) for id in p["source_ids"] if parts.get(id))
        choices[p["id"]] = votes.most_common(1)[0][0] if votes else None
    return sum(
        bool(
            choices[p["id"]]
            and choices.get(p["predecessor"])
            and choices[p["id"]] != choices[p["predecessor"]]
        )
        for p in phrases["phrases"]
        if p.get("predecessor")
    )


def notation_metrics(mapping, assignments, rhythm):
    pitched = [p for p in mapping if not p.get("percussion")]
    raw = {a.note.id: a.note for a in assignments}
    by_voice = defaultdict(list)
    for p in pitched:
        by_voice[p["part"], p["notation_voice"]].append(p)
    small_gaps = 0
    for notes in by_voice.values():
        notes.sort(key=lambda p: p["onset_quarters"])
        for a, b in zip(notes, notes[1:]):
            gap = b["onset_quarters"] - a["onset_quarters"] - a["duration_quarters"]
            raw_gap = seconds_to_quarters(raw[b["source_id"]].start, rhythm) - seconds_to_quarters(
                raw[a["source_id"]].end, rhythm
            )
            if 1e-7 < gap <= 0.25 + 1e-7 and 0 <= raw_gap < 0.10:
                small_gaps += 1
    return dict(
        pitched_score_events=len(pitched),
        extra_voice_events=sum(p["notation_voice"] > 1 for p in pitched),
        short_gaps_near_legato_boundaries=small_gaps,
        repeated_motif_supported_events=sum(p.get("motif_support", 0) >= 2 for p in pitched),
        bounded_rhythm_fallbacks=sum(p.get("rhythm_fitting_fallback", False) for p in pitched),
    )


def write_refinement_report(run, output):
    old = json.loads((run.source / "intermediate/score_note_provenance.json").read_text())
    new = json.loads((output / "intermediate/score_note_provenance.json").read_text())
    before = {a.note.id: a.part for a in run.before}
    report = dict(
        before=notation_metrics(old, run.before, run.rhythm),
        after=notation_metrics(new, run.assignments, run.rhythm),
        connected_phrase_part_changes_before=linked_switches(run.phrases, run.before),
        connected_phrase_part_changes_after=linked_switches(run.phrases, run.assignments),
        identity_links=sum(p.get("predecessor") is not None for p in run.phrases["phrases"]),
        reassigned_source_notes=sum(before.get(a.note.id) != a.part for a in run.assignments),
        previously_scored_now_unassigned=sum(
            bool(before.get(a.note.id)) and a.part is None for a in run.assignments
        ),
        raw_predictions_unchanged=True,
        limitations=[
            "Voice identity and rhythmic-pattern evidence are heuristics, not ground-truth accuracy.",
            "Real rests and handoffs remain possible; infeasible fits are retained and reported.",
        ],
    )
    (output / "diagnostics/refinement_report.json").write_text(json.dumps(report, indent=2))
    return report

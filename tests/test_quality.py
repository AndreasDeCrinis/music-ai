import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from brass_transcriber.backend import strip_training_buffers
from brass_transcriber.cli import app, save_run_settings
from brass_transcriber.consensus import build_consensus
from brass_transcriber.evidence import PitchEvidence
from brass_transcriber.models import (
    Assignment,
    EnsemblePreset,
    NoteEvent,
    TranscriptionResult,
    load_ensemble,
    load_parts,
    read_midi,
)
from brass_transcriber.notation import build_score, export_score, musescore_export
from brass_transcriber.phrases import assign_phrases
from brass_transcriber.rhythm import Rhythm, quarters_to_seconds, seconds_to_quarters, tempo_map
from brass_transcriber.rhythm_quality import BeatAnalysis, beat_rhythm


def mapped_rhythm():
    return Rhythm(
        tempo=120,
        detected_tempo=120,
        beats=[0, 2, 6],
        mode="beats",
        grid_times=[0, 2, 6],
        grid_positions=[0, 4, 8],
    )


@pytest.mark.parametrize("seconds", [-1, 0, 1, 2, 3, 6, 9])
def test_tempo_map_inverse_and_extrapolation(seconds):
    rhythm = mapped_rhythm()
    assert quarters_to_seconds(seconds_to_quarters(seconds, rhythm), rhythm) == pytest.approx(
        seconds
    )
    assert tempo_map(rhythm) == [(0, 120), (4, 60)]


def test_rejects_nonmonotonic_beat_map():
    with pytest.raises(ValueError, match="strictly increasing"):
        Rhythm(
            tempo=120,
            detected_tempo=None,
            beats=[],
            mode="beats",
            grid_times=[0, 1, 1],
            grid_positions=[0, 1, 2],
        )


def prediction(notes):
    return TranscriptionResult(notes=notes, model="fixture", device="cpu", runtime=0)


def event(id, pitch=60, start=0, end=0.5):
    return NoteEvent(id=id, pitch=pitch, start=start, end=end)


def test_primary_survives_missing_augmentations_and_preserves_timing():
    n = event("primary", start=1.1, end=1.7)
    result, selected, decisions = build_consensus(
        {"ps": prediction([n]), "up": prediction([]), "down": prediction([])}, "ps"
    )
    assert len(selected) == len(result.notes) == 1
    assert (selected[0].pitch, selected[0].start, selected[0].end) == (60, 1.1, 1.7)
    assert selected[0].provenance == ["ps:primary"]
    assert decisions[0]["pass_support"] == 1


def test_duplicate_notes_in_one_pass_do_not_count_as_agreement():
    _, selected, decisions = build_consensus(
        {"ps": prediction([]), "up": prediction([event("a"), event("b")])}, "ps"
    )
    assert not selected
    assert all(d["pass_support"] == 1 for d in decisions)
    _, selected, _ = build_consensus(
        {"ps": prediction([]), "up": prediction([event("a")]), "down": prediction([event("b")])},
        "ps",
    )
    assert len(selected) == 1 and len(selected[0].provenance) == 2


def test_listener_only_notes_are_kept_unless_they_duplicate_a_kept_note():
    passes = {
        "ps": prediction([event("melody", pitch=80, start=0, end=1)]),
        # Same pitch, onset outside the matching tolerance: a timing-shifted duplicate.
        "separated": prediction(
            [event("shifted", pitch=80, start=0.2, end=1), event("clarinet", pitch=83, end=1)]
        ),
        "up": prediction([event("alone", pitch=88)]),
    }
    _, selected, decisions = build_consensus(passes, "ps", trusted=frozenset({"separated"}))
    reasons = {d["sources"][0]["note_id"]: d["reason"] for d in decisions}
    assert {n.pitch for n in selected} == {80, 83}
    assert reasons["shifted"] == "duplicates a kept same-pitch note"
    assert reasons["clarinet"] == "separated upper-register listener note"
    assert reasons["alone"] == "unsupported alternate-only note"
    _, selected, _ = build_consensus(passes, "ps")
    assert {n.pitch for n in selected} == {80}


def test_listener_confirms_without_changing_timing_and_flags_dependence():
    passes = {
        "ps": prediction([]),
        "up": prediction([event("up", pitch=80, start=0, end=1)]),
        "down": prediction([event("down", pitch=80, start=0.02, end=1.1)]),
        "separated": prediction(
            [event("confirm", pitch=80, start=0.05, end=2), event("only", pitch=84, end=1)]
        ),
    }
    _, selected, decisions = build_consensus(passes, "ps", trusted=frozenset({"separated"}))
    confirmed = next(n for n in selected if n.pitch == 80)
    assert (confirmed.start, confirmed.end) == (0.01, 1.05)
    flags = {d["pitch"]: d["listener_dependent"] for d in decisions}
    assert flags == {80: False, 84: True}


def test_added_listener_phrases_never_move_existing_assignments():
    from brass_transcriber.phrases import add_phrases

    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    held = NoteEvent(id="held", pitch=82, start=0, end=2)
    existing = [Assignment(note=held, part="eb_clarinet", confidence=0.5, reason="fixture")]
    extra = [NoteEvent(id="new", pitch=83, start=0.5, end=1.5)]
    result = add_phrases(existing, extra, parts)
    assert result[0] == existing[0]
    assert next(a for a in result if a.note.id == "new").part not in {None, "eb_clarinet"}


def flat_evidence(saliences):
    """Evidence where each listed pitch has the given relative activation."""
    pitches = np.array(sorted(saliences) + [100])
    rows = [[saliences.get(p, 1.0)] * 400 for p in pitches[:-1]] + [[1.0] * 400]
    return PitchEvidence(np.array(rows), np.arange(400) * 0.01, pitches)


def test_alternate_duplicates_octave_ghosts_and_weak_single_pass_notes_are_rejected():
    evidence = flat_evidence({60: 1.0, 72: 0.1, 64: 0.15, 67: 0.15, 62: 1.0})
    passes = {
        "ps": prediction(
            [
                event("root", 60, 0, 1),
                event("ghost", 72, 0, 1),
                event("lonely", 64, 1.5, 2),
                event("confirmed", 67, 2.5, 3),
            ]
        ),
        # Same pitch as "root" but outside the matching tolerance: one played note.
        "up": prediction(
            [event("late", 60, 0.09, 1), event("c2", 67, 2.5, 3), event("g2", 72, 0, 1)]
        ),
        "down": prediction([event("late2", 60, 0.09, 1), event("new", 62, 3.2, 3.8)]),
    }
    passes["down"].notes.append(event("new2", 62, 3.2, 3.8))
    passes["up"].notes.append(event("new3", 62, 3.2, 3.8))
    _, selected, decisions = build_consensus(passes, "ps", evidence)
    reasons = {d["sources"][0]["note_id"]: d["reason"] for d in decisions}
    assert sorted(n.pitch for n in selected) == [60, 62, 67]
    assert reasons["late"] == "duplicates a kept same-pitch note"
    assert reasons["ghost"] == "octave overtone of a kept note with weak harmonic evidence"
    assert reasons["lonely"] == "single-pass note with weak harmonic evidence"


def test_unplaceable_phrase_is_split_into_free_parts():
    from brass_transcriber.phrases import place_unassigned

    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)[:2]
    busy = NoteEvent(id="busy", pitch=80, start=0, end=1)
    line = [NoteEvent(id=f"n{i}", pitch=78, start=i * 0.5, end=i * 0.5 + 0.4) for i in range(4)]
    reason = "no sufficiently supported part for phrase-0001; retained for review"
    assignments = [
        Assignment(note=busy, part="eb_clarinet", confidence=0.5, reason="fixture"),
        Assignment(
            note=NoteEvent(id="busy2", pitch=70, start=0, end=1),
            part="bb_clarinet",
            confidence=0.5,
            reason="fixture",
        ),
    ] + [Assignment(note=n, part=None, confidence=0, reason=reason) for n in line]
    placed = {a.note.id: a.part for a in place_unassigned(assignments, parts)}
    assert placed["busy"] == "eb_clarinet" and placed["busy2"] == "bb_clarinet"
    assert placed["n0"] is None and placed["n1"] is None
    assert placed["n2"] == placed["n3"] is not None


def test_octave_shadow_filters_weak_overtones_only():
    from brass_transcriber.listeners import octave_shadow

    melody = (0.0, 1.0, 68, 0.8)
    assert octave_shadow((0.0, 1.0, 80, 0.3), [melody])
    assert not octave_shadow((0.0, 1.0, 80, 0.7), [melody])
    assert not octave_shadow((0.0, 1.0, 81, 0.3), [melody])


def test_weak_short_rejection_is_auditable_and_reversible():
    evidence = PitchEvidence(
        np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.0, 0.1]), np.array([60, 61])
    )
    n = event("short", end=0.04)
    result, selected, decisions = build_consensus({"ps": prediction([n])}, "ps", evidence)
    assert not selected and len(result.notes) == 1
    assert decisions[0]["action"] == "reject"
    assert decisions[0]["sources"][0]["note_id"] == n.id


def test_phrase_assignment_keeps_line_together_and_source_notes_intact():
    notes = [
        event(str(i), pitch=p, start=i * 0.5, end=i * 0.5 + 0.45)
        for i, p in enumerate([59, 63, 61, 59, 52])
    ]
    assignments, _ = assign_phrases(notes, load_parts(), beam_size=16)
    assert len({a.part for a in assignments}) == 1
    assert {a.note.id for a in assignments} == {n.id for n in notes}
    assert all(a.note == next(n for n in notes if n.id == a.note.id) for a in assignments)


def test_ambiguous_short_fragment_remains_unassigned_without_losing_pitch():
    n = event("opening-E3", pitch=52, start=1.89, end=2.048).model_copy(
        update={"program": 56, "provenance": ["baseline:original"]}
    )
    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    assignments, _ = assign_phrases([n], parts)
    assert assignments[0].part is None
    assert assignments[0].note == n
    agreed = n.model_copy(update={"provenance": ["baseline:original", "ps:confirmation"]})
    assignments, _ = assign_phrases([agreed], parts)
    assert assignments[0].part is not None


def test_remove_only_training_augmentation_buffers():
    learned = object()
    state = {"encoder.weight": learned, "pitchshift.pshifters.1.kernel": object()}
    cleaned, removed = strip_training_buffers(state)
    assert cleaned == {"encoder.weight": learned}
    assert removed == ["pitchshift.pshifters.1.kernel"]
    assert len(state) == 2


def test_sunchyme_opening_phase_is_stable():
    prediction = BeatAnalysis(
        beats=[0.28, 0.78, 1.3, 1.84, 2.34, 2.88, 3.36, 3.84],
        downbeats=[1.84, 3.84],
        beat_numbers=[2, 3, 4, 1, 2, 3, 4, 1],
        runtime=0,
    )
    rhythm = beat_rhythm(prediction)
    assert seconds_to_quarters(1.84, rhythm) == 4
    assert seconds_to_quarters(3.84, rhythm) == 8


def test_early_downbeat_does_not_compress_a_whole_bar():
    prediction = BeatAnalysis(
        beats=[0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3],
        downbeats=[0.3, 2.3, 4.3],
        beat_numbers=[1, 2, 3, 4, 1, 2, 3, 4, 1],
        runtime=0,
    )
    rhythm = beat_rhythm(prediction)
    assert quarters_to_seconds(4, rhythm) == pytest.approx(2.3)
    assert 60 < tempo_map(rhythm)[0][1] < 180


def test_detected_pickup_does_not_create_false_slow_opening():
    from brass_transcriber.rhythm import regularize_tempo

    beats = [0.38, 0.86, 1.3, 1.76, 2.2, 2.66, 3.12, 3.56, 4.02, 4.48, 4.94, 5.38, 5.84, 6.3]
    prediction = BeatAnalysis(
        beats=beats,
        downbeats=[0.86, 2.66, 4.48, 6.3],
        beat_numbers=[4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1],
        runtime=0,
    )
    rhythm = regularize_tempo(beat_rhythm(prediction))
    assert rhythm.pickup == 2
    assert seconds_to_quarters(0.86, rhythm) == 2
    assert seconds_to_quarters(2.66, rhythm) == 6
    assert len(tempo_map(rhythm)) == 1
    assert tempo_map(rhythm)[0][1] in {132, 133}


def test_quality_resume_preserves_selection_metadata_and_map(tmp_path):
    source = tmp_path / "source"
    intermediate = source / "intermediate"
    intermediate.mkdir(parents=True)
    accepted, rejected = event("accepted"), event("rejected", pitch=61, start=0.6, end=0.8)
    result = prediction([accepted, rejected])
    (intermediate / "model_notes.json").write_text(result.model_dump_json())
    (intermediate / "quality_notes.json").write_text(json.dumps([accepted.model_dump()]))
    sf.write(intermediate / "normalized.wav", np.zeros(16000), 16000)
    ensemble = load_ensemble(preset=EnsemblePreset.BRASSARANKA)
    rhythm = mapped_rhythm()
    save_run_settings(source, rhythm, 1, 0, ensemble.parts, ensemble)
    settings_path = intermediate / "run_settings.json"
    settings = json.loads(settings_path.read_text())
    settings["quality"] = {"pipeline": "test", "selected_notes": 1}
    settings_path.write_text(json.dumps(settings))
    original = settings_path.read_bytes()
    output = tmp_path / "reexport"
    response = CliRunner().invoke(
        app,
        [
            "resume",
            str(source),
            "--output",
            str(output),
            "--ensemble",
            "brassaranka",
            "--quantize",
            "1/8",
            "--no-mscz",
        ],
    )
    assert response.exit_code == 0, response.output
    saved = json.loads((output / "intermediate/run_settings.json").read_text())
    assert saved["quality"] == settings["quality"]
    assert saved["rhythm"]["grid_times"] == rhythm.grid_times
    assert saved["rhythm"]["grid"] == "1/8"
    assigned = json.loads((output / "intermediate/assigned_voices.json").read_text())
    assert next(a for a in assigned if a["note"]["id"] == "accepted")["part"]
    assert next(a for a in assigned if a["note"]["id"] == "rejected")["part"] is None
    assert settings_path.read_bytes() == original


def test_same_program_desks_get_distinct_channels_for_overlapping_unisons(tmp_path):
    import mido

    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    assignments = [
        Assignment(
            note=event("short", end=0.5), part="trumpet_1", confidence=0.5, reason="fixture"
        ),
        Assignment(
            note=event("long", end=1.5), part="flugelhorn_1", confidence=0.5, reason="fixture"
        ),
    ]
    score, _ = build_score(assignments, parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 2)
    export_score(score, tmp_path)
    midi = mido.MidiFile(tmp_path / "full_score.mid")
    channels = [msg.channel for track in midi.tracks for msg in track if msg.type == "note_on"]
    assert len(set(channels)) == 2
    expected = read_midi(tmp_path / "full_score.mid")
    assert sorted(n.end for n in expected) == [0.5, 1.5]
    binary = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary.exists():
        pytest.skip("MuseScore CLI not installed")
    assert musescore_export(tmp_path) is None
    subprocess.run(
        [
            str(binary),
            "--sound-profile",
            "MuseScore Basic",
            "-o",
            str(tmp_path / "playback.mid"),
            str(tmp_path / "full_score.mscz"),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    actual = read_midi(tmp_path / "playback.mid")
    assert sorted(n.end for n in actual) == pytest.approx([0.5, 1.5], abs=0.002)


@pytest.mark.parametrize("pickup", [0, 1])
@pytest.mark.parametrize("within_bar", [False, True])
def test_variable_tempo_native_playback_preserves_pitches_and_onsets(tmp_path, pickup, within_bar):
    rhythm = mapped_rhythm().model_copy(update={"pickup": pickup})
    if within_bar:
        rhythm = Rhythm(
            tempo=120,
            detected_tempo=120,
            beats=[],
            pickup=pickup,
            mode="beats",
            grid_times=np.cumsum(
                [0, 0.46, 0.50, 0.54, 0.48, 0.52, 0.46, 0.5, 0.54, 0.48, 0.5, 0.46, 0.52]
            ).tolist(),
            grid_positions=list(range(13)),
        )
    parts = load_parts()
    times = [quarters_to_seconds(q, rhythm) for q in [0, 2, 4, 6, 10]]
    assignments = [
        Assignment(
            note=event(str(i), pitch=60 + i, start=t, end=t + 0.5),
            part="solo_cornet",
            confidence=0.5,
            reason="fixture",
        )
        for i, t in enumerate(times)
    ]
    score, _ = build_score(assignments, parts, rhythm, 7)
    export_score(score, tmp_path)
    expected = read_midi(tmp_path / "full_score.mid")
    assert len(expected) == 5
    for i, (n, t) in enumerate(zip(expected, times, strict=True)):
        assert n.pitch == 60 + i
        assert n.start == pytest.approx(t, abs=0.001)
    binary = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary.exists():
        pytest.skip("MuseScore CLI not installed")
    assert musescore_export(tmp_path) is None
    subprocess.run(
        [
            str(binary),
            "--sound-profile",
            "MuseScore Basic",
            "-o",
            str(tmp_path / "playback.mid"),
            str(tmp_path / "full_score.mscz"),
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    actual = read_midi(tmp_path / "playback.mid")
    assert Counter((n.pitch, round(n.start, 2), round(n.end, 2)) for n in actual) == Counter(
        (n.pitch, round(n.start, 2), round(n.end, 2)) for n in expected
    )

import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from brass_transcriber.cli import app
from brass_transcriber.models import (
    Assignment,
    EnsemblePreset,
    NoteEvent,
    TranscriptionResult,
    load_parts,
    read_midi,
)
from brass_transcriber.notation import build_score, export_score, musescore_export
from brass_transcriber.phrases import assign_phrases
from brass_transcriber.rhythm import Rhythm
from brass_transcriber.rhythm_fit import fit_assignments


def rhythm():
    return Rhythm(tempo=120, detected_tempo=120, beats=[], notation_mode="phrases")


def event(id, pitch, start, end):
    return NoteEvent(id=id, pitch=pitch, start=start / 2, end=end / 2, program=56)


def assigned(notes, part="flugelhorn_3"):
    return [Assignment(note=n, part=part, confidence=0.5, reason="fixture") for n in notes]


def test_close_attacks_preserved_without_extra_voice_and_with_bounded_changes():
    notes = [event("a", 60, 0.24, 0.245), event("b", 62, 0.26, 0.28)]
    fits = fit_assignments(assigned(notes), rhythm())
    assert fits["a"].offset <= fits["b"].onset
    assert not any(f.unresolved for f in fits.values())
    for n in notes:
        assert abs(fits[n.id].onset - n.start * 2) <= 0.18 + 1e-8
        assert abs(fits[n.id].offset - n.end * 2) <= 0.20 + 1e-8
    score, mapping = build_score(
        assigned(notes), load_parts(preset=EnsemblePreset.BRASSARANKA), rhythm(), 1
    )
    assert all(m["notation_voice"] == 1 for m in mapping)
    assert {m["source_id"] for m in mapping} == {"a", "b"}


def test_legato_rounding_gap_closed_and_real_staccato_rest_preserved():
    fits = fit_assignments(
        assigned([event("a", 60, 0, 0.86), event("b", 62, 0.95, 1.95)]), rhythm()
    )
    assert fits["a"].offset == fits["b"].onset
    fits = fit_assignments(assigned([event("a", 60, 0, 0.5), event("b", 62, 1, 1.5)]), rhythm())
    assert fits["b"].onset - fits["a"].offset == 0.5


def test_shared_attacks_stay_aligned_across_instruments():
    a = assigned([event("a", 60, 0.23, 0.73)], "trumpet_1") + assigned(
        [event("b", 64, 0.26, 0.76)], "flugelhorn_1"
    )
    fits = fit_assignments(a, rhythm())
    assert fits["a"].onset == fits["b"].onset == 0.25


def test_triplets_require_both_offbeat_thirds_and_export_correctly(tmp_path):
    notes = [event(str(i), 60 + i, i / 3, (i + 1) / 3) for i in range(3)]
    fits = fit_assignments(assigned(notes), rhythm())
    assert [fits[n.id].onset for n in notes] == pytest.approx([0, 1 / 3, 2 / 3])
    score, mapping = build_score(
        assigned(notes), load_parts(preset=EnsemblePreset.BRASSARANKA), rhythm(), 1
    )
    export_score(score, tmp_path)
    assert (
        ET.parse(tmp_path / "full_score.musicxml").find(".//time-modification/actual-notes").text
        == "3"
    )
    source = read_midi(tmp_path / "full_score.mid")
    assert [n.start for n in source] == pytest.approx([0, 1 / 6, 1 / 3], abs=0.001)
    lone = fit_assignments(assigned([event("a", 60, 0, 0.3), event("b", 62, 0.35, 0.9)]), rhythm())
    assert lone["b"].onset == 0.25
    binary = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary.exists():
        pytest.skip("MuseScore unavailable")
    assert musescore_export(tmp_path) is None
    subprocess.run(
        [
            str(binary),
            "--sound-profile",
            "MuseScore Basic",
            "-o",
            str(tmp_path / "native.mid"),
            str(tmp_path / "full_score.mscz"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    actual = read_midi(tmp_path / "native.mid")
    assert Counter((n.pitch, round(n.start, 3)) for n in actual) == Counter(
        (n.pitch, round(n.start, 3)) for n in source
    )


def test_repeated_motif_duration_prior_links_across_rests():
    pitches = [60, 62, 65, 64, 60]
    notes = [
        event(f"{repeat}-{i}", p, repeat * 12 + i, repeat * 12 + i + length)
        for repeat, length in [(0, 0.8), (1, 0.95)]
        for i, p in enumerate(pitches)
    ]
    fits = fit_assignments(assigned(notes), rhythm())
    assert all(f.motif_support >= 2 for f in fits.values())
    assert [fits[f"0-{i}"].offset - fits[f"0-{i}"].onset for i in range(5)] == pytest.approx(
        [fits[f"1-{i}"].offset - fits[f"1-{i}"].onset for i in range(5)]
    )


def two_parts():
    template = load_parts(preset=EnsemblePreset.BRASSARANKA)[6]
    return [
        template.model_copy(update={"id": "a", "name": "A", "preferred_pitch": 60}),
        template.model_copy(update={"id": "b", "name": "B", "preferred_pitch": 65}),
    ]


def test_voice_identity_survives_rest_and_register_shift():
    notes = [event(str(i), p, i, i + 0.8) for i, p in enumerate([60, 62, 60])]
    notes += [event("r" + str(i), p, 8 + i, 8 + i + 0.8) for i, p in enumerate([64, 66, 64])]
    old, _ = assign_phrases(notes, two_parts(), rhythm=rhythm(), persistent=False)
    new, metadata = assign_phrases(notes, two_parts(), rhythm=rhythm())
    assert len({a.part for a in old}) == 2
    assert len({a.part for a in new}) == 1
    assert len({p["voice"] for p in metadata["phrases"]}) == 1
    assert all(a.note in notes for a in new)


def test_persistent_voice_allows_range_required_handoff():
    parts = two_parts()
    parts[0] = parts[0].model_copy(update={"extended": [54, 63], "typical": [54, 63]})
    parts[1] = parts[1].model_copy(update={"extended": [62, 77], "typical": [62, 77]})
    notes = [event(str(i), p, i, i + 0.8) for i, p in enumerate([60, 62, 60])]
    notes += [event("r" + str(i), p, 8 + i, 8 + i + 0.8) for i, p in enumerate([64, 66, 64])]
    new, _ = assign_phrases(notes, parts, rhythm=rhythm())
    assert {a.part for a in new} == {"a", "b"}
    assert all(a.part is not None for a in new)


def test_repeated_motif_identity_survives_interlude_and_long_rest():
    pitches = [60, 62, 65, 64, 62, 60]
    notes = [event("first" + str(i), p, i, i + 0.8) for i, p in enumerate(pitches)]
    notes += [event("middle" + str(i), p, 10 + i, 10 + i + 0.8) for i, p in enumerate([59, 61, 63])]
    notes += [event("return" + str(i), p, 100 + i, 100 + i + 0.8) for i, p in enumerate(pitches)]
    _, metadata = assign_phrases(notes, two_parts(), rhythm=rhythm())
    assert len({p["voice"] for p in metadata["phrases"]}) == 1
    assert "repeated melodic motif" in metadata["phrases"][-1]["identity_evidence"]


def test_cached_refine_preserves_rejections_predictions_and_source(tmp_path, monkeypatch):
    from brass_transcriber.backend import YourMT3Backend

    monkeypatch.setattr(
        YourMT3Backend, "transcribe", lambda *args: pytest.fail("Inference is forbidden")
    )
    source = tmp_path / "source"
    inter = source / "intermediate"
    inter.mkdir(parents=True)
    sf.write(inter / "normalized.wav", np.zeros(48000, dtype="float32"), 16000)
    notes = [event(str(i), 60 + i, i, i + 0.8) for i in range(3)]
    rejected = event("rejected", 65, 4, 4.1)
    model = TranscriptionResult(notes=notes + [rejected], model="fixture", device="cpu", runtime=1)
    (inter / "model_notes.json").write_text(model.model_dump_json())
    (inter / "quality_notes.json").write_text(json.dumps([n.model_dump() for n in notes]))
    ensemble = load_parts(preset=EnsemblePreset.BRASSARANKA)
    old = assigned(notes) + [
        Assignment(
            note=rejected,
            part="flugelhorn_3",
            confidence=0.5,
            reason="obsolete assignment excluded by selection",
        )
    ]
    (inter / "assigned_voices.json").write_text(json.dumps([a.model_dump() for a in old]))
    r = rhythm().model_copy(update={"notation_mode": "independent"})
    (inter / "run_settings.json").write_text(
        json.dumps(
            dict(
                rhythm=r.model_dump(),
                parts=[p.model_dump() for p in ensemble],
                audio_duration=3,
                excerpt_start=0,
            )
        )
    )
    score, mapping = build_score(old, ensemble, r, 3)
    (inter / "score_note_provenance.json").write_text(json.dumps(mapping))
    originals = {p.name: p.read_bytes() for p in inter.iterdir()}
    out = tmp_path / "refined"
    response = CliRunner().invoke(app, ["refine", str(source), "--output", str(out), "--no-mscz"])
    assert response.exit_code == 0, response.output
    assert all(p.read_bytes() == originals[p.name] for p in inter.iterdir())
    assert (out / "intermediate/model_notes.json").read_bytes() == originals["model_notes.json"]
    result = json.loads((out / "intermediate/assigned_voices.json").read_text())
    assert next(a for a in result if a["note"]["id"] == "rejected")["part"] is None
    assert {a["note"]["id"] for a in result} == {n.id for n in model.notes}
    settings = json.loads((out / "intermediate/run_settings.json").read_text())
    response = CliRunner().invoke(app, ["resume", str(out), "--no-mscz", "--tempo", "125"])
    assert response.exit_code == 0, response.output
    assert (
        json.loads((out / "intermediate/run_settings.json").read_text())["refinement"]
        == settings["refinement"]
    )
    assert (
        json.loads((out / "intermediate/run_settings.json").read_text())["rhythm"]["notation_mode"]
        == "phrases"
    )
    response = CliRunner().invoke(
        app, ["refine", str(source), "--output", str(source / "nested"), "--no-mscz"]
    )
    assert response.exit_code != 0


def test_incomplete_triplets_across_parts_do_not_form_malformed_groups(tmp_path):
    a = assigned(
        [
            event("a0", 60, 0, 0.3),
            event("a1", 62, 1 / 3, 2 / 3),
            event("a2", 60, 4, 4 + 1 / 3),
            event("a3", 62, 4 + 1 / 3, 4 + 2 / 3),
            event("a4", 64, 4 + 2 / 3, 5),
        ],
        "trumpet_1",
    )
    a += assigned([event("b0", 67, 2 / 3, 1), event("b1", 67, 4.35, 4.85)], "flugelhorn_1")
    fits = fit_assignments(a, rhythm())
    assert fits["a1"].onset == 0.25
    assert fits["b0"].onset == 0.75
    assert fits["a3"].onset == pytest.approx(4 + 1 / 3)
    score, _ = build_score(a, load_parts(preset=EnsemblePreset.BRASSARANKA), rhythm(), 3)
    export_score(score, tmp_path)
    binary = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary.exists():
        pytest.skip("MuseScore unavailable")
    assert musescore_export(tmp_path) is None


def test_long_motif_release_does_not_introduce_exotic_tuplets():
    from fractions import Fraction

    notes = [
        event(f"{repeat}-{i}", p, repeat * 20 + i, repeat * 20 + i + (0.8 if i < 4 else 6.123))
        for repeat in range(2)
        for i, p in enumerate([60, 62, 65, 64, 60])
    ]
    fits = fit_assignments(assigned(notes), rhythm())
    assert all(
        96 % Fraction(f.offset).limit_denominator(96).denominator == 0 for f in fits.values()
    )

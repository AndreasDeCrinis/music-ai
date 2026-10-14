import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from brass_transcriber.backend import YourMT3Backend
from brass_transcriber.cli import app, save_run_settings
from brass_transcriber.models import (
    Assignment,
    EnsemblePreset,
    NoteEvent,
    TranscriptionResult,
    load_ensemble,
    load_parts,
    read_midi,
    write_midi,
)
from brass_transcriber.notation import build_score, export_score, musescore_export
from brass_transcriber.orchestration import assign_voices
from brass_transcriber.rhythm import Rhythm


def test_brassaranka_roster_and_austrian_notation():
    config = load_ensemble(preset=EnsemblePreset.BRASSARANKA)
    counts = Counter(p.family for p in config.parts)
    assert counts == {
        "clarinet": 2,
        "trumpet": 2,
        "flugelhorn": 4,
        "tenor_horn": 1,
        "baritone": 1,
        "trombone": 3,
        "bass": 1,
        "percussion": 1,
    }
    assert len(config.parts) == 15
    assert config.sources and config.assumptions
    for p in config.parts:
        if p.family in {"trombone", "bass"}:
            assert (p.clef, p.written_to_sounding) == ("bass", "P1")
        if p.family in {"tenor_horn", "baritone"}:
            assert (p.clef, p.written_to_sounding) == ("treble", "M-9")
        if not p.percussion:
            assert p.max_simultaneous == 1


def test_routes_supported_drums_and_preserves_unknowns():
    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    notes = [
        NoteEvent(id=str(p), pitch=p, start=0, end=0.1, drum=True) for p in [36, 38, 42, 0, 82]
    ]
    notes.append(NoteEvent(id="clarinet", pitch=76, start=1, end=2, program=71))
    result = assign_voices(notes, parts)
    by_id = {a.note.id: a for a in result}
    assert all(by_id[str(p)].part == "drum_set" for p in [36, 38, 42])
    assert by_id["0"].part is None and by_id["82"].part is None
    assert by_id["clarinet"].part == "bb_clarinet"
    assert len(result) == len(notes)
    assert assign_voices(notes, load_parts())[0].part is None


def test_reed_predictions_count_as_clarinet_evidence():
    from brass_transcriber.phrases import assign_phrases

    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    # A mid-register line that register alone would give to a flugelhorn.
    notes = [
        NoteEvent(id=f"sax-{i}", pitch=p, start=i * 0.5, end=i * 0.5 + 0.45, program=65)
        for i, p in enumerate([70, 72, 74, 72, 70, 69])
    ]
    assert {a.part for a in assign_voices(notes, parts)} <= {"bb_clarinet", "eb_clarinet"}
    assignments, _ = assign_phrases(notes, parts)
    assert {a.part for a in assignments} == {"bb_clarinet"}


def drum_fixture():
    notes = [
        NoteEvent(id=f"hit-{i}", pitch=p, start=t, end=t + 0.1, drum=True, velocity=70 + i)
        for i, (p, t) in enumerate([(36, 0), (38, 0), (42, 0.5), (46, 1), (49, 1)])
    ]
    return [Assignment(note=n, part="drum_set", confidence=0.7, reason="fixture") for n in notes]


def test_drum_musicxml_ids_and_midi_roundtrip(tmp_path):
    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    assignments = drum_fixture()
    score, mapping = build_score(
        assignments, parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 2.173
    )
    export_score(score, tmp_path)
    root = ET.parse(tmp_path / "full_score.musicxml")
    drums = root.findall("part")[-1]
    definitions = {
        m.get("id"): int(m.findtext("midi-unpitched")) - 1
        for m in root.findall("part-list/score-part")[-1].findall("midi-instrument")
        if m.find("midi-unpitched") is not None
    }
    xml_hits = [n for n in drums.findall(".//note") if n.find("unpitched") is not None]
    assert Counter(definitions[n.find("instrument").get("id")] for n in xml_hits) == Counter(
        a.note.pitch for a in assignments
    )
    assert all(
        m.findtext("midi-channel") == "10"
        for m in root.findall("part-list/score-part")[-1].findall("midi-instrument")
    )
    midi = read_midi(tmp_path / "full_score.mid")
    assert all(n.drum for n in midi)
    assert Counter((n.pitch, n.start) for n in midi) == Counter(
        (a.note.pitch, a.note.start) for a in assignments
    )
    assert {m["source_id"] for m in mapping} == {a.note.id for a in assignments}
    assert (tmp_path / "parts/drum_set.musicxml").exists()


def test_brassaranka_native_templates_and_percussion_playback(tmp_path):
    binary = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary.exists():
        pytest.skip("MuseScore CLI not installed")
    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    assignments = drum_fixture() + [
        Assignment(
            note=NoteEvent(
                id=p.id, pitch=p.preferred_pitch, start=2 + i * 0.125, end=2.1 + i * 0.125
            ),
            part=p.id,
            confidence=0.5,
            reason="fixture",
        )
        for i, p in enumerate(parts)
        if not p.percussion
    ]
    score, _ = build_score(assignments, parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 5)
    export_score(score, tmp_path)
    assert musescore_export(tmp_path) is None
    with ZipFile(tmp_path / "full_score.mscz") as archive:
        root = ET.fromstring(
            archive.read(next(n for n in archive.namelist() if n.endswith(".mscx")))
        )
    templates = [p.findtext("Instrument/instrumentId") for p in root.findall("./Score/Part")]
    assert len(templates) == 15
    assert sum(t.startswith("wind.reed.clarinet") for t in templates) == 2
    assert sum(t.startswith("brass.trumpet") for t in templates) == 2
    assert sum(t == "brass.flugelhorn" for t in templates) == 4
    assert templates[-1] == "drum.group"
    assert all(not t.startswith("voice.") for t in templates)
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
    expected = read_midi(tmp_path / "full_score.mid")
    assert Counter((n.drum, n.pitch, round(n.start, 2)) for n in actual) == Counter(
        (n.drum, n.pitch, round(n.start, 2)) for n in expected
    )


def test_reassign_cached_run_to_new_preset_without_changing_source(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cached reorchestration must not invoke inference")

    monkeypatch.setattr(YourMT3Backend, "transcribe", forbidden)
    source = tmp_path / "source"
    intermediate = source / "intermediate"
    intermediate.mkdir(parents=True)
    r = Rhythm(tempo=120, detected_tempo=None, beats=[])
    n = NoteEvent(id="source-note", pitch=76, start=0, end=0.5, program=71)
    drum = NoteEvent(id="source-drum", pitch=36, start=0, end=0.1, drum=True)
    model = TranscriptionResult(notes=[n, drum], model="cached YourMT3+", device="cpu", runtime=7)
    (intermediate / "model_notes.json").write_text(model.model_dump_json())
    (intermediate / "assigned_voices.json").write_text(
        json.dumps(
            [
                Assignment(
                    note=n, part="solo_cornet", confidence=0.5, reason="old preset"
                ).model_dump(),
                Assignment(note=drum, part=None, confidence=0, reason="old preset").model_dump(),
            ]
        )
    )
    sf.write(intermediate / "normalized.wav", np.zeros(16000), 16000)
    write_midi([n, drum], intermediate / "raw_transcription.mid")
    save_run_settings(source, r, 1, 0, load_parts())
    original = {p.name: p.read_bytes() for p in intermediate.iterdir() if p.is_file()}
    destination = tmp_path / "new"
    response = CliRunner().invoke(
        app,
        [
            "resume",
            str(source),
            "--ensemble",
            "brassaranka",
            "--output",
            str(destination),
            "--no-mscz",
        ],
    )
    assert response.exit_code == 0, response.output
    assert all((intermediate / name).read_bytes() == data for name, data in original.items())
    assert (destination / "intermediate/model_notes.json").read_bytes() == original[
        "model_notes.json"
    ]
    assigned = json.loads((destination / "intermediate/assigned_voices.json").read_text())
    assert {a["note"]["id"]: a["part"] for a in assigned} == {
        "source-note": "bb_clarinet",
        "source-drum": "drum_set",
    }
    report = json.loads((destination / "diagnostics/transcription_report.json").read_text())
    assert report["ensemble"]["id"] == "brassaranka"
    assert report["assigned_percussion_notes"] == 1
    assert report["inference_runtime"] == 7
    assert len(list((destination / "parts").glob("*.musicxml"))) == 15


def test_cli_rejects_conflicting_config_and_preset():
    response = CliRunner().invoke(
        app, ["transcribe", "missing.mp3", "--ensemble", "brassaranka", "--config", "missing.yaml"]
    )
    assert response.exit_code == 2
    assert "Choose --ensemble or --config" in response.output

from pathlib import Path

import pytest
from music21 import converter, interval

from brass_transcriber.backend import deduplicate
from brass_transcriber.models import Assignment, EnsemblePreset, NoteEvent, load_parts, read_midi
from brass_transcriber.notation import build_score, export_score
from brass_transcriber.orchestration import assign_voices
from brass_transcriber.rhythm import Rhythm, quantize


def event(id="n", pitch=60, start=0, end=1, **kwargs):
    return NoteEvent(id=id, pitch=pitch, start=start, end=end, **kwargs)


@pytest.mark.parametrize(
    "part",
    load_parts() + [p for p in load_parts(preset=EnsemblePreset.BRASSARANKA) if not p.percussion],
    ids=lambda p: p.id,
)
def test_transposition_roundtrip(part, tmp_path):
    source = event(pitch=part.preferred_pitch)
    assignment = Assignment(note=source, part=part.id, confidence=0.5, reason="test")
    score, mapping = build_score(
        [assignment], [part], Rhythm(tempo=120, detected_tempo=None, beats=[]), 2
    )
    assert (
        mapping[0]["written_pitch"]
        == source.pitch - interval.Interval(part.written_to_sounding).semitones
    )
    export_score(score, tmp_path)
    parsed = converter.parse(tmp_path / "full_score.musicxml").toSoundingPitch()
    pitches = [n.pitch.midi for n in parsed.recurse().notes]
    assert pitches == [source.pitch]
    assert [n.pitch for n in read_midi(tmp_path / "full_score.mid")] == [source.pitch]
    individual = converter.parse(tmp_path / "parts" / f"{part.id}.musicxml").toSoundingPitch()
    assert [n.pitch.midi for n in individual.recurse().notes] == [source.pitch]


def test_all_staves_and_ties(tmp_path):
    parts = load_parts()
    a = Assignment(note=event(pitch=72, end=5), part="solo_cornet", confidence=0.5, reason="test")
    score, _ = build_score([a], parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 5)
    export_score(score, tmp_path)
    parsed = converter.parse(tmp_path / "full_score.musicxml")
    assert len(parsed.parts) == 17
    assert len(list((tmp_path / "parts").glob("*.musicxml"))) == 17
    assert any(n.tie for n in parsed.recurse().notes)


def test_assignment_conserves_notes_and_constraints():
    notes = [
        event(id=str(i), pitch=p, start=i // 4, end=i // 4 + 0.9)
        for i, p in enumerate([72, 67, 60, 48] * 3)
    ]
    parts = load_parts()
    result = assign_voices(notes, parts)
    assert {a.note.id for a in result} == {n.id for n in notes}
    assert len(result) == len(notes)
    assert all(a.confidence <= 0.75 for a in result)
    for part in parts:
        selected = [a.note for a in result if a.part == part.id]
        assert all(part.extended[0] <= n.pitch <= part.extended[1] for n in selected)
        for n in selected:
            assert (
                sum(other.start <= n.start < other.end for other in selected)
                <= part.max_simultaneous
            )
    assert [a.part for a in result] == [a.part for a in assign_voices(notes, parts)]


def test_unassigned_and_drums():
    result = assign_voices([event(pitch=10), event(id="drum", drum=True)], load_parts())
    assert all(a.part is None for a in result)


def test_quantization_and_tuplets():
    r = Rhythm(tempo=120, detected_tempo=None, beats=[], grid="1/16")
    assert quantize(0.126, 0.374, r) == (0.25, 0.75)
    assert quantize(0.1, 0.101, r)[1] > quantize(0.1, 0.101, r)[0]
    triplet = r.model_copy(update={"grid": "1/12"})
    assert quantize(1 / 6, 1 / 3, triplet) == pytest.approx((1 / 3, 2 / 3))


def test_boundary_dedup():
    notes = [
        event(id="a", track=0),
        event(id="b", start=0.05, end=1.1, track=1),
        event(id="c", start=0.5, end=1.5, track=1),
    ]
    result = deduplicate(notes)
    assert len(result) == 2
    assert result[0].end == 1.1
    assert result[0].provenance == ["b"]


def test_pickup_boundary_and_playback(tmp_path):
    p = load_parts()[1]
    assignments = [
        Assignment(
            note=event(id=str(i), pitch=60 + i, start=i * 0.5, end=(i + 1) * 0.5),
            part=p.id,
            confidence=0.5,
            reason="test",
        )
        for i in range(6)
    ]
    r = Rhythm(tempo=120, detected_tempo=None, beats=[], pickup=1)
    score, _ = build_score(assignments, [p], r, 3)
    assert score.parts[0].measure(0).highestTime == 1
    assert score.parts[0].measure(1).offset == 1
    export_score(score, tmp_path)
    midi = read_midi(tmp_path / "full_score.mid")
    assert [n.start for n in midi] == pytest.approx([i * 0.5 for i in range(6)], abs=0.01)


def test_stitch_sustain_and_duplicate_attack():
    from brass_transcriber.backend import merge_chunk

    previous = [event(id="c0-0", start=0, end=5), event(id="c0-1", pitch=67, start=4, end=4.8)]
    events = [
        event(id="c1-0", start=3, end=7),
        event(id="c1-1", pitch=67, start=4.04, end=4.9),
        event(id="c1-2", pitch=67, start=5, end=6),
    ]
    merge_chunk(previous, events, 4, 10, 3, 10)
    assert len(previous) == 3
    assert previous[0].end == 7
    assert previous[0].provenance == ["c0-0", "c1-0"]
    assert previous[1].provenance == ["c0-1", "c1-1"]


def test_polyphonic_voice_numbers_and_tie_onsets(tmp_path):
    import xml.etree.ElementTree as ET

    p = load_parts()[2]
    assignments = [
        Assignment(
            note=event(id="long", pitch=70, start=3.25, end=5.375),
            part=p.id,
            confidence=0.5,
            reason="test",
        ),
        Assignment(
            note=event(id="short", pitch=66, start=3.75, end=4.5),
            part=p.id,
            confidence=0.5,
            reason="test",
        ),
    ]
    score, _ = build_score(assignments, [p], Rhythm(tempo=120, detected_tempo=None, beats=[]), 8)
    export_score(score, tmp_path)
    root = ET.parse(tmp_path / "full_score.musicxml")
    assert all(int(v.text) > 0 for v in root.findall(".//voice"))
    assert root.find(".//instrument-sound").text == "brass.cornet"
    midi = read_midi(tmp_path / "full_score.mid")
    assert sorted((n.pitch, n.start) for n in midi) == [(66, 3.75), (70, 3.25)]


def test_full_band_midi_programs_and_unisons(tmp_path):
    parts = load_parts()
    assignments = [
        Assignment(
            note=event(id=p.id, pitch=p.preferred_pitch), part=p.id, confidence=0.5, reason="test"
        )
        for p in parts
    ]
    score, _ = build_score(assignments, parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 2)
    export_score(score, tmp_path)
    midi = read_midi(tmp_path / "full_score.mid")
    from collections import Counter

    assert Counter((n.pitch, n.program) for n in midi) == Counter(
        (p.preferred_pitch, p.program) for p in parts
    )


def test_musescore_multivoice_playback(tmp_path):
    import subprocess
    from collections import Counter

    from brass_transcriber.notation import musescore_export

    binary = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary.exists():
        pytest.skip("MuseScore CLI not installed")
    parts = load_parts()
    assignments = [
        Assignment(
            note=event(id=p.id, pitch=p.preferred_pitch, start=i * 0.125, end=2 + i * 0.125),
            part=p.id,
            confidence=0.5,
            reason="test",
        )
        for i, p in enumerate(parts)
    ]
    assignments.append(
        Assignment(
            note=event(id="poly", pitch=67, start=0.75, end=2.25),
            part="solo_cornet",
            confidence=0.5,
            reason="test",
        )
    )
    score, _ = build_score(assignments, parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 5)
    export_score(score, tmp_path)
    assert musescore_export(tmp_path) is None
    # Inspect MuseScore's actual selected templates, not only MusicXML hints.
    import xml.etree.ElementTree as ET
    from zipfile import ZipFile

    with ZipFile(tmp_path / "full_score.mscz") as archive:
        native = ET.fromstring(
            archive.read(next(n for n in archive.namelist() if n.endswith(".mscx")))
        )
    native_parts = native.findall("./Score/Part")
    assert len(native_parts) == 17
    for native_part, config in zip(native_parts, parts, strict=True):
        assert native_part.findtext("Instrument/instrumentId").startswith("brass.")
        if config.family == "bass":
            assert "tuba" in native_part.findtext("Instrument/instrumentId")
            assert native_part.findtext("Instrument/transposeChromatic") == str(
                -21 if config.id == "eb_bass" else -26
            )
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
    playback = read_midi(tmp_path / "playback.mid")
    expected = read_midi(tmp_path / "full_score.mid")
    assert Counter((n.pitch, round(n.start, 2)) for n in playback) == Counter(
        (n.pitch, round(n.start, 2)) for n in expected
    )


@pytest.mark.parametrize("grid,pickup", [("1/16", 0), ("1/12", 0), ("1/16", 1)])
def test_off_grid_audio_endpoint_exports_complete_bars(tmp_path, grid, pickup):
    parts = load_parts()
    r = Rhythm(tempo=133.92857142857142, detected_tempo=None, beats=[], grid=grid, pickup=pickup)
    assignment = Assignment(
        note=event(pitch=72, start=250.13, end=258.79),
        part="solo_cornet",
        confidence=0.5,
        reason="test",
    )
    score, mapping = build_score([assignment], parts, r, 258.795125)
    assert all(p.getElementsByClass("Measure")[-1].highestTime == 4 for p in score.parts)
    export_score(score, tmp_path)
    parsed = converter.parse(tmp_path / "full_score.musicxml").toSoundingPitch()
    assert all(n.pitch.midi == 72 for n in parsed.recurse().notes)
    midi = read_midi(tmp_path / "full_score.mid")
    assert len(midi) == 1
    onset, end = quantize(assignment.note.start, assignment.note.end, r)
    assert midi[0].start == pytest.approx(onset * 60 / r.tempo, abs=0.005)
    assert midi[0].end == pytest.approx(end * 60 / r.tempo, abs=0.005)


def test_resume_uses_saved_notes_and_settings_without_inference(tmp_path, monkeypatch):
    import json

    import numpy as np
    import soundfile as sf
    from typer.testing import CliRunner

    from brass_transcriber.backend import YourMT3Backend
    from brass_transcriber.cli import app, save_run_settings
    from brass_transcriber.models import TranscriptionResult, write_midi

    def unexpected_inference(*args, **kwargs):
        pytest.fail("Resume must not run or load inference")

    monkeypatch.setattr(YourMT3Backend, "transcribe", unexpected_inference)
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    r = Rhythm(tempo=123, detected_tempo=None, beats=[], time_signature="3/4", grid="1/8")
    n = event(pitch=72, start=5, end=5.6)
    result = TranscriptionResult(notes=[n], model="cached model", device="cpu", runtime=9)
    (intermediate / "model_notes.json").write_text(result.model_dump_json())
    assigned = Assignment(note=n, part="solo_cornet", confidence=0.5, reason="cached")
    (intermediate / "assigned_voices.json").write_text(json.dumps([assigned.model_dump()]))
    write_midi([n], intermediate / "raw_transcription.mid", r.tempo)
    sf.write(intermediate / "normalized.wav", np.zeros(12345), 16000)
    save_run_settings(tmp_path, r, 12345 / 16000, 5, load_parts())
    before = (intermediate / "model_notes.json").read_bytes()
    response = CliRunner().invoke(app, ["resume", str(tmp_path), "--no-mscz"])
    assert response.exit_code == 0, response.output
    assert (intermediate / "model_notes.json").read_bytes() == before
    report = json.loads((tmp_path / "diagnostics/transcription_report.json").read_text())
    assert report["selected_model"] == "cached model"
    assert report["inference_runtime"] == 9
    assert report["excerpt_start"] == 5
    assert report["rhythm"]["time_signature"] == "3/4"
    assert report["rhythm"]["grid"] == "1/8"
    assert (tmp_path / "full_score.musicxml").exists()


def test_bass_staff_labels_have_unambiguous_instrument_names(tmp_path):
    import xml.etree.ElementTree as ET

    parts = [p for p in load_parts() if p.family == "bass"]
    score, _ = build_score([], parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 2)
    export_score(score, tmp_path)
    root = ET.parse(tmp_path / "full_score.musicxml")
    for xml_part, part in zip(root.findall(".//score-part"), parts, strict=True):
        assert xml_part.findtext("part-name") == part.name
        assert "Tuba" in xml_part.findtext("score-instrument/instrument-name")
        assert xml_part.findtext("score-instrument/instrument-sound") == "brass.tuba"
    # Cached settings and custom configurations predating instrument_name are safe too.
    for part in parts:
        part.instrument_name = None
    score, _ = build_score([], parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 2)
    export_score(score, tmp_path)
    assert all(
        p.findtext("score-instrument/instrument-name") == "Tuba"
        for p in ET.parse(tmp_path / "full_score.musicxml").findall(".//score-part")
    )

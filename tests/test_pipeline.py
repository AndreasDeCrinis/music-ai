from pathlib import Path

import pytest
from music21 import converter, interval

from brass_transcriber.backend import deduplicate
from brass_transcriber.models import Assignment, NoteEvent, load_parts, read_midi
from brass_transcriber.notation import build_score, export_score
from brass_transcriber.orchestration import assign_voices
from brass_transcriber.rhythm import Rhythm, quantize


def event(id="n", pitch=60, start=0, end=1, **kwargs):
    return NoteEvent(id=id, pitch=pitch, start=start, end=end, **kwargs)


@pytest.mark.parametrize("part", load_parts(), ids=lambda p: p.id)
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

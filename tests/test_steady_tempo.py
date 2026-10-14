import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from brass_transcriber.models import Assignment, EnsemblePreset, NoteEvent, load_parts, read_midi
from brass_transcriber.notation import build_score, export_score, musescore_export
from brass_transcriber.rhythm import Rhythm, quantize, regularize_tempo, tempo_map


def performance(bpms, meter="4/4", pickup=0):
    numerator, denominator = map(int, meter.split("/"))
    bar = numerator * 4 / denominator
    positions = np.arange(len(bpms) + 1) * bar
    times = np.r_[0, np.cumsum(60 * bar / np.array(bpms))]
    return Rhythm(
        tempo=float(np.median(bpms)),
        detected_tempo=float(np.median(bpms)),
        beats=times.tolist(),
        mode="beats",
        grid_times=times.tolist(),
        grid_positions=positions.tolist(),
        time_signature=meter,
        pickup=pickup,
    )


def test_small_variations_produce_one_integer_tempo_without_changing_note_grid():
    raw = performance([130 + 2 * np.sin(i) for i in range(64)])
    steady = regularize_tempo(raw)
    assert tempo_map(steady) == [(0, 130)]
    assert steady.grid_times == raw.grid_times
    assert steady.grid_positions == raw.grid_positions
    assert quantize(3.123, 3.7, steady) == quantize(3.123, 3.7, raw)
    assert regularize_tempo(steady) == steady


@pytest.mark.parametrize("meter,quarter", [("4/4", 64), ("3/4", 48), ("6/8", 48)])
def test_sixteen_bars_then_large_tempo_change_are_preserved(meter, quarter):
    steady = regularize_tempo(performance([130] * 16 + [150] * 16, meter))
    assert tempo_map(steady) == [(0, 130), (quarter, 150)]


def test_single_measure_outlier_is_smoothed_but_sustained_change_is_preserved():
    assert tempo_map(regularize_tempo(performance([130] * 10 + [180] + [130] * 10))) == [(0, 130)]
    assert tempo_map(regularize_tempo(performance([130] * 16 + [190] * 8 + [130] * 16))) == [
        (0, 130),
        (64, 190),
        (96, 130),
    ]


def test_tempo_sections_validation_and_short_recording():
    with pytest.raises(ValueError, match="start at zero"):
        Rhythm(tempo=130, detected_tempo=None, beats=[], tempo_sections=[dict(quarter=4, bpm=130)])
    assert tempo_map(regularize_tempo(performance([130.4]))) == [(0, 130)]


def test_all_instruments_have_same_readable_sections_and_native_playback(tmp_path):
    import mido

    rhythm = regularize_tempo(performance([130] * 16 + [150] * 16))
    parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
    assignments = [
        Assignment(
            note=NoteEvent(
                id=p.id,
                pitch=p.preferred_pitch,
                start=rhythm.grid_times[17],
                end=rhythm.grid_times[18],
            ),
            part=p.id,
            confidence=0.5,
            reason="fixture",
        )
        for p in parts
        if not p.percussion
    ]
    score, _ = build_score(assignments, parts, rhythm, rhythm.grid_times[-1])
    export_score(score, tmp_path)
    root = ET.parse(tmp_path / "full_score.musicxml").getroot()
    for p in root.findall("part"):
        assert [
            d.findtext("direction-type/metronome/per-minute")
            for d in p.findall(".//direction")
            if d.find("sound") is not None and d.find("sound").get("tempo")
        ] == ["130", "150"]
        assert all(d.get("print-object") != "no" for d in p.findall(".//direction"))
    for p in (tmp_path / "parts").glob("*.musicxml"):
        assert [s.get("tempo") for s in ET.parse(p).findall(".//sound") if s.get("tempo")] == [
            "130",
            "150",
        ]
        assert all(d.get("print-object") != "no" for d in ET.parse(p).findall(".//direction"))
    expected = read_midi(tmp_path / "full_score.mid")
    assert max(n.start for n in expected) - min(n.start for n in expected) < 1e-9
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
    midi = mido.MidiFile(tmp_path / "native.mid")
    tempos = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempos.append((tick / midi.ticks_per_beat, round(mido.tempo2bpm(msg.tempo))))
    assert tempos == [(0, 130), (64, 150)]
    actual = read_midi(tmp_path / "native.mid")
    assert Counter((n.pitch, round(n.start, 2)) for n in actual) == Counter(
        (n.pitch, round(n.start, 2)) for n in expected
    )
    (tmp_path / "verification.json").write_text(json.dumps(tempos))

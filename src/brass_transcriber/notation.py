import copy
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pretty_midi
from music21 import (
    chord,
    clef,
    converter,
    instrument,
    interval,
    metadata,
    meter,
    note,
    percussion,
    stream,
    tempo,
)

from .models import Assignment, PartConfig
from .rhythm import Rhythm, grid_quarters, quantize, seconds_to_quarters, tempo_map


def build_score(
    assignments: list[Assignment], parts: list[PartConfig], rhythm: Rhythm, total_seconds: float
) -> tuple[stream.Score, list[dict]]:
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Provisional brass band transcription")
    score.atSoundingPitch = False
    score.editorial["rhythm"] = rhythm.model_dump()
    mapping = []
    bar_length = float(meter.TimeSignature(rhythm.time_signature).barDuration.quarterLength)
    padding = float(bar_length - rhythm.pickup) if rhythm.pickup else 0
    # Audio ends at an arbitrary sample, not a notatable rhythmic duration.
    # Pad with rests to a common complete bar, including any note whose
    # quantized offset extends beyond the audio. Never alter detected notes
    # simply to align the recording's endpoint with the notation grid.
    latest_note = max(
        (quantize(a.note.start, a.note.end, rhythm)[1] for a in assignments if a.part),
        default=0,
    )
    required_quarters = max(seconds_to_quarters(total_seconds, rhythm), latest_note, 1) + padding
    total_quarters = math.ceil(required_quarters / bar_length) * bar_length
    pitched_parts = [p for p in parts if not p.percussion]
    available_channels = [c for c in range(16) if c != 9]
    programs = sorted({p.program for p in pitched_parts})
    for config in parts:
        part = stream.Part(id=config.id)
        part.partName = config.name
        inst = instrument.Instrument()
        # A desk label (especially "Bass") is ambiguous to MuseScore's template
        # matching. Keep it as the staff name, but identify the actual instrument
        # separately so tubas cannot be imported as vocal basses.
        inst.instrumentName = config.instrument_name or (
            "Tuba" if config.family == "bass" else config.name
        )
        inst.partName = config.name
        inst.midiProgram = config.program
        inst.midiChannel = (
            9
            if config.percussion
            else available_channels[
                pitched_parts.index(config)
                if len(pitched_parts) <= 15
                else programs.index(config.program)
            ]
        )
        inst.editorial["instrument_sound"] = config.instrument_sound
        inst.editorial["percussion"] = config.percussion
        inst.transposition = interval.Interval(config.written_to_sounding)
        part.insert(0, inst)
        part.insert(
            0,
            clef.PercussionClef()
            if config.percussion
            else clef.BassClef()
            if config.clef == "bass"
            else clef.TrebleClef(),
        )
        part.insert(0, meter.TimeSignature(rhythm.time_signature))
        for tempo_index, (position, bpm) in enumerate(tempo_map(rhythm)):
            mark = tempo.MetronomeMark(number=bpm)
            if tempo_index:
                mark.style.hideObjectOnPrint = True
            part.insert(position + (padding if position else 0), mark)
        ends = [0.0] * config.max_simultaneous
        voices = [stream.Voice(id=i + 1) for i in range(config.max_simultaneous)]
        selected = sorted(
            (a for a in assignments if a.part == config.id),
            key=lambda a: (a.note.start, a.note.pitch),
        )
        if config.percussion:
            voices = [stream.Voice(id=1)]
            drum_instruments = {}
            attacks = {}
            step = grid_quarters(rhythm.grid)
            for a in selected:
                onset, _ = quantize(a.note.start, a.note.end, rhythm)
                gm_pitch = a.note.pitch
                if gm_pitch not in drum_instruments:
                    drum = instrument.UnpitchedPercussion()
                    drum.instrumentName = pretty_midi.note_number_to_drum_name(gm_pitch)
                    drum.percMapPitch = gm_pitch
                    drum.midiChannel = 9
                    drum_instruments[gm_pitch] = drum
                position, head = drum_display(gm_pitch)
                hit = note.Unpitched(position)
                hit.storedInstrument = drum_instruments[gm_pitch]
                hit.notehead = head
                hit.id = a.note.id
                hit.volume.velocity = a.note.velocity
                attacks.setdefault(onset + padding, []).append(hit)
                mapping.append(
                    dict(
                        source_id=a.note.id,
                        source_ids=a.note.provenance or [a.note.id],
                        part=config.id,
                        notation_voice=1,
                        percussion=True,
                        sounding_pitch=gm_pitch,
                        written_pitch=None,
                        onset_quarters=onset,
                        duration_quarters=step,
                        confidence=a.confidence,
                        percussion_duration_policy="one grid step; raw model offsets preserved in intermediate files",
                    )
                )
            for onset, hits in sorted(attacks.items()):
                event = percussion.PercussionChord(hits) if len(hits) > 1 else hits[0]
                event.duration.quarterLength = step
                voices[0].insert(onset, event)
        for a in selected:
            if config.percussion:
                break
            onset, offset = quantize(a.note.start, a.note.end, rhythm)
            onset += padding
            offset += padding
            lane = next((i for i, end in enumerate(ends) if end <= onset + 1e-6), None)
            if lane is None:
                # Sub-grid overlap from quantization: shorten an earlier note at this attack.
                lane = min(range(len(ends)), key=lambda i: ends[i])
                previous = list(voices[lane].recurse().notes)[-1]
                if onset > previous.offset:
                    previous.duration.quarterLength = onset - previous.offset
                    for mapped in reversed(mapping):
                        if mapped["source_id"] == str(previous.id):
                            mapped["duration_quarters"] = float(previous.duration.quarterLength)
                            mapped["quantization_adjustment"] = "shortened to avoid overlap"
                            break
                else:
                    # Preserve distinct attacks that collapse onto a grid point in an extra
                    # notation voice; report this separately as a quantization collision.
                    voices.append(stream.Voice(id=len(voices) + 1))
                    ends.append(0.0)
                    lane = len(voices) - 1
            n = note.Note(a.note.pitch)
            n.transpose(inst.transposition.reverse(), inPlace=True)
            n.duration.quarterLength = offset - onset
            n.volume.velocity = a.note.velocity
            n.id = a.note.id
            voices[lane].insert(onset, n)
            ends[lane] = offset
            mapping.append(
                dict(
                    source_id=a.note.id,
                    notation_voice=lane + 1,
                    source_ids=a.note.provenance or [a.note.id],
                    part=config.id,
                    sounding_pitch=a.note.pitch,
                    written_pitch=n.pitch.midi,
                    onset_quarters=onset - padding,
                    duration_quarters=offset - onset,
                    confidence=a.confidence,
                )
            )
        for v in voices:
            if len(v):
                v.makeRests(
                    inPlace=True,
                    fillGaps=True,
                    refStreamOrTimeRange=(0, max(total_quarters, v.highestTime)),
                )
                part.insert(0, v)
        if not selected:
            part.insert(0, note.Rest(quarterLength=total_quarters))
        part.atSoundingPitch = False
        part.makeMeasures(inPlace=True)
        part.makeTies(inPlace=True)
        if padding:
            first = part.measure(1)
            containers = list(first.getElementsByClass(stream.Voice)) or [first]
            for container in containers:
                for element in list(container.notesAndRests):
                    old_offset = float(element.offset)
                    end = old_offset + float(element.duration.quarterLength)
                    if end <= padding + 1e-8:
                        container.remove(element)
                    else:
                        if old_offset < padding:
                            element.duration.quarterLength = end - padding
                        container.setElementOffset(element, max(0, old_offset - padding))
            first.paddingLeft = padding
            first.number = 0
            for mark in first.getElementsByClass(tempo.MetronomeMark):
                if mark.offset >= padding:
                    first.setElementOffset(mark, float(mark.offset) - padding)
            for measure in list(part.getElementsByClass(stream.Measure))[1:]:
                part.setElementOffset(measure, float(measure.offset) - padding)
                measure.number -= 1
        # makeMeasures renumbers numeric Voice IDs from zero; MuseScore requires
        # positive MusicXML voice numbers. Normalize after measure/tie processing.
        for measure in part.getElementsByClass(stream.Measure):
            for voice_index, voice in enumerate(measure.getElementsByClass(stream.Voice), 1):
                voice.id = str(voice_index)
        score.insert(0, part)
    return score, mapping


def drum_display(gm_pitch: int) -> tuple[str, str]:
    """Conventional five-line kit positions; unidentified GM percussion is shown above."""
    if gm_pitch in {35, 36}:
        return "F4", "normal"
    if gm_pitch in {37, 38, 39, 40}:
        return "C5", "normal"
    if gm_pitch in {42, 44, 46}:
        return "G5", "x"
    if gm_pitch in {49, 52, 55, 57}:
        return "A5", "x"
    if gm_pitch in {51, 53, 59}:
        return "F5", "x"
    if gm_pitch in {41, 43}:
        return "G4", "normal"
    if gm_pitch in {45, 47}:
        return "D5", "normal"
    if gm_pitch in {48, 50}:
        return "E5", "normal"
    return "E5", "normal"


def export_score(score: stream.Score, output: Path) -> None:
    full = output / "full_score.musicxml"
    score.write("musicxml", fp=full)
    enrich_xml(full, score)
    parsed = converter.parse(full)
    if len(parsed.parts) != len(score.parts):
        raise RuntimeError("MusicXML validation lost instrumental parts")
    write_score_midi(score, output / "full_score.mid")
    (output / "parts").mkdir(exist_ok=True)
    for part in score.parts:
        extracted = stream.Score()
        extracted.atSoundingPitch = False
        extracted.insert(0, copy.deepcopy(part))
        path = output / "parts" / f"{part.id}.musicxml"
        extracted.write("musicxml", fp=path)
        enrich_xml(path, extracted)
        converter.parse(path)


def musescore_export(output: Path) -> str | None:
    binary = shutil.which("mscore") or shutil.which("musescore")
    mac = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if not binary and mac.exists():
        binary = str(mac)
    if not binary:
        return "MuseScore executable unavailable; native export skipped"
    try:
        result = subprocess.run(
            [
                binary,
                "--sound-profile",
                "MuseScore Basic",
                "-o",
                str((output / "full_score.mscz").resolve()),
                str((output / "full_score.musicxml").resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode or not (output / "full_score.mscz").exists():
            return f"MuseScore conversion failed: {result.stderr[-1000:]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"MuseScore conversion unavailable: {exc}"
    return None


def enrich_xml(path: Path, score: stream.Score) -> None:
    """Add instrument-sound metadata music21 omits and reuse GM program channels."""
    root = ET.parse(path).getroot()
    channels = {}
    separate_desks = (
        sum(
            not next(
                iter(p.recurse().getElementsByClass(instrument.Instrument)), p.getInstrument()
            ).editorial.get("percussion", False)
            for p in score.parts
        )
        <= 15
    )
    available_channels = [c for c in range(1, 17) if c != 10]
    for xml_part, part in zip(root.findall("part-list/score-part"), score.parts, strict=True):
        inst = next(
            iter(part.recurse().getElementsByClass(instrument.Instrument)), part.getInstrument()
        )
        xml_inst = xml_part.find("score-instrument")
        if xml_inst is not None:
            ET.SubElement(xml_inst, "instrument-sound").text = inst.editorial.get(
                "instrument_sound", "brass"
            )
        if inst.editorial.get("percussion", False):
            enrich_percussion(xml_part, root.find(f"part[@id='{xml_part.get('id')}']"), part)
        for midi in xml_part.findall("midi-instrument"):
            if inst.editorial.get("percussion", False):
                channel = midi.find("midi-channel")
                if channel is not None:
                    channel.text = "10"
                continue
            program = inst.midiProgram or 0
            channel_key = part.id if separate_desks else program
            if channel_key not in channels:
                channels[channel_key] = available_channels[len(channels)]
            midi.find("midi-channel").text = str(channels[channel_key])
    # Tie fragments copy the source ID; XML IDs must be unique document-wide.
    seen = {}
    for xml_note in root.findall(".//note"):
        source_id = xml_note.get("id")
        if source_id:
            seen[source_id] = seen.get(source_id, 0) + 1
            xml_note.set("id", f"{source_id}-segment-{seen[source_id]}")
    normalize_tempo_directions(root)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def normalize_tempo_directions(root: ET.Element) -> None:
    """Anchor tempo directions with explicit clock advances rather than offsets.

    MuseScore loses some offset-based changes inside long rests. A sequence of
    forward/direction elements followed by backup keeps all changes and leaves
    the original notes, rests and voice clocks unchanged.
    """
    for part in root.findall("part"):
        divisions = 10080
        first = True
        for measure in part.findall("measure"):
            divisions = int(measure.findtext("attributes/divisions", str(divisions)))
            cursor = 0
            directions = []
            for element in list(measure):
                sound = element.find("sound")
                if element.tag == "direction" and sound is not None and sound.get("tempo"):
                    position = cursor + int(element.findtext("offset", "0"))
                    offset = element.find("offset")
                    if offset is not None:
                        element.remove(offset)
                    if not first:
                        element.set("print-object", "no")
                    first = False
                    directions.append((position, element))
                    measure.remove(element)
                elif element.tag in {"note", "forward"} and element.find("chord") is None:
                    cursor += int(element.findtext("duration", "0"))
                elif element.tag == "backup":
                    cursor -= int(element.findtext("duration", "0"))
            insertion = next(
                (i for i, e in enumerate(measure) if e.tag not in {"attributes", "print"}),
                len(measure),
            )
            previous = 0
            for position, direction in sorted(directions, key=lambda item: item[0]):
                if position < 0:
                    raise ValueError("Tempo direction precedes its measure")
                if position > previous:
                    forward = ET.Element("forward")
                    ET.SubElement(forward, "duration").text = str(position - previous)
                    measure.insert(insertion, forward)
                    insertion += 1
                measure.insert(insertion, direction)
                insertion += 1
                previous = position
            if previous:
                backup = ET.Element("backup")
                ET.SubElement(backup, "duration").text = str(previous)
                measure.insert(insertion, backup)


def enrich_percussion(xml_part: ET.Element, xml_music: ET.Element, part: stream.Part) -> None:
    """music21 omits per-hit storedInstrument references in percussion chords.
    Explicitly declare and reference each GM drum without guessing from staff position.
    """
    hits = []
    for event in part.recurse().notes:
        hits.extend(event.notes if isinstance(event, chord.ChordBase) else [event])
    xml_hits = [n for n in xml_music.findall(".//note") if n.find("unpitched") is not None]
    if len(hits) != len(xml_hits):
        raise RuntimeError("Percussion export changed hit count; cannot safely map GM drums")
    instruments = {}
    for hit in hits:
        gm_pitch = hit.storedInstrument.percMapPitch
        instruments[gm_pitch] = f"{xml_part.get('id')}-drum-{gm_pitch}"
    # Keep every score-instrument before every midi-instrument (MusicXML schema order).
    insertion = next(
        (i for i, el in enumerate(xml_part) if el.tag == "midi-instrument"), len(xml_part)
    )
    for gm_pitch, identifier in sorted(instruments.items()):
        score_inst = ET.Element("score-instrument", id=identifier)
        ET.SubElement(score_inst, "instrument-name").text = pretty_midi.note_number_to_drum_name(
            gm_pitch
        )
        xml_part.insert(insertion, score_inst)
        insertion += 1
        midi_inst = ET.SubElement(xml_part, "midi-instrument", id=identifier)
        ET.SubElement(midi_inst, "midi-channel").text = "10"
        ET.SubElement(midi_inst, "midi-unpitched").text = str(gm_pitch + 1)
    for hit, xml_hit in zip(hits, xml_hits, strict=True):
        for old in list(xml_hit.findall("instrument")):
            xml_hit.remove(old)
        reference = ET.Element("instrument", id=instruments[hit.storedInstrument.percMapPitch])
        duration_index = list(xml_hit).index(xml_hit.find("duration"))
        xml_hit.insert(duration_index + 1, reference)
        xml_hit.set("id", str(hit.id))
        xml_hit.set("dynamics", str(round((hit.volume.velocity or 80) * 100 / 90, 2)))


def write_score_midi(score: stream.Score, path: Path) -> None:
    """Write quarter-note ticks directly, with a faithful variable-tempo map.

    Merge tie fragments without onset drift; keep independent desks and GM drums.
    """
    import mido

    sounding = score.toSoundingPitch()
    marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
    bpm = float(marks[0].number) if marks else 120
    rhythm_data = score.editorial.get("rhythm")
    rhythm = (
        Rhythm.model_validate(rhythm_data)
        if rhythm_data
        else Rhythm(tempo=bpm, detected_tempo=None, beats=[])
    )
    ticks = 960
    midi = mido.MidiFile(ticks_per_beat=ticks)
    conductor = mido.MidiTrack()
    midi.tracks.append(conductor)
    signature = rhythm.time_signature.split("/")
    conductor.append(
        mido.MetaMessage(
            "time_signature", numerator=int(signature[0]), denominator=int(signature[1]), time=0
        )
    )
    previous_tick = 0
    for quarter, local_bpm in tempo_map(rhythm):
        tick = max(0, round(quarter * ticks))
        conductor.append(
            mido.MetaMessage(
                "set_tempo", tempo=mido.bpm2tempo(local_bpm), time=tick - previous_tick
            )
        )
        previous_tick = tick
    # getInstrument does not always recurse into the first measure.
    instruments = [
        next(iter(p.recurse().getElementsByClass(instrument.Instrument)), p.getInstrument())
        for p in sounding.parts
    ]
    programs = sorted(
        {i.midiProgram or 0 for i in instruments if not i.editorial.get("percussion", False)}
    )
    channels = [c for c in range(16) if c != 9]
    if len(programs) > len(channels):
        raise ValueError("Too many distinct GM programs for a single-port MIDI file")
    pitched_indices = [
        i for i, inst in enumerate(instruments) if not inst.editorial.get("percussion", False)
    ]
    for part_index, (part, inst) in enumerate(zip(sounding.parts, instruments, strict=True)):
        program = inst.midiProgram or 0
        drum_part = bool(inst.editorial.get("percussion", False))
        channel = (
            9
            if drum_part
            else channels[
                pitched_indices.index(part_index)
                if len(pitched_indices) <= 15
                else programs.index(program)
            ]
        )
        notes = []
        active = {}
        for event in sorted(part.recurse().notes, key=lambda n: n.getOffsetInHierarchy(part)):
            onset = float(event.getOffsetInHierarchy(part))
            offset = onset + float(event.quarterLength)
            components = event.notes if isinstance(event, chord.ChordBase) else [event]
            for n in components:
                is_drum = isinstance(n, note.Unpitched)
                pitch_number = n.storedInstrument.percMapPitch if is_drum else n.pitch.midi
                key = (str(n.id), pitch_number)
                if n.tie and n.tie.type in {"continue", "stop"} and key in active:
                    active[key]["end"] = offset
                    continue
                item = dict(
                    pitch=pitch_number, start=onset, end=offset, velocity=n.volume.velocity or 80
                )
                notes.append(item)
                active[key] = item
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(
            mido.MetaMessage(
                "track_name",
                name=(part.partName or inst.instrumentName or "Part")
                .replace("♭", "b")
                .replace("♯", "#")
                .encode("latin-1", "replace")
                .decode("latin-1"),
                time=0,
            )
        )
        track.append(mido.Message("program_change", channel=channel, program=program, time=0))
        events = []
        for item in notes:
            events.append((round(item["start"] * ticks), 1, item["pitch"], item["velocity"]))
            events.append((round(item["end"] * ticks), 0, item["pitch"], 0))
        previous_tick = 0
        for tick, on, pitch_number, velocity in sorted(events):
            track.append(
                mido.Message(
                    "note_on" if on else "note_off",
                    channel=channel,
                    note=pitch_number,
                    velocity=velocity,
                    time=max(0, tick - previous_tick),
                )
            )
            previous_tick = tick
    midi.save(path)

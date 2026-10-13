import copy
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from music21 import clef, converter, instrument, interval, metadata, meter, note, stream, tempo

from .models import Assignment, NoteEvent, PartConfig, write_midi
from .rhythm import Rhythm, quantize


def build_score(
    assignments: list[Assignment], parts: list[PartConfig], rhythm: Rhythm, total_seconds: float
) -> tuple[stream.Score, list[dict]]:
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Provisional brass band transcription")
    score.atSoundingPitch = False
    mapping = []
    total_quarters = max(total_seconds * rhythm.tempo / 60, 1)
    bar_length = meter.TimeSignature(rhythm.time_signature).barDuration.quarterLength
    padding = float(bar_length - rhythm.pickup) if rhythm.pickup else 0
    total_quarters += padding
    for config in parts:
        part = stream.Part(id=config.id)
        part.partName = config.name
        inst = instrument.Instrument()
        inst.instrumentName = config.name
        inst.partName = config.name
        inst.midiProgram = config.program
        programs = sorted({p.program for p in parts})
        inst.midiChannel = programs.index(config.program)
        inst.editorial["instrument_sound"] = config.instrument_sound
        inst.transposition = interval.Interval(config.written_to_sounding)
        part.insert(0, inst)
        part.insert(0, clef.BassClef() if config.clef == "bass" else clef.TrebleClef())
        part.insert(0, meter.TimeSignature(rhythm.time_signature))
        part.insert(0, tempo.MetronomeMark(number=rhythm.tempo))
        ends = [0.0] * config.max_simultaneous
        voices = [stream.Voice(id=i + 1) for i in range(config.max_simultaneous)]
        selected = sorted(
            (a for a in assignments if a.part == config.id),
            key=lambda a: (a.note.start, a.note.pitch),
        )
        for a in selected:
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
    for xml_part, part in zip(root.findall("part-list/score-part"), score.parts, strict=True):
        inst = next(
            iter(part.recurse().getElementsByClass(instrument.Instrument)), part.getInstrument()
        )
        xml_inst = xml_part.find("score-instrument")
        if xml_inst is not None:
            ET.SubElement(xml_inst, "instrument-sound").text = inst.editorial.get(
                "instrument_sound", "brass"
            )
        midi = xml_part.find("midi-instrument")
        if midi is not None:
            program = inst.midiProgram or 0
            if program not in channels:
                channels[program] = len(channels) + 1
            midi.find("midi-channel").text = str(channels[program])
    # Tie fragments copy the source ID; XML IDs must be unique document-wide.
    seen = {}
    for xml_note in root.findall(".//note"):
        source_id = xml_note.get("id")
        if source_id:
            seen[source_id] = seen.get(source_id, 0) + 1
            xml_note.set("id", f"{source_id}-segment-{seen[source_id]}")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_score_midi(score: stream.Score, path: Path) -> None:
    """Export concert pitches directly, merging tie segments without music21's
    multi-voice MIDI tie-onset drift. Same-program desks share a GM channel in independent tracks.
    """
    sounding = score.toSoundingPitch()
    marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
    bpm = float(marks[0].number) if marks else 120
    events = []
    for part_index, part in enumerate(sounding.parts):
        inst = next(
            iter(part.recurse().getElementsByClass(instrument.Instrument)), part.getInstrument()
        )
        program = inst.midiProgram or 0
        active = {}
        for n in sorted(part.recurse().notes, key=lambda n: n.getOffsetInHierarchy(part)):
            onset = float(n.getOffsetInHierarchy(part)) * 60 / bpm
            offset = onset + float(n.quarterLength) * 60 / bpm
            key = (str(n.id), n.pitch.midi)
            if n.tie and n.tie.type in {"continue", "stop"} and key in active:
                active[key].end = offset
            else:
                ev = NoteEvent(
                    id=f"score-{part_index}-{n.id}",
                    pitch=n.pitch.midi,
                    start=onset,
                    end=offset,
                    velocity=n.volume.velocity or 80,
                    program=program,
                    track=part_index,
                )
                events.append(ev)
                active[key] = ev
    write_midi(events, path, bpm)
    # GM has 15 pitched channels but 17 score desks. Reuse channels only among
    # equal programs, keeping separate MIDI tracks (and unison note identities).
    import mido

    midi = mido.MidiFile(path)
    signatures = list(score.recurse().getElementsByClass(meter.TimeSignature))
    if signatures:
        signature = signatures[0]
        midi.tracks[0].insert(
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=signature.numerator,
                denominator=signature.denominator,
                time=0,
            ),
        )
    programs = sorted({event.program for event in events})
    channels = [c for c in range(16) if c != 9]
    for track in midi.tracks:
        program = next((m.program for m in track if m.type == "program_change"), None)
        if program is not None:
            channel = channels[programs.index(program)]
            for message in track:
                if hasattr(message, "channel"):
                    message.channel = channel
    midi.save(path)

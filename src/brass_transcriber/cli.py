import hashlib
import json
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

import typer

from .audio import preprocess
from .backend import CHECKPOINT_NAME, YourMT3Backend
from .diagnostics import piano_roll, report
from .models import Assignment, TranscriptionResult, load_parts, read_midi, write_midi
from .notation import build_score, export_score, musescore_export
from .orchestration import assign_voices
from .rhythm import analyze

app = typer.Typer(
    no_args_is_help=True, help="Local YourMT3+ to provisional British brass band score"
)
CACHE = Path(".cache/models")
SHA256 = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"


@app.command()
def download_model(cache: Path = CACHE):
    """Explicitly download and verify the ~536 MB YourMT3+ checkpoint."""
    target = cache / "yourmt3" / CHECKPOINT_NAME / "last.ckpt"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/{CHECKPOINT_NAME}/checkpoints/last.ckpt"
    partial = target.with_suffix(".partial")
    if not target.exists():
        typer.echo("Downloading ~536 MB checkpoint locally; audio is never uploaded.")
        urllib.request.urlretrieve(url, partial)
        check = partial
    else:
        check = target
    with check.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != SHA256:
        raise typer.BadParameter(f"Checkpoint checksum mismatch: {digest}")
    if check == partial:
        partial.replace(target)
    typer.echo(f"Verified checkpoint: {target.resolve()}")


@app.command()
def doctor():
    """Print local environment and inference support."""
    import torch

    from .audio import ffmpeg_path

    typer.echo(
        json.dumps(
            dict(
                python=sys.version,
                torch=torch.__version__,
                mps_available=torch.backends.mps.is_available(),
                selected_backend_mps_supported=False,
                selected_device="cpu",
                ffmpeg=ffmpeg_path(),
                chip=subprocess.getoutput("sysctl -n machdep.cpu.brand_string"),
                memory_bytes=subprocess.getoutput("sysctl -n hw.memsize"),
            ),
            indent=2,
        )
    )


@app.command()
def transcribe(
    input: Path,
    output: Path = Path("output"),
    midi_only: bool = False,
    start: float = typer.Option(0, min=0),
    duration: float | None = typer.Option(None, min=0.01),
    tempo: float | None = typer.Option(None, min=20, max=400),
    time_signature: str | None = None,
    quantize: str = "1/16",
    pickup: float = typer.Option(0, min=0, help="Pickup length in quarter notes"),
    normalize: bool = False,
    chunk_seconds: float = typer.Option(30, min=1),
    overlap: float = typer.Option(2, min=0),
    device: str = "auto",
    cache: Path = CACHE,
    config: Path | None = None,
    from_midi: Path | None = typer.Option(
        None, help="Reuse real or fixture MIDI; bypasses audio inference explicitly"
    ),
    mscz: bool = typer.Option(True, "--mscz/--no-mscz"),
    separate: bool = typer.Option(
        False, help="Reserved experimental separation; currently unsupported"
    ),
):
    """Transcribe MP3/WAV/FLAC and export editable MusicXML."""
    if separate:
        raise typer.BadParameter(
            "Source separation is not implemented; brass stems are not reliably isolated"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if (output / "diagnostics/transcription_report.json").exists():
        raise typer.BadParameter("Output already contains a run; choose a fresh --output directory")
    for directory in (output, output / "intermediate", output / "diagnostics"):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        audio = output / "intermediate/normalized.wav"
        length = preprocess(input, audio, start, duration, normalize)
        rhythm = analyze(audio, tempo, time_signature, quantize, start, pickup)
        if from_midi:
            result = TranscriptionResult(
                notes=read_midi(from_midi), model="MIDI import (no audio inference)", device="none"
            )
        else:
            backend = YourMT3Backend(
                cache, output / "intermediate/model_chunks", start, chunk_seconds, overlap, device
            )
            result = backend.transcribe(audio)
        write_midi(result.notes, output / "intermediate/raw_transcription.mid", rhythm.tempo)
        (output / "intermediate/model_notes.json").write_text(result.model_dump_json(indent=2))
        parts = load_parts(config)
        warnings = []
        assignments = [] if midi_only else assign_voices(result.notes, parts)
        (output / "intermediate/assigned_voices.json").write_text(
            json.dumps([a.model_dump() for a in assignments], indent=2)
        )
        if midi_only:
            warnings.append("Orchestration and notation skipped by --midi-only.")
            write_midi(result.notes, output / "full_score.mid", rhythm.tempo)
        else:
            score, mapping = build_score(assignments, parts, rhythm, start + length)
            export_score(score, output)
            adjustments = [m for m in mapping if m.get("quantization_adjustment")]
            extra_voices = [
                m
                for m in mapping
                if m["notation_voice"]
                > next(p.max_simultaneous for p in parts if p.id == m["part"])
            ]
            if adjustments:
                warnings.append(
                    f"Quantization shortened {len(adjustments)} notes to avoid overlaps; see provenance."
                )
            if extra_voices:
                warnings.append(
                    f"Quantization collapsed attacks into {len(extra_voices)} extra notation voices; see provenance."
                )
            (output / "intermediate/score_note_provenance.json").write_text(
                json.dumps(mapping, indent=2)
            )
            if mscz:
                warning = musescore_export(output)
                if warning:
                    warnings.append(warning)
        piano_roll(result.notes, output / "diagnostics/piano_roll.png")
        report(result, assignments, parts, rhythm, length, start, output, warnings)
        typer.echo(
            f"Generated {len(result.notes)} notes using {result.model} on {result.device}: {output.resolve()}"
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def diagnose(output: Path):
    """Compare raw transcription with assigned and quantized score notes."""
    data = json.loads((output / "diagnostics/transcription_report.json").read_text())
    raw = read_midi(output / "intermediate/raw_transcription.mid")
    assignments = [
        Assignment.model_validate(a)
        for a in json.loads((output / "intermediate/assigned_voices.json").read_text())
    ]
    assigned = [a for a in assignments if a.part]
    data["comparison"] = dict(
        raw_notes=len(raw),
        assigned_source_notes=len(assigned),
        unassigned_source_notes=len(assignments) - len(assigned),
        invented_source_notes=len(
            {a.note.id for a in assigned}
            - {
                n.id
                for n in TranscriptionResult.model_validate_json(
                    (output / "intermediate/model_notes.json").read_text()
                ).notes
            }
        ),
        source_pitch_histogram={
            str(p): sum(n.pitch == p for n in raw) for p in sorted({n.pitch for n in raw})
        },
    )
    mapping = output / "intermediate/score_note_provenance.json"
    if mapping.exists():
        mapped = json.loads(mapping.read_text())
        data["comparison"]["score_events_before_ties"] = len(mapped)
        score_notes = read_midi(output / "full_score.mid")
        data["comparison"]["score_midi_notes"] = len(score_notes)
        data["comparison"]["assigned_pitch_histogram"] = {
            str(p): sum(a.note.pitch == p for a in assigned)
            for p in sorted({a.note.pitch for a in assigned})
        }
        data["comparison"]["score_pitch_histogram"] = {
            str(p): sum(n.pitch == p for n in score_notes)
            for p in sorted({n.pitch for n in score_notes})
        }
        data["comparison"]["pitch_counts_preserved"] = (
            data["comparison"]["assigned_pitch_histogram"]
            == data["comparison"]["score_pitch_histogram"]
        )
        bpm = data["rhythm"]["tempo"]
        original = {a.note.id: a.note for a in assigned}
        data["comparison"]["max_onset_quantization_shift_seconds"] = max(
            (abs(m["onset_quarters"] * 60 / bpm - original[m["source_id"]].start) for m in mapped),
            default=0,
        )
        data["comparison"]["quantization_adjustments"] = [
            m["source_id"] for m in mapped if m.get("quantization_adjustment")
        ]
    typer.echo(json.dumps(data, indent=2))


def main():
    # Preserve requested bare-file interface while retaining diagnostic subcommands.
    commands = {"transcribe", "diagnose", "doctor", "download-model"}
    if len(sys.argv) > 1 and sys.argv[1] not in commands and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "transcribe")
    app()

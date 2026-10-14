import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pretty_midi
import soundfile as sf
import typer
from music21.exceptions21 import Music21Exception

from .audio import preprocess
from .backend import CHECKPOINT_NAME, PS_CHECKPOINT_NAME, PS_SHA256, YourMT3Backend
from .diagnostics import piano_roll, report
from .models import (
    Assignment,
    EnsembleConfig,
    EnsemblePreset,
    NoteEvent,
    PartConfig,
    TranscriptionResult,
    load_ensemble,
    load_parts,
    read_midi,
    write_midi,
)
from .notation import build_score, export_score, musescore_export
from .orchestration import assign_voices
from .rhythm import Rhythm, analyze, quarters_to_seconds

app = typer.Typer(
    no_args_is_help=True, help="Local YourMT3+ to provisional brass/wind ensemble score"
)
CACHE = Path(".cache/models")
SHA256 = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"


@app.command()
def download_model(cache: Path = CACHE, variant: str = "nops"):
    """Download and verify nops, ps, or beats weights explicitly."""
    target = cache / "yourmt3" / CHECKPOINT_NAME / "last.ckpt"
    url = f"https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/{CHECKPOINT_NAME}/checkpoints/last.ckpt"
    expected, size = SHA256, "536 MB"
    if variant == "ps":
        target = cache / "yourmt3" / PS_CHECKPOINT_NAME / "model.ckpt"
        url = f"https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/{PS_CHECKPOINT_NAME}/checkpoints/model.ckpt"
        expected, size = PS_SHA256, "724 MB"
    elif variant == "beats":
        target = cache / "beat_this/final0.ckpt"
        url = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
        expected = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
        size = "78 MB"
    elif variant != "nops":
        raise typer.BadParameter("Variant must be nops, ps or beats")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    if not target.exists():
        typer.echo(f"Downloading ~{size} checkpoint locally; audio is never uploaded.")
        urllib.request.urlretrieve(url, partial)
        check = partial
    else:
        check = target
    with check.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected:
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
    ensemble: EnsemblePreset | None = typer.Option(None, help="Built-in instrumentation preset"),
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
    if config is not None and ensemble is not None:
        raise typer.BadParameter("Choose --ensemble or --config, not both")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if (output / "diagnostics/transcription_report.json").exists():
        raise typer.BadParameter("Output already contains a run; choose a fresh --output directory")
    for directory in (output, output / "intermediate", output / "diagnostics"):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        audio = output / "intermediate/normalized.wav"
        length = preprocess(input, audio, start, duration, normalize)
        rhythm = analyze(audio, tempo, time_signature, quantize, start, pickup)
        selected_ensemble = load_ensemble(config, ensemble or EnsemblePreset.BRITISH)
        parts = selected_ensemble.parts
        save_run_settings(output, rhythm, length, start, parts, selected_ensemble)
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
        warnings = list(selected_ensemble.assumptions)
        assignments = [] if midi_only else assign_voices(result.notes, parts)
        (output / "intermediate/assigned_voices.json").write_text(
            json.dumps([a.model_dump() for a in assignments], indent=2)
        )
        finish_run(
            result, assignments, parts, rhythm, length, start, output, midi_only, mscz, warnings
        )
    except (ValueError, RuntimeError, FileNotFoundError, Music21Exception) as exc:
        raise typer.BadParameter(
            f"{exc}. Saved transcription can be recovered with brass-transcribe resume {output}"
        ) from exc


def save_run_settings(
    output: Path,
    rhythm: Rhythm,
    duration: float,
    start: float,
    parts: list[PartConfig],
    ensemble: EnsembleConfig | None = None,
) -> None:
    settings = dict(
        rhythm=rhythm.model_dump(),
        audio_duration=duration,
        excerpt_start=start,
        parts=[p.model_dump() for p in parts],
        ensemble=ensemble.model_dump(exclude={"parts"}) if ensemble else None,
    )
    (output / "intermediate/run_settings.json").write_text(json.dumps(settings, indent=2))


def finish_run(
    result: TranscriptionResult,
    assignments: list[Assignment],
    parts: list[PartConfig],
    rhythm: Rhythm,
    length: float,
    start: float,
    output: Path,
    midi_only: bool,
    mscz: bool,
    warnings: list[str],
) -> None:
    if midi_only:
        warnings.append("Orchestration and notation skipped by --midi-only.")
        write_midi(result.notes, output / "full_score.mid", rhythm.tempo)
    else:
        unknown = {a.part for a in assignments if a.part} - {p.id for p in parts}
        if unknown:
            raise ValueError(f"Saved assignments reference unknown parts: {sorted(unknown)}")
        score, mapping = build_score(assignments, parts, rhythm, start + length)
        settings_path = output / "intermediate/run_settings.json"
        if settings_path.exists():
            selected = json.loads(settings_path.read_text()).get("ensemble")
            if selected:
                score.metadata.title = f"Provisional {selected['name']} transcription"
        # Preserve provenance even if notation export fails.
        (output / "intermediate/score_note_provenance.json").write_text(
            json.dumps(mapping, indent=2)
        )
        export_score(score, output)
        adjustments = [m for m in mapping if m.get("quantization_adjustment")]
        extra_voices = [
            m
            for m in mapping
            if m["notation_voice"] > next(p.max_simultaneous for p in parts if p.id == m["part"])
        ]
        if adjustments:
            warnings.append(
                f"Quantization shortened {len(adjustments)} notes to avoid overlaps; see provenance."
            )
        if extra_voices:
            warnings.append(
                f"Quantization collapsed attacks into {len(extra_voices)} extra notation voices; see provenance."
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


@app.command()
def resume(
    output: Path,
    config: Path | None = typer.Option(None, help="Override saved instrumentation metadata"),
    ensemble: EnsemblePreset | None = typer.Option(
        None, help="Reassign cached notes to this preset"
    ),
    destination: Path | None = typer.Option(
        None, "--output", help="Export to a separate directory"
    ),
    tempo: float | None = typer.Option(None, min=20, max=400),
    time_signature: str | None = None,
    quantize: str | None = None,
    pickup: float | None = typer.Option(None, min=0),
    start: float | None = typer.Option(
        None, min=0, help="Excerpt start for legacy runs without saved settings"
    ),
    mscz: bool = typer.Option(True, "--mscz/--no-mscz"),
    regularize: bool = typer.Option(
        False,
        "--regularize-tempo",
        help="Use shared integer-BPM sections, smoothing small timing variations",
    ),
):
    """Re-export a saved transcription without running or loading the model."""
    if config is not None and ensemble is not None:
        raise typer.BadParameter("Choose --ensemble or --config, not both")
    try:
        intermediate = output / "intermediate"
        result = TranscriptionResult.model_validate_json(
            (intermediate / "model_notes.json").read_text()
        )
        audio = intermediate / "normalized.wav"
        settings_path = intermediate / "run_settings.json"
        warnings = ["Score regenerated from cached model notes; inference was not repeated."]
        selected_ensemble = None
        settings = {}
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            original_rhythm = Rhythm.model_validate(settings["rhythm"])
            length = settings["audio_duration"]
            original_start = settings["excerpt_start"]
            parts = [PartConfig.model_validate(p) for p in settings["parts"]]
            if settings.get("ensemble"):
                selected_ensemble = EnsembleConfig.model_validate(
                    {**settings["ensemble"], "parts": parts}
                )
        else:
            length = sf.info(audio).duration
            original_start = 0
            original_tempo = float(
                pretty_midi.PrettyMIDI(
                    str(intermediate / "raw_transcription.mid")
                ).get_tempo_changes()[1][0]
            )
            original_rhythm = analyze(audio, original_tempo, None, "1/16", 0)
            parts = load_parts()
            warnings.append(
                "Legacy run settings unavailable: assumed excerpt start 0, meter 4/4 and grid 1/16 unless overridden; tempo recovered from raw MIDI."
            )
        previous_part_ids = {p.id for p in parts}
        if config is not None or ensemble is not None:
            selected_ensemble = load_ensemble(config, ensemble or EnsemblePreset.BRITISH)
            parts = selected_ensemble.parts
        if selected_ensemble:
            warnings.extend(selected_ensemble.assumptions)
        excerpt_start = original_start if start is None else start
        if original_rhythm.mode == "beats" and all(
            value is None for value in (tempo, time_signature, start)
        ):
            updated_rhythm = original_rhythm.model_dump()
            updated_rhythm["grid"] = quantize or original_rhythm.grid
            updated_rhythm["pickup"] = original_rhythm.pickup if pickup is None else pickup
            from .rhythm import grid_quarters

            grid_quarters(updated_rhythm["grid"])
            numerator, denominator = map(int, original_rhythm.time_signature.split("/"))
            if not 0 <= updated_rhythm["pickup"] < numerator * 4 / denominator:
                raise ValueError("Pickup must be shorter than a bar")
            rhythm = Rhythm.model_validate(updated_rhythm)
        elif any(value is not None for value in (tempo, time_signature, quantize, pickup, start)):
            rhythm = analyze(
                audio,
                original_rhythm.tempo if tempo is None else tempo,
                time_signature or original_rhythm.time_signature,
                quantize or original_rhythm.grid,
                excerpt_start,
                original_rhythm.pickup if pickup is None else pickup,
            )
        else:
            rhythm = original_rhythm
        rhythm = Rhythm.model_validate(
            {**rhythm.model_dump(), "notation_mode": original_rhythm.notation_mode}
        )
        assigned_path = intermediate / "assigned_voices.json"
        if regularize:
            from .rhythm import regularize_tempo

            rhythm = regularize_tempo(rhythm)
        assignments = (
            [Assignment.model_validate(a) for a in json.loads(assigned_path.read_text())]
            if assigned_path.exists()
            else []
        )
        reassign = ensemble is not None or previous_part_ids != {p.id for p in parts}
        assignment_changed = reassign or (not assignments and bool(result.notes))
        if assignment_changed:
            effective_path = intermediate / "quality_notes.json"
            if effective_path.exists():
                from .phrases import assign_phrases

                effective = [
                    NoteEvent.model_validate(n) for n in json.loads(effective_path.read_text())
                ]
                assignments, _ = assign_phrases(effective, parts, rhythm=rhythm)
                retained = {n.id for n in effective}
                assignments.extend(
                    Assignment(
                        note=n, part=None, confidence=0, reason="cached quality rejection preserved"
                    )
                    for n in result.notes
                    if n.id not in retained
                )
            else:
                assignments = assign_voices(result.notes, parts)
            warnings.append(
                "Cached source notes reassigned to the selected ensemble; source pitches and timings unchanged."
            )
        if destination is not None and destination.resolve() != output.resolve():
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("Destination is not empty; choose a new --output directory")
            # Preserve source audio, untouched model chunks and note provenance.
            shutil.copytree(intermediate, destination / "intermediate", dirs_exist_ok=True)
            output = destination
            assigned_path = output / "intermediate/assigned_voices.json"
        if assignment_changed or not assigned_path.exists():
            assigned_path.write_text(json.dumps([a.model_dump() for a in assignments], indent=2))
        save_run_settings(output, rhythm, length, excerpt_start, parts, selected_ensemble)
        for key in ("quality", "refinement"):
            if settings.get(key):
                settings_path = output / "intermediate/run_settings.json"
                updated = json.loads(settings_path.read_text())
                updated[key] = settings[key]
                settings_path.write_text(json.dumps(updated, indent=2))
        (output / "diagnostics").mkdir(exist_ok=True)
        finish_run(
            result, assignments, parts, rhythm, length, excerpt_start, output, False, mscz, warnings
        )
    except (ValueError, RuntimeError, FileNotFoundError, Music21Exception) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def refine(
    source: Path,
    output: Path = typer.Option(Path("output-refined"), "--output"),
    beam_size: int = typer.Option(64, min=4, max=256),
    mscz: bool = typer.Option(True, "--mscz/--no-mscz"),
):
    """Refine cached notes with persistent instrument identities and joint rhythm fitting."""
    from .refinement import refine_cached, write_refinement_report

    try:
        run = refine_cached(source, output, beam_size)
        finish_run(
            run.result,
            run.assignments,
            run.parts,
            run.rhythm,
            run.duration,
            run.start,
            output,
            False,
            mscz,
            ["Cached predictions refined; raw notes unchanged."],
        )
        typer.echo(json.dumps(write_refinement_report(run, output), indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, Music21Exception) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def improve(
    source: Path,
    output: Path = typer.Option(Path("output-improved"), "--output"),
    ensemble: EnsemblePreset | None = None,
    passes: int = typer.Option(4, min=2, max=4),
    beam_size: int = typer.Option(64, min=4, max=256),
    cache: Path = CACHE,
    tempo: float | None = typer.Option(None, min=20, max=400),
    mscz: bool = typer.Option(True, "--mscz/--no-mscz"),
):
    """Run slower local multi-pass transcription, neural timing and phrase assignment."""
    from .quality import improve_run

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run = improve_run(source, output, cache, ensemble, passes, beam_size, tempo)
        save_run_settings(
            output, run.rhythm, run.duration, run.start, run.ensemble.parts, run.ensemble
        )
        settings_path = output / "intermediate/run_settings.json"
        settings = json.loads(settings_path.read_text())
        settings["quality"] = run.summary
        settings_path.write_text(json.dumps(settings, indent=2))
        finish_run(
            run.result,
            run.assignments,
            run.ensemble.parts,
            run.rhythm,
            run.duration,
            run.start,
            output,
            False,
            mscz,
            run.ensemble.assumptions + run.summary["limitations"],
        )
    except (ValueError, RuntimeError, FileNotFoundError, Music21Exception) as exc:
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
        original = {a.note.id: a.note for a in assigned}
        data["comparison"]["max_onset_quantization_shift_seconds"] = max(
            (
                abs(
                    quarters_to_seconds(m["onset_quarters"], Rhythm.model_validate(data["rhythm"]))
                    - original[m["source_id"]].start
                )
                for m in mapped
            ),
            default=0,
        )
        data["comparison"]["quantization_adjustments"] = [
            m["source_id"] for m in mapped if m.get("quantization_adjustment")
        ]
    typer.echo(json.dumps(data, indent=2))


def main():
    # Preserve requested bare-file interface while retaining diagnostic subcommands.
    commands = {"transcribe", "diagnose", "doctor", "download-model", "resume", "improve", "refine"}
    if len(sys.argv) > 1 and sys.argv[1] not in commands and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "transcribe")
    app()

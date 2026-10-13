from pathlib import Path
from typing import Protocol

import pretty_midi
import yaml
from pydantic import BaseModel, Field, model_validator


class NoteEvent(BaseModel):
    id: str
    pitch: int = Field(ge=0, le=127)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    velocity: int = Field(default=80, ge=1, le=127)
    program: int = Field(default=56, ge=0, le=127)
    track: int = 0
    drum: bool = False
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def duration_positive(self):
        if self.end <= self.start:
            raise ValueError("Note end must follow start")
        return self


class PartConfig(BaseModel):
    id: str
    name: str
    family: str
    typical: tuple[int, int]
    extended: tuple[int, int]
    clef: str = "treble"
    written_to_sounding: str
    preferred_pitch: int
    max_simultaneous: int = Field(default=2, ge=1)
    program: int
    instrument_sound: str


class Assignment(BaseModel):
    note: NoteEvent
    part: str | None
    confidence: float
    reason: str


class TranscriptionResult(BaseModel):
    notes: list[NoteEvent]
    model: str
    device: str
    runtime: float = 0
    warnings: list[str] = Field(default_factory=list)


class TranscriptionBackend(Protocol):
    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


def load_parts(path: Path | None = None) -> list[PartConfig]:
    path = path or Path(__file__).parent / "configs/british_brass_band.yaml"
    parts = [PartConfig.model_validate(p) for p in yaml.safe_load(path.read_text())["parts"]]
    if len({p.id for p in parts}) != len(parts):
        raise ValueError("Part IDs must be unique")
    return parts


def read_midi(path: Path, offset: float = 0, prefix: str = "n") -> list[NoteEvent]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        NoteEvent(
            id=f"{prefix}-{ti}-{ni}",
            pitch=n.pitch,
            start=n.start + offset,
            end=n.end + offset,
            velocity=n.velocity,
            program=track.program,
            track=ti,
            drum=track.is_drum,
        )
        for ti, track in enumerate(midi.instruments)
        for ni, n in enumerate(track.notes)
        if n.end > n.start
    ]


def write_midi(notes: list[NoteEvent], path: Path, tempo: float = 120) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    tracks = {}
    for n in notes:
        key = (n.track, n.program, n.drum)
        if key not in tracks:
            tracks[key] = pretty_midi.Instrument(n.program, is_drum=n.drum, name=f"Track {n.track}")
            midi.instruments.append(tracks[key])
        tracks[key].notes.append(pretty_midi.Note(n.velocity, n.pitch, n.start, n.end))
    midi.write(str(path))

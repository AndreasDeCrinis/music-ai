from enum import Enum
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
    instrument_name: str | None = None
    family: str
    typical: tuple[int, int]
    extended: tuple[int, int]
    clef: str = "treble"
    written_to_sounding: str
    preferred_pitch: int
    max_simultaneous: int = Field(default=2, ge=1)
    program: int
    # Model output programs counted as evidence for this part. YourMT3 rarely emits
    # the exact GM program for band woodwinds (e.g. clarinet as soprano/alto sax).
    model_programs: list[int] = Field(default_factory=list)
    instrument_sound: str
    percussion: bool = False

    @property
    def evidence_programs(self) -> set[int]:
        return set(self.model_programs) | {self.program}


def evidence_groups(parts: list[PartConfig]) -> dict[int, int]:
    """Map each model program that is evidence for a part to that part's own program."""
    groups: dict[int, int] = {}
    for p in parts:
        if not p.percussion:
            for program in sorted(p.evidence_programs):
                groups.setdefault(program, p.program)
    return groups


class EnsemblePreset(str, Enum):
    BRITISH = "british-brass-band"
    BRASSARANKA = "brassaranka"


class EnsembleConfig(BaseModel):
    id: str = "custom"
    name: str = "Custom ensemble"
    sources: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    parts: list[PartConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_parts(self):
        if len({p.id for p in self.parts}) != len(self.parts):
            raise ValueError("Part IDs must be unique")
        return self


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


def load_ensemble(
    path: Path | None = None, preset: EnsemblePreset = EnsemblePreset.BRITISH
) -> EnsembleConfig:
    preset = EnsemblePreset(preset)
    filenames = {
        EnsemblePreset.BRITISH: "british_brass_band.yaml",
        EnsemblePreset.BRASSARANKA: "brassaranka.yaml",
    }
    path = path or Path(__file__).parent / "configs" / filenames[preset]
    return EnsembleConfig.model_validate(yaml.safe_load(path.read_text()))


def load_parts(
    path: Path | None = None, preset: EnsemblePreset = EnsemblePreset.BRITISH
) -> list[PartConfig]:
    return load_ensemble(path, preset).parts


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

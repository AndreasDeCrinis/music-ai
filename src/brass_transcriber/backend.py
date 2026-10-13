import logging
import os
import time
from pathlib import Path

import soundfile as sf

from .models import NoteEvent, TranscriptionResult, read_midi

log = logging.getLogger(__name__)
CHECKPOINT_NAME = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"


def deduplicate(notes: list[NoteEvent], tolerance: float = 0.12) -> list[NoteEvent]:
    """Merge boundary duplicates; keep rapid repetitions with distinct attacks."""
    result: list[NoteEvent] = []
    index: dict[tuple, list[NoteEvent]] = {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        key = (n.program, n.pitch, n.drum)
        candidates = index.setdefault(key, [])
        match = next(
            (
                p
                for p in reversed(candidates[-20:])
                if p.track != n.track
                and abs(p.start - n.start) <= tolerance
                and min(p.end, n.end) > max(p.start, n.start)
            ),
            None,
        )
        if match:
            match.end = max(match.end, n.end)
            match.provenance.extend(n.provenance or [n.id])
        else:
            result.append(n)
            candidates.append(n)
    return result


class YourMT3Backend:
    def __init__(
        self,
        cache: Path,
        raw_dir: Path,
        start: float = 0,
        chunk_seconds: float = 30,
        overlap: float = 2,
        device: str = "auto",
    ):
        if chunk_seconds <= 2 * overlap or overlap < 0:
            raise ValueError("chunk-seconds must exceed twice the nonnegative overlap")
        if device not in {"auto", "cpu"}:
            raise ValueError(
                "This verified MT3-Infer backend supports CPU on Mac; MPS is unsupported"
            )
        self.cache, self.raw_dir, self.start = cache, raw_dir, start
        self.chunk_seconds, self.overlap = chunk_seconds, overlap

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        import torch
        from mt3_infer import load_model

        checkpoint = self.cache / "yourmt3" / CHECKPOINT_NAME / "last.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}. Run brass-transcribe download-model"
            )
        os.environ["MT3_CHECKPOINT_DIR"] = str(self.cache.resolve())
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        log.info("YourMT3+ inference device: cpu (MPS unsupported by selected adapter)")
        started = time.perf_counter()
        model = load_model(
            "yourmt3", checkpoint_path=str(checkpoint.resolve()), device="cpu", auto_download=False
        )
        notes = []
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(audio_path) as audio:
            sr = audio.samplerate
            if sr != 16000 or audio.channels != 1:
                raise ValueError("Backend requires mono 16 kHz audio")
            length = audio.frames / sr
            step = self.chunk_seconds - self.overlap
            position, chunk = 0.0, 0
            while position < length:
                audio.seek(round(position * sr))
                data = audio.read(round(self.chunk_seconds * sr), dtype="float32")
                log.info("Transcribing chunk %d at %.2fs", chunk, position + self.start)
                midi = model.transcribe(data, sr=sr)
                raw = self.raw_dir / f"chunk_{chunk:05d}.mid"
                midi.save(str(raw))
                events = read_midi(raw, offset=position + self.start, prefix=f"c{chunk}")
                # Ownership by midpoint of overlap prevents truncated boundary attacks.
                lower = position + (self.overlap / 2 if chunk else 0)
                upper = position + self.chunk_seconds - self.overlap / 2
                final = position + self.chunk_seconds >= length
                merge_chunk(
                    notes,
                    events,
                    lower + self.start,
                    float("inf") if final else upper + self.start,
                    position + self.start,
                    self.start + length,
                )
                if final:
                    break
                position += step
                chunk += 1
        # Track is source track; don't deduplicate unrelated unisons within a chunk.
        return TranscriptionResult(
            notes=notes,
            model="YourMT3+ YPTF.MoE+Multi (noPS)",
            device="cpu",
            runtime=time.perf_counter() - started,
            warnings=[
                "Model velocities may be fixed rather than expressive estimates.",
                "Chunk stitching is heuristic; uncertain sustained-note seams need review.",
            ],
        )


def merge_chunk(
    previous: list[NoteEvent],
    events: list[NoteEvent],
    lower: float,
    upper: float,
    chunk_start: float,
    recording_end: float,
) -> None:
    """Merge duplicate attacks and sustained notes observed in adjacent overlaps.

    Only match earlier chunks; preserve within-chunk unisons and repeated attacks.
    New attacks belong to one midpoint ownership interval.
    """
    earlier = [n for n in previous if n.end >= chunk_start - 0.12]
    matched = set()
    for n in sorted(events, key=lambda n: (n.start, n.pitch)):
        n.end = min(n.end, recording_end)
        if n.end <= n.start:
            continue
        n.provenance = [n.id]
        candidates = [
            p
            for p in earlier
            if p.id not in matched
            and (p.pitch, p.program, p.drum) == (n.pitch, n.program, n.drum)
            and min(p.end, n.end) > max(p.start, n.start)
            and (
                abs(p.start - n.start) <= 0.12
                or (
                    n.start <= chunk_start + 0.12
                    and p.start < chunk_start
                    and p.end > chunk_start + 0.12
                )
            )
        ]
        if candidates:
            p = min(candidates, key=lambda p: abs(p.start - n.start))
            p.end = max(p.end, n.end)
            p.provenance = (p.provenance or [p.id]) + n.provenance
            matched.add(p.id)
        elif lower <= n.start < upper:
            previous.append(n)

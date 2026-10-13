import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf

SAMPLE_RATE = 16000


def ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def preprocess(
    source: Path,
    target: Path,
    start: float = 0,
    duration: float | None = None,
    normalize: bool = False,
) -> float:
    if not source.is_file():
        raise ValueError(f"Input does not exist: {source}")
    if source.suffix.lower() not in {".mp3", ".wav", ".flac"}:
        raise ValueError("Supported inputs: MP3, WAV, FLAC")
    args = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    args += ["-ss", str(start)]
    if duration is not None:
        args += ["-t", str(duration)]
    if normalize:
        args += ["-af", "loudnorm=I=-23:TP=-2:LRA=7"]
    args += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(target)]
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg failed: {completed.stderr[-2000:]}")
    info = sf.info(target)
    if info.frames == 0:
        raise ValueError("Requested excerpt contains no audio")
    return info.duration

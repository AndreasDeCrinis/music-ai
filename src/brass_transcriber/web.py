"""Local upload UI and persistent, single-worker transcription queue."""

import contextlib
import fcntl
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .backend import CHECKPOINT_NAME, PS_CHECKPOINT_NAME

STATIC = Path(__file__).with_name("static")


@dataclass
class Settings:
    data: Path
    cache: Path
    max_upload_mb: int = 100
    max_duration: int = 900
    max_jobs: int = 20
    job_timeout: int = 7200

    @classmethod
    def environment(cls):
        return cls(
            Path(os.getenv("BRASS_DATA", "web-data")).resolve(),
            Path(os.getenv("BRASS_CACHE", ".cache/models")).resolve(),
            int(os.getenv("BRASS_MAX_UPLOAD_MB", "100")),
            int(os.getenv("BRASS_MAX_DURATION", "900")),
            int(os.getenv("BRASS_MAX_JOBS", "20")),
            int(os.getenv("BRASS_JOB_TIMEOUT", "7200")),
        )

    def missing_models(self):
        required = {
            "YourMT3 noPS": self.cache / "yourmt3" / CHECKPOINT_NAME / "last.ckpt",
            "YourMT3 PS": self.cache / "yourmt3" / PS_CHECKPOINT_NAME / "model.ckpt",
            "Beat This": self.cache / "beat_this/final0.ckpt",
        }
        return [name for name, path in required.items() if not path.is_file()]


def score_files(output):
    """Only publish deliberate score artifacts, never source audio or intermediates."""
    return sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and (
            path.parent == output
            and path.suffix in {".mscz", ".musicxml", ".mid"}
            or path.parent == output / "parts"
            and path.suffix == ".musicxml"
            or path == output / "diagnostics/transcription_report.json"
        )
    )


def validate_audio(source: Path, max_duration: int) -> float:
    """Decode a bounded length using the same FFmpeg as inference (also works locally)."""
    import soundfile as sf

    from .audio import ffmpeg_path

    target = source.with_name("validation.wav")
    try:
        decoded = subprocess.run(
            [
                ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-t",
                str(max_duration + 1),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            capture_output=True,
            timeout=120,
        )
        if decoded.returncode:
            raise ValueError("This file could not be decoded as audio")
        return sf.info(target).duration
    finally:
        target.unlink(missing_ok=True)


class Jobs:
    def __init__(self, settings, runner=None):
        self.settings = settings
        self.runner = runner or self.transcribe
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.process = None
        self.guard = threading.Lock()
        self.thread = None

    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(self.settings.data / "jobs.sqlite", timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def start(self):
        self.settings.data.mkdir(parents=True, exist_ok=True)
        self.lock = (self.settings.data / ".worker.lock").open("w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("This data directory already has a worker; use one web process.")
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, filename TEXT, ensemble TEXT, passes INTEGER,
                status TEXT, stage TEXT, created REAL, updated REAL, error TEXT)""")
            db.execute(
                "UPDATE jobs SET status='failed', stage='Interrupted', error=? WHERE status='running'",
                (
                    "Server stopped during transcription. Upload again to retry; cached files are retained.",
                ),
            )
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        with self.guard:
            if self.process and self.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
        if self.thread:
            self.thread.join(timeout=10)
        self.lock.close()

    def get(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(404, "Job not found")
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found")
        return dict(row)

    def update(self, job_id, **values):
        values["updated"] = time.time()
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?",
                [*values.values(), job_id],
            )

    def work(self):
        while not self.stop.is_set():
            with self.connect() as db:
                row = db.execute(
                    "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
                ).fetchone()
            if row is None:
                self.wake.wait(1)
                self.wake.clear()
                continue
            job = dict(row)
            self.update(job["id"], status="running", stage="Checking audio")
            try:
                self.runner(job)
                output = self.settings.data / job["id"] / "score"
                if not (output / "full_score.mscz").is_file():
                    raise RuntimeError(
                        "Native MuseScore export failed. See the job log; MusicXML may still be available on disk."
                    )
                files = score_files(output)
                with zipfile.ZipFile(output / "scores.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in files:
                        archive.write(path, path.relative_to(output))
                self.update(job["id"], status="completed", stage="Ready to download")
            except Exception as exc:
                self.update(job["id"], status="failed", stage="Failed", error=str(exc)[:2000])

    def command(self, job, args, stage):
        if self.stop.is_set():
            raise RuntimeError("Server is shutting down")
        self.update(job["id"], stage=stage)
        folder = self.settings.data / job["id"]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        with (folder / "worker.log").open("a") as log:
            log.write(f"\n{stage}\n")
            log.flush()
            with self.guard:
                if self.stop.is_set():
                    raise RuntimeError("Server is shutting down")
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "brass_transcriber.cli", *map(str, args)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                process = self.process
            try:
                code = process.wait(timeout=self.settings.job_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError("Transcription exceeded the job time limit") from None
            finally:
                with self.guard:
                    self.process = None
            if code:
                raise RuntimeError(
                    f"{stage} failed (exit {code}). Open the processing log for details."
                )

    def transcribe(self, job):
        folder = self.settings.data / job["id"]
        duration = validate_audio(folder / "input.mp3", self.settings.max_duration)
        if not 0 < duration <= self.settings.max_duration:
            raise ValueError(
                f"Audio must be between 0 and {self.settings.max_duration} seconds long"
            )
        self.command(
            job,
            [
                "transcribe",
                folder / "input.mp3",
                "--output",
                folder / "baseline",
                "--cache",
                self.settings.cache,
                "--ensemble",
                job["ensemble"],
                "--no-mscz",
            ],
            "Transcribing the original recording",
        )
        self.command(
            job,
            [
                "improve",
                folder / "baseline",
                "--output",
                folder / "score",
                "--cache",
                self.settings.cache,
                "--passes",
                job["passes"],
                "--mscz",
            ],
            "Refining notes, rhythm and instrument parts",
        )


class UploadLimit:
    """Bound the entire body before multipart parsing can spool it to disk."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        count = 0

        async def bounded_receive():
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > self.limit:
                raise HTTPException(413, "Upload exceeds the configured size limit")
            return message

        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = 0
        if length > self.limit:
            return await JSONResponse(
                {"detail": "Upload exceeds the configured size limit"}, status_code=413
            )(scope, receive, send)
        await self.app(scope, bounded_receive, send)


def create_app(settings=None, runner=None):
    settings = settings or Settings.environment()
    jobs = Jobs(settings, runner)

    @asynccontextmanager
    async def lifespan(app):
        jobs.start()
        try:
            yield
        finally:
            jobs.close()

    app = FastAPI(title="Brass Studio", lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(UploadLimit, limit=settings.max_upload_mb * 1024 * 1024 + 65536)

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method in {"POST", "DELETE"}
            and origin
            and origin != str(request.base_url).rstrip("/")
        ):
            return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health():
        missing = settings.missing_models()
        return {
            "status": "ok",
            "ready": not missing,
            "missing_models": missing,
            "max_upload_mb": settings.max_upload_mb,
            "max_duration": settings.max_duration,
        }

    @app.get("/api/jobs")
    def listing():
        with jobs.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created DESC")]

    @app.post("/api/jobs", status_code=202)
    async def upload(
        file: UploadFile = File(...), ensemble: str = Form("brassaranka"), passes: int = Form(4)
    ):
        if ensemble not in {"brassaranka", "british-brass-band"} or passes not in {2, 4}:
            raise HTTPException(422, "Choose a supported ensemble and quality mode")
        if not (file.filename or "").lower().endswith(".mp3"):
            raise HTTPException(415, "Please upload an MP3 file")
        if missing := settings.missing_models():
            raise HTTPException(503, "Models are not installed: " + ", ".join(missing))
        job_id = uuid.uuid4().hex
        folder = settings.data / job_id
        folder.mkdir()
        try:
            size = 0
            with (folder / "input.mp3").open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "MP3 exceeds the configured size limit")
                    target.write(chunk)
            if not size:
                raise HTTPException(422, "The uploaded MP3 is empty")
            now = time.time()
            with jobs.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                if count >= settings.max_jobs:
                    raise HTTPException(
                        429, "Job storage is full. Delete an old job before uploading."
                    )
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        Path(file.filename).name[:200],
                        ensemble,
                        passes,
                        "queued",
                        "Waiting for the worker",
                        now,
                        now,
                        None,
                    ),
                )
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await file.close()
        jobs.wake.set()
        return jobs.get(job_id)

    @app.get("/api/jobs/{job_id}")
    def detail(job_id: str):
        job = jobs.get(job_id)
        job["files"] = []
        if job["status"] == "completed":
            output = settings.data / job_id / "score"
            job["files"] = [p.relative_to(output).as_posix() for p in score_files(output)] + [
                "scores.zip"
            ]
        return job

    @app.get("/api/jobs/{job_id}/log")
    def log(job_id: str):
        jobs.get(job_id)
        path = settings.data / job_id / "worker.log"
        if not path.exists():
            return {"log": "Waiting for processing to start."}
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 16000))
            return {"log": handle.read().decode("utf-8", errors="replace")}

    @app.get("/api/jobs/{job_id}/files/{filename:path}")
    def download(job_id: str, filename: str):
        if jobs.get(job_id)["status"] != "completed":
            raise HTTPException(409, "Job is not complete")
        output = settings.data / job_id / "score"
        allowed = {p.relative_to(output).as_posix(): p for p in score_files(output)}
        allowed["scores.zip"] = output / "scores.zip"
        if filename not in allowed:
            raise HTTPException(404, "File not found")
        return FileResponse(allowed[filename], filename=allowed[filename].name)

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        with jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = jobs.get(job_id)
            if job["status"] in {"running", "queued"}:
                raise HTTPException(409, "Wait for the job to finish before deleting it")
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            shutil.rmtree(settings.data / job_id, ignore_errors=True)
        return {"deleted": job_id}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    return app


def main():
    import uvicorn

    uvicorn.run(
        create_app(), host=os.getenv("BRASS_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000"))
    )


if __name__ == "__main__":
    main()

import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from brass_transcriber.backend import CHECKPOINT_NAME, PS_CHECKPOINT_NAME
from brass_transcriber.web import Settings, create_app


@pytest.fixture
def settings(tmp_path):
    settings = Settings(tmp_path / "data", tmp_path / "models", max_upload_mb=1)
    for path in [
        settings.cache / "yourmt3" / CHECKPOINT_NAME / "last.ckpt",
        settings.cache / "yourmt3" / PS_CHECKPOINT_NAME / "model.ckpt",
        settings.cache / "beat_this/final0.ckpt",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    return settings


def runner(settings):
    def run(job):
        folder = settings.data / job["id"] / "score"
        folder.mkdir()
        for name in ["full_score.mscz", "full_score.musicxml", "full_score.mid"]:
            (folder / name).write_bytes(b"score fixture")
        (folder / "parts").mkdir()
        (folder / "parts/tuba.musicxml").write_text("part")
        (folder / "intermediate").mkdir()
        (folder / "intermediate/private.json").write_text("not public")

    return run


def wait(client, job_id):
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    pytest.fail("Worker did not finish")


def upload(client):
    return client.post("/api/jobs", files={"file": ("song.mp3", b"fake audio", "audio/mpeg")})


def test_upload_download_restart_delete(settings):
    app = create_app(settings, runner(settings))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/style.css").status_code == 200
        assert client.get("/api/health").json()["ready"]
        response = upload(client)
        assert response.status_code == 202
        job = wait(client, response.json()["id"])
        assert job["status"] == "completed"
        assert "parts/tuba.musicxml" in job["files"]
        base = f"/api/jobs/{job['id']}"
        assert client.get(base + "/files/full_score.mscz").content == b"score fixture"
        assert client.get(base + "/files/intermediate/private.json").status_code == 404
        with zipfile.ZipFile(settings.data / job["id"] / "score/scores.zip") as archive:
            assert "parts/tuba.musicxml" in archive.namelist()
            assert not any("intermediate" in name for name in archive.namelist())
    with TestClient(create_app(settings, runner(settings))) as client:
        assert client.get(base).json()["status"] == "completed"
        assert client.delete(base).status_code == 200
        assert client.get(base).status_code == 404
        assert not (settings.data / job["id"]).exists()


def test_validation_and_limits(settings):
    with TestClient(create_app(settings, runner(settings))) as client:
        assert client.post("/api/jobs", files={"file": ("bad.wav", b"x")}).status_code == 415
        assert client.post("/api/jobs", files={"file": ("empty.mp3", b"")}).status_code == 422
        assert (
            client.post(
                "/api/jobs", files={"file": ("song.mp3", b"x")}, data={"ensemble": "choir"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/jobs", files={"file": ("song.mp3", b"x" * (2 * 1024 * 1024))}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/api/jobs",
                files={"file": ("song.mp3", b"x")},
                headers={"origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert client.get("/api/jobs/../../etc/passwd").status_code == 404
        settings.max_jobs = 0
        assert upload(client).status_code == 429
        assert not any(p.is_dir() for p in settings.data.iterdir())
        (settings.cache / "beat_this/final0.ckpt").unlink()
        assert not client.get("/api/health").json()["ready"]
        assert upload(client).status_code == 503


def test_failure_and_native_required(settings):
    def fail(job):
        raise RuntimeError("bad audio")

    with TestClient(create_app(settings, fail)) as client:
        job = wait(client, upload(client).json()["id"])
        assert job["status"] == "failed"
        assert job["error"] == "bad audio"
        assert client.get(f"/api/jobs/{job['id']}/files/full_score.mscz").status_code == 409
    with TestClient(create_app(settings, lambda job: None)) as client:
        job = wait(client, upload(client).json()["id"])
        assert job["status"] == "failed"
        assert "MuseScore export failed" in job["error"]


def test_single_worker_and_interrupted_job(settings):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def blocked(job):
        calls.append(job["id"])
        started.set()
        release.wait(5)
        runner(settings)(job)

    app = create_app(settings, blocked)
    with TestClient(app) as client:
        first = upload(client).json()["id"]
        assert started.wait(2)
        second = upload(client).json()["id"]
        assert client.get(f"/api/jobs/{second}").json()["status"] == "queued"
        assert client.delete(f"/api/jobs/{first}").status_code == 409
        assert client.delete(f"/api/jobs/{second}").status_code == 409
        with pytest.raises(RuntimeError, match="already has a worker"):
            with TestClient(create_app(settings)):
                pass
        assert calls == [first]
        release.set()
        assert wait(client, first)["status"] == "completed"
        assert wait(client, second)["status"] == "completed"
        app.state.jobs.update(first, status="running")
    with TestClient(create_app(settings, runner(settings))) as client:
        assert client.get(f"/api/jobs/{first}").json()["status"] == "failed"
        assert "Server stopped" in client.get(f"/api/jobs/{first}").json()["error"]


def test_cli_module_entrypoint():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "brass_transcriber.cli", "--help"], capture_output=True
    )
    assert result.returncode == 0
    assert b"transcribe" in result.stdout


def test_streamed_upload_limit(settings):
    with TestClient(create_app(settings, runner(settings))) as client:
        boundary = "upload-boundary"
        prefix = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="song.mp3"\r\nContent-Type: audio/mpeg\r\n\r\n'
        ).encode()
        chunks = iter(
            [prefix, b"x" * (1024 * 1024), b"x" * (1024 * 1024), f"\r\n--{boundary}--\r\n".encode()]
        )
        response = client.post(
            "/api/jobs",
            content=chunks,
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        assert response.status_code == 413
        assert client.get("/api/jobs").json() == []


def test_bad_audio_is_rejected_before_inference(settings):
    with TestClient(create_app(settings)) as client:
        job = wait(client, upload(client).json()["id"])
        assert job["status"] == "failed"
        assert "decoded as audio" in job["error"]
        assert not (settings.data / job["id"] / "validation.wav").exists()
        assert not (settings.data / job["id"] / "baseline").exists()

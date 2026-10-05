"""Spotimind API. Run: uv run uvicorn app.main:app --reload --port 8000 (from backend/)."""

from __future__ import annotations

import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import explain
from .similarity import overlap, top_k
from .store import Store, load_store

SPACES = ("brain", "audio")
CHUNK = 256 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "store", None) is None:
        app.state.store = load_store()
    yield


app = FastAPI(title="Spotimind API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SPOTIMIND_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


def store() -> Store:
    return app.state.store


def song_index(song_id: str) -> int:
    i = store().index.get(song_id)
    if i is None:
        raise HTTPException(404, f"unknown song id {song_id!r}")
    return i


def neighbor_list(i: int, k: int, space: str) -> list[dict]:
    s = store()
    return [{"id": s.songs[j]["id"], "title": s.songs[j]["title"], "artist": s.songs[j].get("artist"),
             "score": round(score, 6)} for j, score in top_k(s.sims[space], i, k)]


@app.get("/api/meta")
def get_meta() -> dict:
    s = store()
    key, model = explain.gemini_config()
    return {**s.meta, "mock": bool(s.meta.get("mock")), "n_songs": len(s.songs),
            "networks": [{k: n[k] for k in ("id", "name", "description")} for n in s.networks["networks"]],
            # catalog-wide range of per-song network means, so bars share one scale across songs
            "network_range": {"min": float(s.net_means.min()), "max": float(s.net_means.max())},
            "explain_available": key is not None, "explain_model": model if key else None}


@app.get("/api/songs")
def list_songs() -> list[dict]:
    return store().songs


@app.get("/api/songs/{song_id}")
def get_song(song_id: str) -> dict:
    s = store()
    i = song_index(song_id)
    return {**s.songs[i], "networks": s.network_values(i)}


@app.get("/api/songs/{song_id}/timeline")
def get_timeline(song_id: str) -> Response:
    song_index(song_id)
    return Response(store().timeline_bytes(song_id), media_type="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/songs/{song_id}/neighbors")
def get_neighbors(song_id: str, k: int = Query(10, ge=1, le=100), space: str = Query("brain")) -> dict:
    if space not in SPACES:
        raise HTTPException(422, f"space must be one of {SPACES}")
    i = song_index(song_id)
    return {"id": song_id, "space": space, "k": k, "neighbors": neighbor_list(i, k, space)}


@app.get("/api/compare/{song_id}")
def compare(song_id: str, k: int = Query(10, ge=1, le=100)) -> dict:
    i = song_index(song_id)
    brain, audio = neighbor_list(i, k, "brain"), neighbor_list(i, k, "audio")
    ov = overlap([n["id"] for n in brain], [n["id"] for n in audio])
    return {"id": song_id, "k": k, "brain": brain, "audio": audio, "overlap": ov}


@app.get("/api/mesh")
def get_mesh_descriptor() -> Response:
    path = store().data_dir / "mesh" / "cortex.json"
    if not path.exists():
        raise HTTPException(404, "mesh not exported; run scripts/export_mesh.py")
    return FileResponse(path, media_type="application/json")


@app.get("/api/mesh/bin")
def get_mesh_bin() -> Response:
    path = store().data_dir / "mesh" / "cortex.bin"
    if not path.exists():
        raise HTTPException(404, "mesh not exported; run scripts/export_mesh.py")
    return FileResponse(path, media_type="application/octet-stream")


@app.get("/api/networks")
def get_networks() -> dict:
    return store().networks


def _parse_range(header: str, size: int) -> tuple[int, int]:
    unit, _, spec = header.partition("=")
    if unit.strip() != "bytes" or "," in spec:
        raise ValueError("only single byte ranges are supported")
    start_s, _, end_s = spec.strip().partition("-")
    if start_s == "":  # suffix range: last N bytes
        n = int(end_s)
        start, end = max(size - n, 0), size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        raise ValueError("unsatisfiable range")
    return start, end


def _file_chunks(path: Path, start: int, length: int):
    with open(path, "rb") as f:
        f.seek(start)
        while length > 0:
            chunk = f.read(min(CHUNK, length))
            if not chunk:
                break
            length -= len(chunk)
            yield chunk


@app.get("/api/audio/{song_id}")
def get_audio(song_id: str, request: Request) -> Response:
    s = store()
    path = s.audio_dir / s.songs[song_index(song_id)]["filename"]
    if not path.is_file():
        raise HTTPException(404, f"audio file missing: {path.name} (expected in {s.audio_dir})")
    size = path.stat().st_size
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes"}
    range_header = request.headers.get("range")
    if not range_header:
        return StreamingResponse(_file_chunks(path, 0, size), media_type=media,
                                 headers={**headers, "Content-Length": str(size)})
    try:
        start, end = _parse_range(range_header, size)
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    length = end - start + 1
    return StreamingResponse(_file_chunks(path, start, length), status_code=206, media_type=media,
                             headers={**headers, "Content-Range": f"bytes {start}-{end}/{size}",
                                      "Content-Length": str(length)})


@app.get("/api/songs/{song_id}/cover")
def get_cover(song_id: str) -> Response:
    """Cover art embedded in the audio file (ID3 APIC). 404 when the file has none (e.g. mock WAVs)."""
    s = store()
    cover = s.cover(s.songs[song_index(song_id)]["filename"])
    if cover is None:
        raise HTTPException(404, "no embedded cover art")
    data, mime = cover
    return Response(data, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})


class ExplainBody(BaseModel):
    other_id: str


@app.post("/api/songs/{song_id}/explain")
def explain_match(song_id: str, body: ExplainBody) -> dict:
    s = store()
    i, j = song_index(song_id), song_index(body.other_id)
    key, model = explain.gemini_config()
    if not key:
        return {"available": False, "text": None,
                "message": "Explanations are disabled: set GEMINI_API_KEY in the backend environment."}
    a = {"title": s.songs[i]["title"], "artist": s.songs[i].get("artist"),
         "networks": s.network_values(i), "percentiles": s.network_percentiles(i)}
    b = {"title": s.songs[j]["title"], "artist": s.songs[j].get("artist"),
         "networks": s.network_values(j), "percentiles": s.network_percentiles(j)}
    sims = {space: float(s.sims[space][i, j]) for space in SPACES}
    names = {n["id"]: n["name"] for n in s.networks["networks"]}
    catalog = {nid: {"std": float(sd)} for nid, sd in zip(s.network_ids, s.net_means.std(0))}
    prompt = explain.build_prompt(a, b, names, sims, catalog)
    try:
        text, model = explain.generate(prompt)
    except explain.GeminiBusy:
        raise HTTPException(503, "Gemini is overloaded right now (retried, and tried the fallback model). Try again in a minute.")
    except Exception as e:  # bad key, unknown model, network: report, don't crash
        raise HTTPException(502, f"Gemini request failed: {type(e).__name__}: {str(e)[:200]}")
    return {"available": True, "text": text, "model": model, "inputs": {"similarity": sims,
            "networks": {"a": a["networks"], "b": b["networks"]}},
            "label": "Interpretation, not a measurement"}

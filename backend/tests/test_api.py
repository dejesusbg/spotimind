import numpy as np
import pytest

from app.similarity import cosine_matrix


def test_meta_flags_mock(client):
    meta = client.get("/api/meta").json()
    assert meta["mock"] is True
    assert meta["explain_available"] is False
    assert len(meta["networks"]) == 8


def test_songs_and_detail(client):
    songs = client.get("/api/songs").json()
    assert len(songs) == 12
    detail = client.get(f"/api/songs/{songs[0]['id']}").json()
    assert set(detail["networks"]) == {"auditory", "visual", "somatomotor", "dorsal_attention",
                                       "ventral_attention", "limbic", "control", "default"}
    assert client.get("/api/songs/nope").status_code == 404


def test_network_means_are_plain_parcel_averages(client, mock_dir):
    import json
    song = client.get("/api/songs").json()[3]
    detail = client.get(f"/api/songs/{song['id']}").json()
    tl = np.load(mock_dir / "data" / "timelines" / f"{song['id']}.npy")
    nets = json.loads((mock_dir / "data" / "networks.json").read_text())["networks"]
    for net in nets:
        assert detail["networks"][net["id"]] == pytest.approx(float(tl[:, net["parcels"]].mean()), abs=1e-5)


@pytest.mark.parametrize("space", ["brain", "audio"])
def test_neighbors_match_numpy(client, mock_dir, space):
    songs = client.get("/api/songs").json()
    emb = np.load(mock_dir / "data" / f"embeddings_{space}.npy")
    sims = cosine_matrix(emb)
    i = 5
    res = client.get(f"/api/songs/{songs[i]['id']}/neighbors", params={"k": 4, "space": space}).json()
    ids = [n["id"] for n in res["neighbors"]]
    assert songs[i]["id"] not in ids and len(ids) == 4
    row = sims[i].copy()
    row[i] = -np.inf
    expected = [songs[j]["id"] for j in np.argsort(-row, kind="stable")[:4]]
    assert ids == expected
    assert res["neighbors"][0]["score"] == pytest.approx(row.max(), abs=1e-5)


def test_neighbors_bad_space(client):
    sid = client.get("/api/songs").json()[0]["id"]
    assert client.get(f"/api/songs/{sid}/neighbors", params={"space": "vibes"}).status_code == 422


def test_compare_overlap(client):
    sid = client.get("/api/songs").json()[2]["id"]
    res = client.get(f"/api/compare/{sid}", params={"k": 5}).json()
    b, a = {n["id"] for n in res["brain"]}, {n["id"] for n in res["audio"]}
    assert res["overlap"]["shared"] == len(a & b)
    assert res["overlap"]["jaccard"] == pytest.approx(len(a & b) / len(a | b))
    # k = N-1 means both lists contain every other song
    full = client.get(f"/api/compare/{sid}", params={"k": 11}).json()
    assert full["overlap"]["shared"] == 11 and full["overlap"]["jaccard"] == 1.0


def test_timeline_binary(client, mock_dir):
    song = client.get("/api/songs").json()[1]
    raw = client.get(f"/api/songs/{song['id']}/timeline").content
    T, P = np.frombuffer(raw[:8], "<u4")
    start, rate = np.frombuffer(raw[8:16], "<f4")
    data = np.frombuffer(raw[16:], "<f4").reshape(T, P)
    assert (T, P) == (song["bin_count"], 400)
    assert start == pytest.approx(song["timeline_start_s"]) and rate == 1.0
    assert np.allclose(data, np.load(mock_dir / "data" / "timelines" / f"{song['id']}.npy"))


def test_audio_range_requests(client):
    sid = client.get("/api/songs").json()[0]["id"]
    full = client.get(f"/api/audio/{sid}")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    size = len(full.content)
    part = client.get(f"/api/audio/{sid}", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{size}"
    assert part.content == full.content[100:200]
    tail = client.get(f"/api/audio/{sid}", headers={"Range": "bytes=-50"})
    assert tail.content == full.content[-50:]
    bad = client.get(f"/api/audio/{sid}", headers={"Range": f"bytes={size + 10}-"})
    assert bad.status_code == 416


def test_mesh_endpoints(client):
    desc = client.get("/api/mesh").json()
    raw = client.get("/api/mesh/bin").content
    par = desc["buffers"]["parcel"]
    parcel = np.frombuffer(raw, "<i2", count=par["count"], offset=par["offset"])
    assert parcel.shape[0] == desc["n_vertices"] and parcel.max() < 400


def test_explain_without_key_degrades(client):
    ids = [s["id"] for s in client.get("/api/songs").json()[:2]]
    res = client.post(f"/api/songs/{ids[0]}/explain", json={"other_id": ids[1]})
    assert res.status_code == 200
    assert res.json()["available"] is False and "GEMINI_API_KEY" in res.json()["message"]


def test_explain_prompt_uses_only_numbers():
    from app.explain import build_prompt
    a = {"title": "A", "artist": None, "networks": {"auditory": 0.5, "visual": -0.1},
         "percentiles": {"auditory": 90.0, "visual": 10.0}}
    b = {"title": "B", "artist": "X", "networks": {"auditory": 0.2, "visual": 0.3},
         "percentiles": {"auditory": 40.0, "visual": 80.0}}
    catalog = {"auditory": {"std": 0.1}, "visual": {"std": 0.2}}
    p = build_prompt(a, b, {"auditory": "Auditory", "visual": "Visual"}, {"brain": 0.81, "audio": 0.42}, catalog)
    assert "0.810" in p and "0.420" in p and "Do not invent" in p
    assert "3.00 catalog SDs" in p and "2.00 catalog SDs" in p  # |0.5-0.2|/0.1, |-0.1-0.3|/0.2
    assert "90th percentile" in p and "executive" not in p  # only roles of the networks present
    assert "early sound processing" in p


def test_network_percentiles(client):
    from app.main import app
    s = app.state.store
    top = int(s.net_means[:, 0].argmax())
    assert s.network_percentiles(top)["auditory"] == 100.0


def test_cover_missing_for_mock_wav(client):
    sid = client.get("/api/songs").json()[0]["id"]
    assert client.get(f"/api/songs/{sid}/cover").status_code == 404


class _FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents):
        self.calls.append(model)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return type("R", (), {"text": step})()


def _patch_client(monkeypatch, script):
    from google import genai
    fake = _FakeModels(script)
    monkeypatch.setattr(genai, "Client", lambda api_key: type("C", (), {"models": fake})())
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GEMINI_MODEL", "main-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "fallback-model")
    return fake


def _busy():
    from google.genai import errors
    return errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})


def test_generate_retries_then_falls_back(monkeypatch):
    from app import explain
    fake = _patch_client(monkeypatch, [_busy(), _busy(), _busy(), "ok text"])
    text, model = explain.generate("p", sleep=lambda s: None)
    assert (text, model) == ("ok text", "fallback-model")
    assert fake.calls == ["main-model"] * 3 + ["fallback-model"]


def test_generate_gives_up_when_all_busy(monkeypatch):
    from app import explain
    _patch_client(monkeypatch, [_busy()] * 6)
    with pytest.raises(explain.GeminiBusy):
        explain.generate("p", sleep=lambda s: None)


def test_generate_does_not_retry_client_errors(monkeypatch):
    from google.genai import errors
    from app import explain
    fake = _patch_client(monkeypatch, [errors.ClientError(400, {"error": {"code": 400, "message": "bad key", "status": "INVALID_ARGUMENT"}})])
    with pytest.raises(errors.ClientError):
        explain.generate("p", sleep=lambda s: None)
    assert fake.calls == ["main-model"]

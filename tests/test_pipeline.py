"""Fast checks: port parity, edge-graph parity, clip layout, API contract.

Run from the repo root:  python -m pytest -q tests
Needs the converted models in models/ (see README).
"""
import io
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import data_utils as du  # noqa: E402


@pytest.fixture(scope="module")
def clip():
    return np.random.default_rng(1).random((du.FRAME_SIZE, du.FRAME_SIZE, du.CLIP_LEN), dtype=np.float32)


@pytest.fixture(scope="module")
def keras_out(clip):
    tf_keras = pytest.importorskip("tf_keras")
    model = tf_keras.models.load_model("models/s_model.h5", compile=False)
    return model.predict(du.to_keras_batch(clip), verbose=0)[0, ..., 0]


def test_clip_layout_matches_original_resize():
    frames = [np.full((120, 160), i * 20, np.uint8) + np.arange(160, dtype=np.uint8) for i in range(10)]
    stack = np.array([du.preprocess_frame(f) for f in frames])
    legacy = stack.copy()
    legacy.resize(du.FRAME_SIZE, du.FRAME_SIZE, du.CLIP_LEN)  # what the original project did
    np.testing.assert_array_equal(du.frames_to_clip(frames), legacy)


def test_pytorch_port_matches_keras(clip, keras_out):
    torch = pytest.importorskip("torch")
    from pytorch_autoencoder import SpatioTemporalAutoencoder, load_keras_weights
    model = load_keras_weights(SpatioTemporalAutoencoder().eval(), "models/s_model.h5")
    with torch.no_grad():
        out = model(torch.from_numpy(du.to_torch_batch(clip)))[0, 0].numpy()
    assert np.abs(out - keras_out).max() < 1e-4


def test_edge_tflite_matches_keras(clip, keras_out):
    from benchmark import tflite_runner
    fp32 = tflite_runner("models/edge_fp32.tflite", 2)(clip)
    int8 = tflite_runner("models/edge_int8.tflite", 2)(clip)
    assert np.abs(fp32 - keras_out).max() < 1e-4
    ref = du.anomaly_score(clip, keras_out)
    assert abs(du.anomaly_score(clip, int8) - ref) / ref < 0.02


def test_api_contract():
    pytest.importorskip("fastapi")
    import cv2
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    ok, jpg = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (120, 160), dtype=np.uint8))
    img = jpg.tobytes()
    for i in range(du.CLIP_LEN - 1):
        r = client.post("/frame?stream_id=t", files={"file": ("f.jpg", io.BytesIO(img))}).json()
        assert r["decision"] == "buffering" and r["frames_buffered"] == i + 1
    r = client.post("/frame?stream_id=t", files={"file": ("f.jpg", io.BytesIO(img))}).json()
    assert r["decision"] in ("normal", "anomaly") and r["anomaly_score"] > 0
    r = client.post("/clip", files=[("files", ("f.jpg", io.BytesIO(img)))] * du.CLIP_LEN)
    assert r.status_code == 200
    assert client.post("/clip", files=[("files", ("f.jpg", io.BytesIO(img)))]).status_code == 400
    assert client.post("/frame", files={"file": ("x.jpg", io.BytesIO(b"nope"))}).status_code == 400

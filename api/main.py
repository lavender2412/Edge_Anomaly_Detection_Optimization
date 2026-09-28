"""FastAPI inference service for the int8 TFLite spatiotemporal autoencoder.

The model scores 10-frame clips. Two ways to call it:

  POST /frame?stream_id=cam1   one image per request (e.g. a camera pushing frames).
                               Frames are buffered per stream; once 10 are buffered,
                               every request scores the latest 10 (sliding window).
  POST /clip                   10 images in one multipart request, scored at once.

Response: anomaly_score (reconstruction MSE), threshold, decision (normal/anomaly).
"""
import json
import os
import sys
import threading
import time
from collections import OrderedDict, deque

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils as du  # noqa: E402

try:
    from ai_edge_litert.interpreter import Interpreter  # LiteRT (standalone TFLite runtime)
except ImportError:  # full TensorFlow as a fallback for local development
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = os.environ.get("MODEL_PATH", "models/edge_int8.tflite")
THRESHOLDS_PATH = os.environ.get("THRESHOLDS_PATH", "models/thresholds.json")
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))
MAX_STREAMS = int(os.environ.get("MAX_STREAMS", "32"))


def load_threshold():
    if "ANOMALY_THRESHOLD" in os.environ:
        return float(os.environ["ANOMALY_THRESHOLD"])
    with open(THRESHOLDS_PATH) as f:
        return float(json.load(f)[os.path.basename(MODEL_PATH)])


class Model:
    def __init__(self, path, threads):
        self.interp = Interpreter(model_path=path, num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]
        self.lock = threading.Lock()  # a TFLite interpreter is not thread-safe

    def score(self, clip):
        with self.lock:
            self.interp.set_tensor(self.inp, du.to_keras_batch(clip))
            self.interp.invoke()
            recon = self.interp.get_tensor(self.out)
        return du.anomaly_score(clip, recon)


app = FastAPI(title="Edge Anomaly Detection", version="1.0")
model = Model(MODEL_PATH, NUM_THREADS)
threshold = load_threshold()
streams = OrderedDict()  # stream_id -> deque of preprocessed frames (LRU-bounded)
streams_lock = threading.Lock()


async def decode(upload: UploadFile):
    data = np.frombuffer(await upload.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise HTTPException(400, f"could not decode image {upload.filename!r}")
    return du.preprocess_frame(frame)


def result(clip, **extra):
    t0 = time.perf_counter()
    score = model.score(clip)
    return {
        "anomaly_score": score,
        "threshold": threshold,
        "decision": "anomaly" if score > threshold else "normal",
        "inference_ms": round(1e3 * (time.perf_counter() - t0), 1),
        **extra,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH), "threshold": threshold,
            "clip_len": du.CLIP_LEN, "threads": NUM_THREADS}


@app.post("/frame")
async def frame(file: UploadFile = File(...), stream_id: str = Query("default", max_length=64)):
    processed = await decode(file)
    with streams_lock:
        buf = streams.pop(stream_id, None) or deque(maxlen=du.CLIP_LEN)
        buf.append(processed)
        streams[stream_id] = buf
        while len(streams) > MAX_STREAMS:
            streams.popitem(last=False)
        frames = list(buf)
    if len(frames) < du.CLIP_LEN:
        return {"decision": "buffering", "frames_buffered": len(frames), "clip_len": du.CLIP_LEN,
                "stream_id": stream_id}
    return result(du.clip_from_preprocessed(frames), stream_id=stream_id)


@app.post("/clip")
async def clip(files: list[UploadFile] = File(...)):
    if len(files) != du.CLIP_LEN:
        raise HTTPException(400, f"send exactly {du.CLIP_LEN} frames, got {len(files)}")
    frames = [await decode(f) for f in files]
    return result(du.clip_from_preprocessed(frames))


@app.delete("/frame")
def reset(stream_id: str = Query("default", max_length=64)):
    with streams_lock:
        streams.pop(stream_id, None)
    return {"stream_id": stream_id, "reset": True}

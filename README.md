# Edge Anomaly Detection Optimization

Getting a published video anomaly-detection model ready for CPU edge devices: porting it from TensorFlow to PyTorch, quantizing it to int8 in both frameworks, benchmarking every variant, and serving the fastest one from a small FastAPI + LiteRT container.

The source model is the spatiotemporal autoencoder from **"Suspicious Activity Detection in Recorded and Live Surveillance"**, presented at IEEE TENCON 2023 ([IEEE Xplore](https://ieeexplore.ieee.org/document/10322454), [original code](https://github.com/lavender2412/Suspicious_Activity_Detection)). It was trained on the [CUHK Avenue dataset](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) and flags a 10-frame clip as anomalous when its reconstruction error is high.

## Results

Measurements: 103 Avenue test clips (5 per test video), 1 CPU thread (a single-core edge device), median latency. The full output is in [`results/benchmark.md`](results/benchmark.md).

| Model | Precision | Size (MB) | Latency / clip (ms) | Latency / frame (ms) | Pearson r vs original | Mean score error | Decision agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| Keras (original) | fp32 | 12.87 * | 273 | 27.3 | 1.0000 | 0.00% | 100.0% |
| TFLite, direct conversion | int8 dyn-range | 5.15 | 374 | 37.4 | 1.0000 | 0.06% | 100.0% |
| TFLite, edge graph | fp32 | 4.52 | 431 | 43.1 | 1.0000 | 0.00% | 100.0% |
| **TFLite, edge graph** | **int8 dyn-range** | **1.34** | **118** | **11.8** | **1.0000** | **1.05%** | **100.0%** |
| PyTorch port | fp32 | 4.28 | 376 | 37.6 | 1.0000 | 0.00% | 100.0% |
| PyTorch `quantize_dynamic` | int8 dynamic | 4.28 | 372 | 37.2 | 1.0000 | 0.00% | 100.0% |
| PyTorch static PTQ | int8 static | 1.21 | 226 | 22.6 | 1.0000 | 5.85% | 100.0% |

\* The `.h5` file also stores the Adam optimizer state. The weights alone are 4.27 MB in fp32 (1,068,225 parameters).

Summary:

- **TFLite int8 (served model):** 70.4% smaller than the fp32 TFLite model (4.52 → 1.34 MB) and 89.6% smaller than the original `.h5`. Latency drops 3.7× against fp32 TFLite and 2.3× against the original Keras model. It makes the same normal/anomaly decision as the original on all 103 test clips.
- **PyTorch static int8:** 71.8% smaller and 1.7× faster than the PyTorch fp32 port, again with every decision unchanged.

How "decision agreement" is measured: each model gets its own threshold, set at the 95th percentile of its scores on 48 normal training clips. We then count how often its normal/anomaly decision on the test clips matches the original fp32 Keras model's decision. At this threshold the original model flags 47.6% of test clips, so agreement is tested on both classes, not just on easy "normal" clips. This measures how faithful the quantized models are to the original. It does not measure detection accuracy against ground truth, because Avenue's frame-level labels aren't in the dataset copy used here (see [Limitations](#limitations)).

## What turned up along the way

1. **The port is exact.** `pytorch_autoencoder.py` copies the trained Keras weights into a layer-for-layer PyTorch model. On the same input, the largest output difference is 4.5e-6. Keras' `ConvLSTM2D` treats the first non-batch axis as time, so on this model's `(B, H, W, T, C)` tensors it runs its recurrence over image rows, not frames. The port does the same, because that is what the published weights were trained on.
2. **The model's input layout comes from `ndarray.resize`.** The original pipeline turns a `(10, 227, 227)` frame stack into `(227, 227, 10)` with `ndarray.resize`, which reshapes memory rather than moving frames to the last axis. The weights only work with that layout. With it, normal clips score an MSE of about 0.01. With frames properly stacked on the last axis, the MSE is about 0.43. `data_utils.frames_to_clip` keeps the original layout on purpose, and `tests/` checks it.
3. **Rewriting the graph is where the TFLite savings come from.** Converting the Keras model directly shrinks it by only 27%, for two reasons: TFLite has no int8 kernel for `CONV_3D`, and each ConvLSTM becomes a `WHILE` loop whose zero-initialized TensorArrays are stored as about 2.7 MB of constants. But every Conv3D in this model has a `k×k×1` kernel, which is just a Conv2D applied to each frame. `convert_tflite.py` rewrites the model as Conv2D / Conv2DTranspose layers plus an unrolled ConvLSTM, using the same weights (it verifies a max error of 3.2e-6 against Keras). After that rewrite, every weight can be quantized: 70% smaller, and 3.7× faster.
4. **`torch.quantization.quantize_dynamic` does nothing here.** It only quantizes `Linear` / `LSTM` / `GRU` layers, and this model is all convolutions. Its row is in the table to show that. Static post-training quantization, calibrated on 32 training clips, is the approach that works.
5. **PyTorch bug: int8 `ConvTranspose` gives wrong results on the default `x86` engine** once there are 8 or more channels. The relative error is about 130%; it's reproducible with a single `nn.ConvTranspose2d(8, 4, 5, stride=2)`. The `fbgemm` engine gets it right, so the port uses `fbgemm`.
6. **Mixed precision beats all-int8 in PyTorch.** fbgemm's int8 kernels are slow for 1-channel convs: the int8 `deconv2` alone took about 1.5 s per clip, against 0.16 s in fp32. The first and last layers therefore stay fp32. They hold about 3% of the weights, and keeping them in fp32 turns a 4.5× slowdown into a 1.7× speedup.

## Repository layout

```
Edge_Anomaly_Detection_Optimization/
├── pytorch_autoencoder.py   # PyTorch port, Keras weight transfer, training, int8 quantization
├── convert_tflite.py        # TFLite conversion (direct + edge graph), int8 PTQ, size report
├── benchmark.py             # size / latency / fidelity table for all variants
├── data_utils.py            # preprocessing shared by every script and the API
├── api/
│   ├── main.py              # FastAPI service (LiteRT runtime)
│   ├── requirements.txt     # runtime-only dependencies
│   └── Dockerfile
├── models/                  # original .h5, converted/quantized models, calibrated thresholds
├── results/                 # benchmark output
└── tests/test_pipeline.py
```

## Reproducing

```bash
pip install -r requirements.txt

# Avenue volumes ship with the original project
git clone https://github.com/lavender2412/Suspicious_Activity_Detection ../Suspicious_Activity_Detection
ln -s ../Suspicious_Activity_Detection/data data        # or pass --data-dir / set AVENUE_DIR

python pytorch_autoencoder.py port --verify             # Keras -> PyTorch, checks parity
python pytorch_autoencoder.py quantize --method static  # PyTorch int8 (calibrated PTQ)
python pytorch_autoencoder.py quantize --method dynamic # quantize_dynamic, for comparison
python convert_tflite.py                                # TFLite fp32/int8, direct + edge graph
python benchmark.py                                     # results/ + models/thresholds.json
python -m pytest -q tests
```

`pytorch_autoencoder.py train --init scratch|keras` trains or fine-tunes the PyTorch model on the Avenue training volumes, using the original optimizer settings (Adam, lr 1e-3). The results above use the transferred published weights, not a retrained model.

## Inference API

The image contains only LiteRT, FastAPI and OpenCV, with no TensorFlow or PyTorch. Build it from the repository root and run it with CPU and memory limits to simulate an edge device:

```bash
docker build -f api/Dockerfile -t edge-anomaly .
docker run --rm -p 8000:8000 --cpus=1 --memory=512m edge-anomaly
```

The model scores 10-frame clips, so there are two ways to call it:

```bash
# Stream frames one at a time. After 10 frames, each request scores the latest 10 (sliding window).
curl -X POST "localhost:8000/frame?stream_id=cam1" -F file=@frame_0001.jpg

# Or send a whole clip in one request
curl -X POST localhost:8000/clip $(for f in clip/*.jpg; do echo -F files=@$f; done)
```

```json
{"anomaly_score": 0.0230, "threshold": 0.0206, "decision": "anomaly", "inference_ms": 118.3, "stream_id": "cam1"}
```

`GET /health` reports the model and threshold, and `DELETE /frame?stream_id=...` clears a stream's buffer. Environment variables: `NUM_THREADS` (default 1), `ANOMALY_THRESHOLD` (overrides the calibrated value), `MODEL_PATH`, `MAX_STREAMS`.

## Limitations

- **Fidelity, not accuracy.** Without Avenue's frame-level ground truth, the benchmark shows the int8 models make the same decisions as the fp32 original. It doesn't show how good those decisions are. The next step is to add the `testing_label_mask` files and report frame-level AUC for each model.
- **Crude threshold.** A percentile of training-clip scores flags about 48% of test clips, which is far above Avenue's real anomaly rate. Scores on test videos run slightly higher overall than on training videos. A threshold chosen on labelled data, or one normalized per video, would be more useful in production.
- **Deprecated PyTorch API.** Eager-mode `torch.ao.quantization` is deprecated in favor of `torchao`'s PT2E flow, and migrating is a natural follow-up. TFLite full-integer quantization (int8 activations, using a representative dataset) is the other one.

"""Benchmark original vs quantized models (TensorFlow/TFLite and PyTorch).

For each model:
  * size on disk
  * latency per 10-frame clip and per frame (median over the evaluation clips)
  * anomaly score = reconstruction MSE on Avenue test clips
  * fidelity to the original fp32 Keras model:
      - Pearson r of anomaly scores
      - mean relative score error
      - decision agreement: each model gets its own threshold, calibrated as
        a percentile of its scores on (normal) training clips, and we count how
        often its normal/anomaly decision matches the Keras fp32 decision.
        "Accuracy loss" = 100% - agreement.

The Avenue frame-level ground truth isn't in the dataset copy we use, so
fidelity is measured against the published fp32 model's decisions rather than
against labels.

Usage:
    python benchmark.py --data-dir /path/to/Avenue_Dataset [--threads 1]
"""
import argparse
import json
import os
import statistics
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np

import data_utils as du


# --------------------------------------------------------------------------- #
# Runners: each returns a callable clip(227,227,10) -> reconstruction (227,227,10)
# --------------------------------------------------------------------------- #
def keras_runner(path, threads):
    import tensorflow as tf
    import tf_keras
    model = tf_keras.models.load_model(path, compile=False)
    fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    return lambda clip: fn(tf.constant(du.to_keras_batch(clip))).numpy()[0, ..., 0]


def tflite_interpreter(path, threads):
    try:  # LiteRT, the standalone TFLite runtime
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=path, num_threads=threads)


def tflite_runner(path, threads):
    interp = tflite_interpreter(path, threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]

    def run(clip):
        interp.set_tensor(inp, du.to_keras_batch(clip))
        interp.invoke()
        return interp.get_tensor(out)[0, ..., 0]
    return run


def torch_runner(path, threads):
    import torch
    from pytorch_autoencoder import load_model
    torch.set_num_threads(threads)
    model, _ = load_model(path)

    def run(clip):
        with torch.inference_mode():
            return model(torch.from_numpy(du.to_torch_batch(clip)))[0, 0].numpy()
    return run


MODELS = [
    # name, framework, precision, path, runner
    ("Keras (original)", "TensorFlow", "fp32", "models/s_model.h5", keras_runner),
    ("TFLite direct", "TFLite", "int8 dyn-range", "models/keras_int8.tflite", tflite_runner),
    ("TFLite edge graph", "TFLite", "fp32", "models/edge_fp32.tflite", tflite_runner),
    ("TFLite edge graph", "TFLite", "int8 dyn-range", "models/edge_int8.tflite", tflite_runner),
    ("PyTorch port", "PyTorch", "fp32", "models/pytorch_fp32.pt", torch_runner),
    ("PyTorch quantize_dynamic", "PyTorch", "int8 dynamic", "models/pytorch_int8_dynamic.pt", torch_runner),
    ("PyTorch static PTQ", "PyTorch", "int8 static", "models/pytorch_int8_static.pt", torch_runner),
]


def evaluate(runner, clips, warmup):
    for clip in clips[:warmup]:
        runner(clip)
    scores, times = [], []
    for clip in clips:
        t0 = time.perf_counter()
        recon = runner(clip)
        times.append(time.perf_counter() - t0)
        scores.append(du.anomaly_score(clip, recon))
    return np.array(scores), times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=du.DEFAULT_DATA_DIR)
    p.add_argument("--train-clips-per-volume", type=int, default=3, help="for threshold calibration")
    p.add_argument("--test-clips-per-volume", type=int, default=5)
    p.add_argument("--percentile", type=float, default=95.0,
                   help="threshold = this percentile of a model's scores on training clips")
    p.add_argument("--threads", type=int, default=1, help="CPU threads per model (1 = single-core edge device)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    # TF threading must be configured before any TF op runs.
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("Loading clips ...", flush=True)
    train = [c for *_, c in du.iter_clips(args.data_dir, "training", clips_per_volume=args.train_clips_per_volume)]
    test = [c for *_, c in du.iter_clips(args.data_dir, "testing", clips_per_volume=args.test_clips_per_volume)]
    print(f"{len(train)} training clips (calibration), {len(test)} test clips, {args.threads} thread(s)\n")

    rows = []
    for name, fw, prec, path, make_runner in MODELS:
        if not os.path.exists(path):
            print(f"skip {name} {prec}: {path} missing (run convert_tflite.py / pytorch_autoencoder.py)")
            continue
        print(f"{name} [{prec}] ...", flush=True)
        run = make_runner(path, args.threads)
        train_scores, _ = evaluate(run, train, args.warmup)
        test_scores, times = evaluate(run, test, 0)
        rows.append(dict(model=name, framework=fw, precision=prec, path=path,
                         size_mb=os.path.getsize(path) / 1e6,
                         ms_per_clip=1e3 * statistics.median(times),
                         threshold=float(np.percentile(train_scores, args.percentile)),
                         scores=test_scores))

    ref = rows[0]
    assert ref["path"].endswith(".h5"), "the Keras reference model is required"
    ref_flags = ref["scores"] > ref["threshold"]
    for r in rows:
        flags = r["scores"] > r["threshold"]
        r["agreement_pct"] = 100.0 * float(np.mean(flags == ref_flags))
        r["accuracy_loss_pct"] = 100.0 - r["agreement_pct"]
        r["pearson_r"] = float(np.corrcoef(r["scores"], ref["scores"])[0, 1])
        r["rel_score_err_pct"] = 100.0 * float(np.mean(np.abs(r["scores"] - ref["scores"]) / ref["scores"]))
        r["anomaly_rate_pct"] = 100.0 * float(np.mean(flags))
        r["mean_score"] = float(np.mean(r["scores"]))

    header = ("| Model | Precision | Size (MB) | Latency / clip (ms) | Latency / frame (ms) | "
              "Mean MSE | Pearson r vs orig. | Rel. score err. | Decision agreement | Accuracy loss |")
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']} | {r['size_mb']:.2f} | {r['ms_per_clip']:.0f} | "
                     f"{r['ms_per_clip'] / du.CLIP_LEN:.1f} | {r['mean_score']:.5f} | {r['pearson_r']:.4f} | "
                     f"{r['rel_score_err_pct']:.2f}% | {r['agreement_pct']:.1f}% | {r['accuracy_loss_pct']:.1f}% |")
    table = "\n".join(lines)

    by = {r["path"]: r for r in rows}
    summary = []
    for base, q in [("models/edge_fp32.tflite", "models/edge_int8.tflite"),
                    ("models/pytorch_fp32.pt", "models/pytorch_int8_static.pt")]:
        if base in by and q in by:
            b, qq = by[base], by[q]
            summary.append(f"- {qq['model']} {qq['precision']}: size -{100 * (1 - qq['size_mb'] / b['size_mb']):.1f}%, "
                           f"latency {b['ms_per_clip']:.0f} -> {qq['ms_per_clip']:.0f} ms/clip "
                           f"({b['ms_per_clip'] / qq['ms_per_clip']:.2f}x), accuracy loss {qq['accuracy_loss_pct']:.1f}%")

    print("\n" + table + "\n")
    print("\n".join(summary))

    os.makedirs(args.out_dir, exist_ok=True)
    meta = dict(threads=args.threads, percentile=args.percentile, n_train_clips=len(train),
                n_test_clips=len(test), anomaly_rate_ref_pct=ref["anomaly_rate_pct"])
    with open(os.path.join(args.out_dir, "benchmark.md"), "w") as f:
        f.write(f"Avenue test clips: {len(test)}, calibration clips: {len(train)}, "
                f"threads: {args.threads}, threshold percentile: {args.percentile}\n\n")
        f.write(table + "\n\n" + "\n".join(summary) + "\n")
    with open(os.path.join(args.out_dir, "benchmark.json"), "w") as f:
        json.dump(dict(meta=meta, rows=[{k: v for k, v in r.items() if k != "scores"} for r in rows]), f, indent=2)
    # Thresholds for the API, calibrated per model.
    with open("models/thresholds.json", "w") as f:
        json.dump({os.path.basename(r["path"]): r["threshold"] for r in rows}, f, indent=2)
    print(f"\nWrote {args.out_dir}/benchmark.md, {args.out_dir}/benchmark.json, models/thresholds.json")


if __name__ == "__main__":
    main()

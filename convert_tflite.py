"""Convert the TENCON 2023 Keras autoencoder to TFLite with post-training int8 quantization.

Two graphs are converted from the same trained weights:

  keras  Direct conversion of the Keras model. The ConvLSTM2D layers become
         TFLite WHILE loops with large zero-initialised TensorArray constants,
         and CONV_3D kernels have no int8 path, so quantization gains little.

  edge   (default for serving) Same maths, rewritten for the TFLite runtime:
         every Conv3D / Conv3DTranspose in this model has a (k, k, 1) kernel, so
         it's exactly a Conv2D / Conv2DTranspose applied to each of the 10
         frames (frames go on the batch axis). The ConvLSTM recurrence is
         unrolled over its 26 steps. Every weight is then in a Conv2D-family op
         that TFLite can quantize.

Quantization is post-training dynamic-range int8 (tf.lite.Optimize.DEFAULT):
weights are stored as int8, activations stay float.

Usage:
    python convert_tflite.py                       # writes models/*.tflite and prints sizes
    python convert_tflite.py --keras-model path/to/s_model.h5 --out-dir models
"""
import argparse
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", message=r"(?s).*tf\.lite\.Interpreter is deprecated")

import numpy as np
import tensorflow as tf
import tf_keras

from data_utils import CLIP_LEN, FRAME_SIZE

INPUT_SPEC = tf.TensorSpec([1, FRAME_SIZE, FRAME_SIZE, CLIP_LEN, 1], tf.float32, name="clip")


def _hard_sigmoid(x):
    return tf.clip_by_value(0.2 * x + 0.5, 0.0, 1.0)  # Keras 2.x definition


class EdgeAutoencoder(tf.Module):
    """TFLite-friendly re-expression of the Keras model (identical outputs)."""

    def __init__(self, keras_model):
        super().__init__()
        L = {l.name: l for l in keras_model.layers}
        get = lambda name: [tf.Variable(w, trainable=False) for w in L[name].get_weights()]
        # Conv3D (k, k, 1, in, out) -> Conv2D (k, k, in, out)
        self.c1_k, self.c1_b = get("conv3d_1")
        self.c2_k, self.c2_b = get("conv3d_2")
        self.lstm = [get(n) for n in ("conv_lst_m2d_1", "conv_lst_m2d_2", "conv_lst_m2d_3")]
        self.d1_k, self.d1_b = get("conv3d_transpose_1")
        self.d2_k, self.d2_b = get("conv3d_transpose_2")

    @staticmethod
    def _conv_lstm(x, kernel, recurrent_kernel, bias):
        """x: (S, R, K, C) with the sequence on axis 0 -> (S, R, K, F)."""
        steps, rows, cols = x.shape[0], x.shape[1], x.shape[2]
        filters = recurrent_kernel.shape[-1] // 4
        gx = tf.nn.conv2d(x, kernel, 1, "SAME") + bias  # input conv for all steps at once
        h = tf.zeros([1, rows, cols, filters])
        c = tf.zeros([1, rows, cols, filters])
        outs = []
        for t in range(steps):
            z = gx[t:t + 1] + tf.nn.conv2d(h, recurrent_kernel, 1, "SAME")
            i, f, g, o = tf.split(z, 4, axis=-1)
            c = _hard_sigmoid(f) * c + _hard_sigmoid(i) * tf.tanh(g)
            h = _hard_sigmoid(o) * tf.tanh(c)
            outs.append(h)
        return tf.concat(outs, axis=0)

    @tf.function(input_signature=[INPUT_SPEC])
    def __call__(self, clip):
        x = tf.transpose(clip[0], [2, 0, 1, 3])  # (T=10, 227, 227, 1): frames as batch
        x = tf.tanh(tf.nn.conv2d(x, self.c1_k[:, :, 0], 4, "VALID") + self.c1_b)  # (10, 55, 55, 128)
        x = tf.tanh(tf.nn.conv2d(x, self.c2_k[:, :, 0], 2, "VALID") + self.c2_b)  # (10, 26, 26, 64)
        # Keras ConvLSTM2D on (B, H, W, T, C) recurs over H with a (W, T) image.
        x = tf.transpose(x, [1, 2, 0, 3])  # (H=26, W=26, T=10, C)
        for kernel, rec, bias in self.lstm:
            x = self._conv_lstm(x, kernel, rec, bias)
        x = tf.transpose(x, [2, 0, 1, 3])  # back to (T, H, W, C)
        x = tf.tanh(tf.nn.conv2d_transpose(x, self.d1_k[:, :, 0], [CLIP_LEN, 55, 55, 128], 2, "VALID")
                    + self.d1_b)
        x = tf.tanh(tf.nn.conv2d_transpose(x, self.d2_k[:, :, 0], [CLIP_LEN, FRAME_SIZE, FRAME_SIZE, 1], 4,
                                           "VALID") + self.d2_b)
        return tf.transpose(x, [1, 2, 0, 3])[None]  # (1, 227, 227, 10, 1)


def keras_concrete_fn(keras_model):
    @tf.function(input_signature=[INPUT_SPEC])
    def fn(clip):
        return keras_model(clip, training=False)
    return fn.get_concrete_function()


def to_tflite(concrete_fn, trackable, quantize):
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], trackable)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def run_tflite(model_content, clip_batch):
    interp = tf.lite.Interpreter(model_content=model_content)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], clip_batch)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"])


def mb(n):
    return f"{n / 1e6:6.2f} MB"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keras-model", default="models/s_model.h5")
    p.add_argument("--out-dir", default="models")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keras_model = tf_keras.models.load_model(args.keras_model, compile=False)
    n_params = keras_model.count_params()
    edge = EdgeAutoencoder(keras_model)

    x = np.random.default_rng(0).random(INPUT_SPEC.shape, dtype=np.float32)
    y_ref = keras_model.predict(x, verbose=0)
    diff = float(np.abs(edge(tf.constant(x)).numpy() - y_ref).max())
    print(f"Edge graph vs Keras (fp32, TF): max |diff| = {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("Edge graph does not match the Keras model")

    graphs = {"keras": (keras_concrete_fn(keras_model), keras_model),
              "edge": (edge.__call__.get_concrete_function(), edge)}
    sizes = {}
    for graph, (fn, obj) in graphs.items():
        for quant in (False, True):
            name = f"{graph}_{'int8' if quant else 'fp32'}"
            blob = to_tflite(fn, obj, quant)
            path = os.path.join(args.out_dir, f"{name}.tflite")
            with open(path, "wb") as f:
                f.write(blob)
            err = float(np.abs(run_tflite(blob, x) - y_ref).max())
            sizes[name] = len(blob)
            print(f"  wrote {path:32s} {mb(len(blob))}   max |diff| vs Keras = {err:.2e}")

    h5 = os.path.getsize(args.keras_model)
    print()
    print(f"Original .h5 file            : {mb(h5)}  (includes Adam optimizer state)")
    print(f"Weights only, fp32           : {mb(4 * n_params)}  ({n_params:,} params)")
    print(f"TFLite keras graph fp32/int8 : {mb(sizes['keras_fp32'])} -> {mb(sizes['keras_int8'])}  "
          f"({100 * (1 - sizes['keras_int8'] / sizes['keras_fp32']):.1f}% reduction)")
    print(f"TFLite edge graph  fp32/int8 : {mb(sizes['edge_fp32'])} -> {mb(sizes['edge_int8'])}  "
          f"({100 * (1 - sizes['edge_int8'] / sizes['edge_fp32']):.1f}% reduction)")
    print(f"Original .h5 -> edge int8    : {100 * (1 - sizes['edge_int8'] / h5):.1f}% smaller on disk")


if __name__ == "__main__":
    main()

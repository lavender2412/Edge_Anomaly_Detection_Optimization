"""PyTorch port of the TENCON 2023 spatiotemporal autoencoder + int8 quantization.

The architecture is a layer-for-layer port of models/s_model.h5 from
Suspicious_Activity_Detection (Keras 2.3.1):

    Conv3D(128, 11x11x1, s=4x4x1, tanh)  ->  Conv3D(64, 5x5x1, s=2x2x1, tanh)
    ConvLSTM2D(64) -> ConvLSTM2D(32) -> ConvLSTM2D(64)          (3x3, 'same')
    Conv3DTranspose(128, 5x5x1, s=2x2x1, tanh) -> Conv3DTranspose(1, 11x11x1, s=4x4x1, tanh)

Tensor layout: Keras sees (B, H, W, T, C). Here we use (B, C, H, W, T) so every
Conv3d kernel is (k, k, 1) exactly like the original. Keras' ConvLSTM2D
treats the first non-batch axis as the sequence, so on a (B, 26, 26, 10, C)
tensor it recurs over H (26 steps) with a 26x10 "image" of (W, T). We
reproduce that behaviour faithfully so the original trained weights can be
loaded and give the same outputs (see `port --verify`).

Subcommands:
    port      copy weights from the Keras .h5 into the PyTorch model (optionally verify parity)
    train     train / fine-tune on Avenue training volumes
    quantize  int8 quantization: 'static' (conv layers, calibrated) or 'dynamic'
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.ao import quantization as tq

import data_utils as du

# Quantized-kernel backend. Not "x86": in torch 2.x the x86 engine routes
# quantized ConvTranspose to oneDNN, which returns wrong results once
# in/out channels >= 8 (~130% relative error on this model's deconv1).
# fbgemm is correct for every layer here.
QUANT_ENGINE = "fbgemm"

# Eager-mode torch.ao.quantization is deprecated in favour of torchao and
# prints migration notices on every call; they are noise for this script.
warnings.filterwarnings("ignore", message=r"(?s).*(torchao|migrat|quantized tensor creation|reduce_range)")


def hard_sigmoid(x):
    # Keras 2.x definition (used by ConvLSTM2D's recurrent_activation).
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class ConvLSTM2D(nn.Module):
    """Keras-compatible ConvLSTM2D (return_sequences=True, padding='same').

    Input/output: (B, C, S, R, K) where S is the sequence axis.
    """

    def __init__(self, in_channels, filters, kernel_size=3, dropout=0.0, recurrent_dropout=0.0):
        super().__init__()
        self.filters = filters
        pad = kernel_size // 2
        self.input_conv = nn.Conv2d(in_channels, 4 * filters, kernel_size, padding=pad, bias=True)
        self.recurrent_conv = nn.Conv2d(filters, 4 * filters, kernel_size, padding=pad, bias=False)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    def _mask(self, like, p):
        # One mask per sequence, shared across timesteps (like Keras).
        # Keras draws a separate mask per gate; one shared mask is a close approximation.
        keep = 1.0 - p
        return torch.bernoulli(torch.full_like(like, keep)) / keep

    def forward(self, x):
        b, c, s, r, k = x.shape
        xs = x.permute(0, 2, 1, 3, 4)  # (B, S, C, R, K)
        if self.training and self.dropout > 0:
            xs = xs * self._mask(xs[:, :1], self.dropout)
        # The input convolution does not depend on the state: run it for all steps at once.
        gx = self.input_conv(xs.reshape(b * s, c, r, k)).reshape(b, s, 4 * self.filters, r, k)

        h = x.new_zeros(b, self.filters, r, k)
        cell = x.new_zeros(b, self.filters, r, k)
        rec_mask = None
        if self.training and self.recurrent_dropout > 0:
            rec_mask = self._mask(h, self.recurrent_dropout)
        outputs = []
        for t in range(s):
            hin = h * rec_mask if rec_mask is not None else h
            gates = gx[:, t] + self.recurrent_conv(hin)
            i, f, g, o = gates.chunk(4, dim=1)  # Keras gate order: i, f, c, o
            i, f, o = hard_sigmoid(i), hard_sigmoid(f), hard_sigmoid(o)
            cell = f * cell + i * torch.tanh(g)
            h = o * torch.tanh(cell)
            outputs.append(h)
        return torch.stack(outputs, dim=2)  # (B, F, S, R, K)


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 128, (11, 11, 1), stride=(4, 4, 1))
        self.conv2 = nn.Conv3d(128, 64, (5, 5, 1), stride=(2, 2, 1))
        self.lstm1 = ConvLSTM2D(64, 64, dropout=0.4, recurrent_dropout=0.3)
        self.lstm2 = ConvLSTM2D(64, 32, dropout=0.3)
        self.lstm3 = ConvLSTM2D(32, 64, dropout=0.5)
        self.deconv1 = nn.ConvTranspose3d(64, 128, (5, 5, 1), stride=(2, 2, 1))
        self.deconv2 = nn.ConvTranspose3d(128, 1, (11, 11, 1), stride=(4, 4, 1))

    def forward(self, x):  # x: (B, 1, 227, 227, 10)
        x = torch.tanh(self.conv1(x))    # (B, 128, 55, 55, 10)
        x = torch.tanh(self.conv2(x))    # (B, 64, 26, 26, 10)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = torch.tanh(self.deconv1(x))  # (B, 128, 55, 55, 10)
        return torch.tanh(self.deconv2(x))  # (B, 1, 227, 227, 10)


# --------------------------------------------------------------------------- #
# Keras -> PyTorch weight transfer
# --------------------------------------------------------------------------- #
def load_keras_weights(model, h5_path):
    """Copy weights from the original Keras .h5 (reads HDF5 directly, no TF needed)."""
    import h5py

    def w(f, layer, name):
        return torch.from_numpy(np.array(f["model_weights"][layer][layer][f"{name}:0"]))

    with h5py.File(h5_path, "r") as f:
        sd = {}
        # Conv3D kernel (k1, k2, k3, in, out) -> (out, in, k1, k2, k3)
        for tl, kl in [("conv1", "conv3d_1"), ("conv2", "conv3d_2")]:
            sd[f"{tl}.weight"] = w(f, kl, "kernel").permute(4, 3, 0, 1, 2)
            sd[f"{tl}.bias"] = w(f, kl, "bias")
        # ConvLSTM2D kernel (k, k, in, 4F) -> (4F, in, k, k)
        for tl, kl in [("lstm1", "conv_lst_m2d_1"), ("lstm2", "conv_lst_m2d_2"), ("lstm3", "conv_lst_m2d_3")]:
            sd[f"{tl}.input_conv.weight"] = w(f, kl, "kernel").permute(3, 2, 0, 1)
            sd[f"{tl}.input_conv.bias"] = w(f, kl, "bias")
            sd[f"{tl}.recurrent_conv.weight"] = w(f, kl, "recurrent_kernel").permute(3, 2, 0, 1)
        # Conv3DTranspose kernel (k1, k2, k3, out, in) -> (in, out, k1, k2, k3)
        for tl, kl in [("deconv1", "conv3d_transpose_1"), ("deconv2", "conv3d_transpose_2")]:
            sd[f"{tl}.weight"] = w(f, kl, "kernel").permute(4, 3, 0, 1, 2)
            sd[f"{tl}.bias"] = w(f, kl, "bias")
    model.load_state_dict({k: v.contiguous() for k, v in sd.items()})
    return model


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #
def _conv_paths(model):
    return [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d))]


def _wrap_convs_for_static_quant(model):
    """Wrap every conv in QuantWrapper (quant -> int8 conv -> dequant).

    The recurrent gate arithmetic (hard_sigmoid, tanh, mul, add) stays fp32,
    which keeps the LSTM state numerically stable while all conv weights and
    conv activations run in int8.
    """
    per_channel = tq.get_default_qconfig(QUANT_ENGINE)
    # Per-channel weight observers are not supported for ConvTranspose.
    per_tensor = tq.QConfig(activation=tq.HistogramObserver.with_args(reduce_range=True),
                            weight=tq.default_weight_observer)
    for path in _conv_paths(model):
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        conv = getattr(parent, attr)
        wrapped = tq.QuantWrapper(conv)
        wrapped.qconfig = per_tensor if isinstance(conv, nn.ConvTranspose3d) else per_channel
        setattr(parent, attr, wrapped)
    return model


def quantize_static(model, calibration_clips):
    torch.backends.quantized.engine = QUANT_ENGINE
    model = _wrap_convs_for_static_quant(model).eval()
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for clip in calibration_clips:
            model(torch.from_numpy(du.to_torch_batch(clip)))
    tq.convert(model, inplace=True)
    return model


def quantize_dynamic(model):
    """torch.quantization.quantize_dynamic, as in the original plan.

    Dynamic quantization only covers nn.Linear / nn.LSTM / nn.GRU. This model is
    all convolutions, so it's a no-op here, which is why `static` is the
    default. The benchmark keeps this row to show that.
    """
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8)


def load_model(path):
    """Load a checkpoint written by this script (fp32, static int8 or dynamic int8)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SpatioTemporalAutoencoder().eval()
    kind = ckpt.get("kind", "fp32")
    if kind == "int8_static":
        torch.backends.quantized.engine = QUANT_ENGINE
        model = _wrap_convs_for_static_quant(model)
        tq.prepare(model, inplace=True)
        tq.convert(model, inplace=True)
    elif kind == "int8_dynamic":
        model = quantize_dynamic(model)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), kind


def save_model(model, path, kind):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"kind": kind, "state_dict": model.state_dict()}, path)


def count_quantized(model):
    """Number of layers whose weights are stored in int8."""
    return sum(1 for m in model.modules()
               if type(m).__module__.startswith("torch.ao.nn.quantized") and hasattr(m, "weight"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cmd_port(args):
    model = load_keras_weights(SpatioTemporalAutoencoder().eval(), args.keras_model)
    save_model(model, args.out, "fp32")
    print(f"Ported {args.keras_model} -> {args.out} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    if args.verify:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tf_keras
        keras_model = tf_keras.models.load_model(args.keras_model, compile=False)
        rng = np.random.default_rng(0)
        clip = rng.random((du.FRAME_SIZE, du.FRAME_SIZE, du.CLIP_LEN), dtype=np.float32)
        y_tf = keras_model.predict(du.to_keras_batch(clip), verbose=0)[0, ..., 0]
        with torch.no_grad():
            y_pt = model(torch.from_numpy(du.to_torch_batch(clip)))[0, 0].numpy()
        diff = np.abs(y_tf - y_pt)
        print(f"Parity vs Keras: max |diff| = {diff.max():.2e}, mean |diff| = {diff.mean():.2e}")
        print(f"Anomaly score  : keras={du.anomaly_score(clip, y_tf):.6f} "
              f"torch={du.anomaly_score(clip, y_pt):.6f}")
        if diff.max() > 1e-3:
            raise SystemExit("Parity check FAILED")


def cmd_train(args):
    torch.manual_seed(args.seed)
    model = SpatioTemporalAutoencoder()
    if args.init == "keras":
        load_keras_weights(model, args.keras_model)
    elif args.init != "scratch":
        model.load_state_dict(load_model(args.init)[0].state_dict())
    # Same optimiser settings as the original Keras training config.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    loss_fn = nn.MSELoss()
    torch.set_num_threads(args.threads or torch.get_num_threads())

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, total = time.time(), 0, 0.0
        batch = []
        clips = du.iter_clips(args.data_dir, "training", max_clips=args.max_clips,
                              clips_per_volume=args.clips_per_volume)
        for _, _, clip in clips:
            batch.append(du.to_torch_batch(clip)[0])
            if len(batch) < args.batch_size:
                continue
            x = torch.from_numpy(np.stack(batch))
            batch = []
            opt.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            opt.step()
            seen += x.shape[0]
            total += loss.item() * x.shape[0]
            print(f"\repoch {epoch} clips {seen} loss {total / seen:.6f}", end="", flush=True)
        print(f"\repoch {epoch}: {seen} clips, mean loss {total / max(seen, 1):.6f}, "
              f"{time.time() - t0:.0f}s")
        save_model(model.eval(), args.out, "fp32")
    print(f"Saved {args.out}")


def cmd_quantize(args):
    model, kind = load_model(args.weights)
    if kind != "fp32":
        raise SystemExit(f"{args.weights} is already {kind}")
    if args.method == "static":
        calib = [c for _, _, c in du.iter_clips(args.data_dir, "training",
                                                max_clips=args.calib_clips, clips_per_volume=2)]
        print(f"Calibrating on {len(calib)} training clips ...")
        qmodel = quantize_static(model, calib)
        kind = "int8_static"
    else:
        qmodel = quantize_dynamic(model)
        kind = "int8_dynamic"
    save_model(qmodel, args.out, kind)
    src, dst = os.path.getsize(args.weights), os.path.getsize(args.out)
    print(f"{kind}: {count_quantized(qmodel)} quantized modules")
    print(f"Size: fp32 {src / 1e6:.2f} MB -> {kind} {dst / 1e6:.2f} MB "
          f"({max(0.0, 100 * (1 - dst / src)):.1f}% reduction)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("port", help="convert Keras .h5 weights to PyTorch")
    s.add_argument("--keras-model", default="models/s_model.h5")
    s.add_argument("--out", default="models/pytorch_fp32.pt")
    s.add_argument("--verify", action="store_true", help="compare outputs against Keras (needs tf_keras)")
    s.set_defaults(func=cmd_port)

    s = sub.add_parser("train", help="train or fine-tune on Avenue")
    s.add_argument("--data-dir", default=du.DEFAULT_DATA_DIR)
    s.add_argument("--init", default="scratch", help="'scratch', 'keras', or a .pt checkpoint")
    s.add_argument("--keras-model", default="models/s_model.h5")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--max-clips", type=int, default=None)
    s.add_argument("--clips-per-volume", type=int, default=None)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="models/pytorch_fp32.pt")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("quantize", help="int8 quantization")
    s.add_argument("--weights", default="models/pytorch_fp32.pt")
    s.add_argument("--method", choices=["static", "dynamic"], default="static")
    s.add_argument("--data-dir", default=du.DEFAULT_DATA_DIR)
    s.add_argument("--calib-clips", type=int, default=32)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_quantize)

    args = p.parse_args()
    if getattr(args, "out", "") is None:
        args.out = f"models/pytorch_int8_{args.method}.pt"
    args.func(args)


if __name__ == "__main__":
    main()

"""Shared preprocessing and data loading for the Avenue dataset.

The preprocessing mirrors the inference code of the original TENCON 2023
project (src/detection/activitydetection.py in Suspicious_Activity_Detection):
each frame is converted to grayscale, resized to 227x227, standardised
per-frame and clipped to [0, 1]. Ten consecutive frames form one clip.

Clip shape used across this repo: (227, 227, 10) float32, which is what
the Keras model expects once batch and channel axes are added:
(B, 227, 227, 10, 1). See frames_to_clip for how the frames are laid out.
"""
import glob
import os

import cv2
import numpy as np

FRAME_SIZE = 227
CLIP_LEN = 10
DEFAULT_DATA_DIR = os.environ.get("AVENUE_DIR", "data/Avenue_Dataset")


def preprocess_frame(frame):
    """Raw frame (gray or BGR, any size, uint8) -> (227, 227) float32 in [0, 1]."""
    frame = np.asarray(frame)
    if frame.ndim == 3:
        # Same luma weights as the original project.
        frame = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    frame = cv2.resize(frame.astype(np.float32), (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    frame = (frame - frame.mean()) / (frame.std() + 1e-8)
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def frames_to_clip(frames):
    """List of CLIP_LEN raw frames -> (227, 227, 10) clip, in the layout the model was trained on.

    The original pipeline builds a (10, 227, 227) stack and calls
    ndarray.resize((227, 227, 10)), which is a row-major *reshape*, not a
    transpose: each "time" slice of the result interleaves pixels from
    different frames. The published weights were trained on that layout.
    Feeding frames stacked on the last axis instead raises the
    reconstruction MSE on normal Avenue clips from ~0.01 to ~0.43, so we
    keep the original layout on purpose.
    """
    return clip_from_preprocessed([preprocess_frame(f) for f in frames])


def clip_from_preprocessed(frames):
    """CLIP_LEN outputs of preprocess_frame -> (227, 227, 10) clip (see frames_to_clip)."""
    if len(frames) != CLIP_LEN:
        raise ValueError(f"expected {CLIP_LEN} frames, got {len(frames)}")
    return np.stack(frames).reshape(FRAME_SIZE, FRAME_SIZE, CLIP_LEN)  # (10, 227, 227) -> (227, 227, 10)


def load_volume(mat_path):
    """Avenue *_vol/volXX.mat -> (T, H, W) uint8 array of grayscale frames."""
    import scipy.io as sio  # only needed for training/benchmarking, not in the API image

    vol = sio.loadmat(mat_path)["vol"]  # (120, 160, T)
    return np.transpose(vol, (2, 0, 1))


def volume_paths(data_dir=DEFAULT_DATA_DIR, split="training"):
    paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_vol", "vol*.mat")))
    if not paths:
        raise FileNotFoundError(
            f"No {split}_vol/*.mat files under {data_dir!r}. Point --data-dir (or "
            "AVENUE_DIR) at the Avenue_Dataset folder from Suspicious_Activity_Detection."
        )
    return paths


def iter_clips(data_dir=DEFAULT_DATA_DIR, split="training", max_clips=None, clips_per_volume=None):
    """Yield (volume_name, clip_index, clip) with non-overlapping 10-frame clips.

    clips_per_volume spreads a small evaluation budget evenly over all videos
    instead of taking every clip from the first one.
    """
    produced = 0
    for path in volume_paths(data_dir, split):
        vol = load_volume(path)
        n = len(vol) // CLIP_LEN
        idx = range(n)
        if clips_per_volume is not None and n > clips_per_volume:
            idx = np.linspace(0, n - 1, clips_per_volume).astype(int)
        name = os.path.splitext(os.path.basename(path))[0]
        for i in idx:
            yield name, int(i), frames_to_clip(vol[i * CLIP_LEN:(i + 1) * CLIP_LEN])
            produced += 1
            if max_clips is not None and produced >= max_clips:
                return


def to_keras_batch(clip):
    """(227, 227, 10) -> (1, 227, 227, 10, 1)."""
    return clip[None, ..., None].astype(np.float32)


def to_torch_batch(clip):
    """(227, 227, 10) -> (1, 1, 227, 227, 10) numpy; wrap with torch.from_numpy."""
    return clip[None, None].astype(np.float32)


def anomaly_score(x, y):
    """Reconstruction MSE between input clip and reconstruction (any matching shapes)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return float(np.mean((x - y) ** 2))

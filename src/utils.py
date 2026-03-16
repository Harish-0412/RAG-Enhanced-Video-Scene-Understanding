"""
utils.py — Shared utility functions for the Video-RAG pipeline.

Covers:
  • Directory management
  • JSON metadata persistence
  • Visual hash deduplication  (perceptual hashing via imagehash)
  • Adaptive bitrate downsampling  (OpenCV resize before processing)
  • Structured console logging
"""

import os
import json
import logging
from pathlib import Path

import cv2
import numpy as np

# ── Optional dependency: imagehash (pip install imagehash Pillow) ──────────────
try:
    import imagehash
    from PIL import Image
    _HASH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HASH_AVAILABLE = False

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("video_rag")


# ── 1. Directory helpers ───────────────────────────────────────────────────────

def ensure_dirs(paths: list[str]) -> None:
    """Create directories (and all parents) if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    log.debug("Directories ready: %s", paths)


# ── 2. Metadata persistence ────────────────────────────────────────────────────

def save_metadata(data: list[dict], filepath: str) -> None:
    """
    Persist the frame-to-timestamp mapping as a pretty-printed JSON file.

    Parameters
    ----------
    data     : List of scene-metadata dicts.
    filepath : Destination path (parent directory must exist).
    """
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    log.info("✅  Metadata saved → %s  (%d records)", filepath, len(data))


def load_metadata(filepath: str) -> list[dict]:
    """Load and return a previously saved mapping.json."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── 3. Adaptive Bitrate Downsampling ──────────────────────────────────────────

def downsample_frame(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Adaptive Bitrate Downsampling — resize a frame so its width ≤ max_width
    while preserving the aspect ratio.

    Reducing resolution before hashing / embedding dramatically cuts CPU/RAM
    overhead without losing semantic features (text, colour blocks, faces).

    Parameters
    ----------
    frame     : BGR numpy array from cv2.
    max_width : Target width ceiling (default 640 px).

    Returns
    -------
    Resized BGR numpy array (or original if already within limit).
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


# ── 4. Visual Hash Deduplication ──────────────────────────────────────────────

class VisualDeduplicator:
    """
    Visual Hash Deduplication using perceptual hashing (pHash).

    When a video lingers on a static slide for several minutes the scene
    detector still emits multiple cuts. This class ensures each unique
    visual is stored *once* by comparing perceptual hashes.

    Hash distance ≤ `threshold` Hamming bits → frames are considered
    duplicates and the second is discarded.

    Usage
    -----
    dedup = VisualDeduplicator(threshold=8)
    if not dedup.is_duplicate(frame):
        # save the frame
    """

    def __init__(self, threshold: int = 8) -> None:
        if not _HASH_AVAILABLE:
            raise RuntimeError(
                "imagehash + Pillow are required for deduplication.\n"
                "Install with:  pip install imagehash Pillow"
            )
        self.threshold = threshold
        self._seen_hashes: list[imagehash.ImageHash] = []

    def _to_pil(self, frame: np.ndarray) -> "Image.Image":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def compute_hash(self, frame: np.ndarray) -> "imagehash.ImageHash":
        """Return the perceptual hash of a BGR frame."""
        return imagehash.phash(self._to_pil(frame))

    def is_duplicate(self, frame: np.ndarray) -> bool:
        """
        Return True if *frame* is visually too similar to any previously
        accepted frame (Hamming distance ≤ threshold).

        If not a duplicate, register the frame's hash for future checks.
        """
        h = self.compute_hash(frame)
        for seen in self._seen_hashes:
            if (h - seen) <= self.threshold:
                return True
        self._seen_hashes.append(h)
        return False

    def reset(self) -> None:
        """Clear all stored hashes (e.g. between videos)."""
        self._seen_hashes.clear()


# ── 5. Inter-frame Difference (SAD) helper ────────────────────────────────────

def sum_absolute_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Inter-frame Difference Threshold — Sum of Absolute Differences (SAD).

    Converts both frames to greyscale and computes the mean-normalised SAD.
    The result is comparable regardless of frame resolution.

    Returns
    -------
    Float in [0, 255]; higher ⟹ more change between frames.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(gray_a - gray_b)))

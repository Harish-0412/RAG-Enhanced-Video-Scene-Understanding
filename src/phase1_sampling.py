"""
src/phase1_sampling.py — Phase 1: Temporal Segmentation & Frame Extraction
============================================================================

Pipeline overview
-----------------
Step A │ Temporal Segmentation   — content-aware scene detection via PySceneDetect
Step B │ Center-Frame Strategy   — extract the middle frame of each scene to avoid
       │                           transition blur at cut boundaries
Step C │ Metadata Mapping        — produce mapping.json consumed by the RAG retriever

Advanced techniques employed
-----------------------------
• Adaptive Bitrate Downsampling  — video is analysed at reduced resolution (↓ CPU/RAM)
• Visual Hash Deduplication      — perceptual hashing (pHash) discards near-duplicate
                                   frames when a slide stays static for multiple scenes
• Inter-frame Difference (SAD)   — Sum of Absolute Differences logged per scene for
                                   downstream filtering / thresholding
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import cv2
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector, AdaptiveDetector

# Local utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    ensure_dirs,
    save_metadata,
    downsample_frame,
    VisualDeduplicator,
    sum_absolute_difference,
    log,
)

# ── Configuration ──────────────────────────────────────────────────────────────

# Resolve paths relative to the repository root (not the current working directory).
REPO_ROOT         = Path(__file__).resolve().parent.parent
DATA_ROOT         = REPO_ROOT / "data"

VIDEO_PATH        = str(DATA_ROOT / "input_videos" / "demo_video.mp4")
OUTPUT_FRAME_DIR  = str(DATA_ROOT / "processed" / "frames")
METADATA_PATH     = str(DATA_ROOT / "processed" / "metadata" / "mapping.json")

# Scene detection
DETECTOR          = "adaptive"   # "adaptive" | "content"
CONTENT_THRESHOLD = 27.0         # Inter-frame Difference Threshold for ContentDetector
ADAPTIVE_DELTA    = 3.0          # AdaptiveDetector sensitivity (lower = more cuts)

# Downsampling — analyse video at this width to reduce CPU/RAM load
ANALYSIS_WIDTH    = 640          # px  (Adaptive Bitrate Downsampling)

# Visual Hash Deduplication — Hamming distance ≤ this value → duplicate
DEDUP_THRESHOLD   = 8            # bits  (0 = exact match, 10 = near-match)
ENABLE_DEDUP      = True

# Output image quality
JPEG_QUALITY      = 92           


# ── Step A: Temporal Segmentation (Scene Detection) ───────────────────────────

def detect_scenes(video_path: str) -> list:
    """
    Temporal Segmentation via content-aware scene detection.

    Uses PySceneDetect's AdaptiveDetector (preferred) or ContentDetector.
    Both operate on Luma (brightness) and Hue (colour) frame differences —
    exactly the Inter-frame Difference Threshold principle (SAD-based).

    Returns
    -------
    List of (start_timecode, end_timecode) FrameTimecode pairs.
    """
    log.info("🎬  Opening video: %s", video_path)
    video = open_video(video_path)

    scene_manager = SceneManager()

    if DETECTOR == "adaptive":
        # AdaptiveDetector normalises the SAD against the local rolling average,
        # making it robust to overall brightness shifts (e.g., screen dimming).
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=ADAPTIVE_DELTA))
        log.info("🔍  Detector: AdaptiveDetector  (delta=%.1f)", ADAPTIVE_DELTA)
    else:
        scene_manager.add_detector(ContentDetector(threshold=CONTENT_THRESHOLD))
        log.info("🔍  Detector: ContentDetector  (threshold=%.1f)", CONTENT_THRESHOLD)

    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    log.info("✂️   Detected %d scene(s).", len(scene_list))
    return scene_list


# ── Step B: Center-Frame Extraction ───────────────────────────────────────────

def extract_center_frame(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
) -> tuple[bool, any]:
    """
    Center-Frame Strategy: seek to the midpoint of [start_frame, end_frame].

    The first and last frames of a scene often contain motion blur from the
    cut itself. The center frame is statistically the most stable, representative
    image of the scene's visual content.

    Returns
    -------
    (success: bool, frame: np.ndarray | None)
    """
    mid_frame = (start_frame + end_frame) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    return ret, frame if ret else None


# ── Step C: Metadata Mapping ──────────────────────────────────────────────────

def build_metadata_record(
    scene_index: int,
    frame_filename: str,
    start_tc,
    end_tc,
    scene_score: float,
    sad_score: float,
    dedup_skipped: bool = False,
    center_frame_tc = None,
) -> dict:
    """
    Construct a single metadata record for mapping.json.

    Fields
    ------
    frame_id       : Filename of the saved JPEG.
    timestamp      : Human-readable start timecode ("HH:MM:SS.mmm").
    center_timestamp: Human-readable center frame timecode ("HH:MM:SS.mmm").
    start_seconds  : Scene start in fractional seconds.
    end_seconds    : Scene end in fractional seconds.
    center_seconds : Center frame time in fractional seconds.
    duration_sec   : Scene duration in seconds.
    scene_score    : PySceneDetect content-change score at the cut boundary.
    sad_score      : Mean normalised SAD between the previous & current center frames.
    dedup_skipped  : True if the frame was kept but flagged as a near-duplicate.
    """
    return {
        "frame_id":      frame_filename,
        "timestamp":     start_tc.get_timecode(),        # "00:01:23.456"
        "center_timestamp": center_frame_tc.get_timecode() if center_frame_tc else start_tc.get_timecode(),
        "start_seconds": round(start_tc.get_seconds(), 3),
        "end_seconds":   round(end_tc.get_seconds(),   3),
        "center_seconds": round(center_frame_tc.get_seconds(), 3) if center_frame_tc else round(start_tc.get_seconds(), 3),
        "duration_sec":  round(end_tc.get_seconds() - start_tc.get_seconds(), 3),
        "scene_score":   round(scene_score, 2),
        "sad_score":     round(sad_score, 2),
        "dedup_skipped": dedup_skipped,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_video(video_path: str = VIDEO_PATH) -> list[dict]:
    """
    Full Phase 1 pipeline:

      1. Setup output directories.
      2. Temporal segmentation (Step A).
      3. Center-frame extraction with adaptive downsampling (Step B).
      4. Visual hash deduplication to skip static slides.
      5. Metadata mapping and JSON export (Step C).

    Returns
    -------
    List of metadata dicts (same content as mapping.json).
    """
    # Ensure paths are resolved relative to the repository root, not the
    # current working directory. This makes the script runnable from anywhere.
    video_path = str((REPO_ROOT / video_path) if not Path(video_path).is_absolute() else Path(video_path))
    if not Path(video_path).exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            f"Expected to find the video at the path above.\n"
            f"Tip: run this script from the repo root or pass --video with an absolute path."
        )
    # ── 1. Directories ──────────────────────────────────────────────────────
    ensure_dirs([OUTPUT_FRAME_DIR, str(Path(METADATA_PATH).parent)])

    # ── 2. Scene detection (Step A) ─────────────────────────────────────────
    scene_list = detect_scenes(video_path)
    if not scene_list:
        log.warning("No scenes detected. Check video path or lower the threshold.")
        return []

    # ── 3. Frame extraction (Step B) ────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    deduplicator = VisualDeduplicator(threshold=DEDUP_THRESHOLD) if ENABLE_DEDUP else None
    metadata:      list[dict] = []
    saved_count    = 0
    skipped_dedup  = 0
    prev_frame     = None   # used for SAD computation

    for i, (start_tc, end_tc) in enumerate(scene_list):
        start_frame = start_tc.get_frames()
        end_frame   = end_tc.get_frames()

        # Center-frame strategy
        ok, raw_frame = extract_center_frame(cap, start_frame, end_frame)
        if not ok:
            log.warning("Scene %03d: failed to read frame — skipping.", i + 1)
            continue

        # Calculate center frame timestamp for accurate metadata
        # Use the existing timecode objects to ensure consistency
        start_seconds = start_tc.get_seconds()
        end_seconds = end_tc.get_seconds()
        center_seconds = (start_seconds + end_seconds) / 2
        center_tc = FrameTimecode(center_seconds, start_tc.framerate)

        # Adaptive Bitrate Downsampling for hashing / SAD (not for saved JPEG)
        small_frame = downsample_frame(raw_frame, max_width=ANALYSIS_WIDTH)

        # ── Visual Hash Deduplication ────────────────────────────────────
        is_dup = False
        if deduplicator is not None and deduplicator.is_duplicate(small_frame):
            is_dup = True
            skipped_dedup += 1
            log.debug("Scene %03d: duplicate detected — skipping save.", i + 1)
            # For duplicates, use a placeholder filename that indicates it's a duplicate
            # This maintains the one-to-one mapping between scenes and metadata records
            frame_filename = f"scene_{i + 1:03d}_duplicate.jpg"
        else:
            # ── Save the full-resolution center frame ────────────────────
            frame_filename = f"frame_{saved_count + 1:03d}.jpg"
            save_path = os.path.join(OUTPUT_FRAME_DIR, frame_filename)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            cv2.imwrite(save_path, raw_frame, encode_params)
            saved_count += 1

        # ── SAD vs previous frame ────────────────────────────────────────
        sad = sum_absolute_difference(prev_frame, small_frame) if prev_frame is not None else 0.0
        prev_frame = small_frame

        # PySceneDetect doesn't expose a per-scene score post-hoc, so we use
        # SAD as the scene_score proxy (consistent with ContentDetector internals)
        scene_score = sad

        # ── Step C: Build metadata record ───────────────────────────────
        record = build_metadata_record(
            scene_index=i + 1,
            frame_filename=frame_filename,
            start_tc=start_tc,
            end_tc=end_tc,
            scene_score=scene_score,
            sad_score=sad,
            dedup_skipped=is_dup,
            center_frame_tc=center_tc,
        )
        metadata.append(record)

        log.info(
            "Scene %03d | %s → %s | SAD=%.1f | %s",
            i + 1,
            start_tc.get_timecode(),
            end_tc.get_timecode(),
            sad,
            "SKIP(dup)" if is_dup else f"SAVED → {frame_filename}",
        )

    cap.release()

    # ── 4. Save metadata (Step C) ────────────────────────────────────────────
    save_metadata(metadata, METADATA_PATH)

    log.info(
        "🚀  Phase 1 complete | %d scenes | %d frames saved | %d duplicates skipped",
        len(scene_list),
        saved_count,
        skipped_dedup,
    )
    return metadata


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Video temporal segmentation")
    parser.add_argument(
        "--video", default=VIDEO_PATH,
        help=f"Path to input video (default: {VIDEO_PATH})",
    )
    parser.add_argument(
        "--detector", choices=["adaptive", "content"], default=DETECTOR,
        help="Scene detector algorithm",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable visual hash deduplication",
    )
    args = parser.parse_args()

    # Override module-level config from CLI flags
    DETECTOR     = args.detector        # noqa: F841
    ENABLE_DEDUP = not args.no_dedup    # noqa: F841

    process_video(video_path=args.video)

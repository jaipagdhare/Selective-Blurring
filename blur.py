#!/usr/bin/env python3
"""
Blur strangers in video; keep faces matching the allowlist gallery (Faces/).

  python build_gallery.py
  python blur.py --input try.mp4
  python tune_blur.py --input try.mp4   # find better parameters
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from blur_config import BlurConfig
from face_blur_utils import (
    FaceTrack,
    build_gallery_from_dir,
    create_face_app,
    expand_bbox,
    gaussian_blur_roi,
    init_track,
    load_gallery,
    match_detections_to_tracks,
    max_similarity,
    resolve_faces_dir,
    save_gallery,
)

DEFAULT_CONFIG = BlurConfig()


def process_detections(
    app,
    frame_bgr: np.ndarray,
    gallery: np.ndarray,
    cfg: BlurConfig,
) -> List[Tuple[Tuple[float, float, float, float], float]]:
    """Return list of (bbox, max_gallery_similarity)."""
    faces = app.get(frame_bgr)
    out: List[Tuple[Tuple[float, float, float, float], float]] = []

    for face in faces:
        if face.det_score is not None and face.det_score < cfg.min_det_score:
            continue
        x1, y1, x2, y2 = face.bbox.astype(float)
        if (x2 - x1) < cfg.min_face_px or (y2 - y1) < cfg.min_face_px:
            out.append(((x1, y1, x2, y2), 0.0))
            continue
        if face.normed_embedding is None:
            out.append(((x1, y1, x2, y2), 0.0))
            continue
        sim = max_similarity(gallery, face.normed_embedding)
        out.append(((x1, y1, x2, y2), sim))
    return out


def apply_blur_for_tracks(
    frame_bgr: np.ndarray,
    tracks: List[FaceTrack],
    cfg: BlurConfig,
    *,
    use_predicted: bool,
) -> None:
    h, w = frame_bgr.shape[:2]
    for tr in tracks:
        if tr.allow:
            continue
        bbox = tr.predict() if use_predicted else tr.bbox
        x1, y1, x2, y2 = expand_bbox(*bbox, cfg.bbox_pad, w, h)
        gaussian_blur_roi(frame_bgr, x1, y1, x2, y2, cfg.blur_kernel)


def run_video(
    app,
    gallery: np.ndarray,
    input_path: str,
    output_path: str,
    cfg: BlurConfig = DEFAULT_CONFIG,
    *,
    print_every: int = 50,
    max_frames: int = 0,
) -> dict:
    """Process video; return summary stats."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps and fps > 0 else 30

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Frames : {length}  |  {width}x{height}  @ {fps} fps")
    print(f"Config : {cfg.summary()}\n")

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    FaceTrack.reset_ids()
    tracks: List[FaceTrack] = []
    frame_idx = 0
    count = 0
    total_flips = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if max_frames > 0 and count >= max_frames:
                break

            frame_idx += 1
            count += 1
            if count == 1 or count % print_every == 0:
                pct = f" ({100.0 * count / length:.1f}%)" if length > 0 else ""
                print(f"  Frame {count}/{length}{pct}  tracks={len(tracks)}")

            do_detect = (frame_idx % cfg.detect_every == 1) or not tracks

            if do_detect:
                detections = process_detections(app, frame_bgr, gallery, cfg)
                unmatched = match_detections_to_tracks(
                    detections,
                    tracks,
                    iou_thresh=cfg.iou_match,
                    allow_thresh=cfg.similarity_allow,
                    blur_thresh=cfg.similarity_blur,
                    instant_allow=cfg.instant_allow,
                    ema_alpha=cfg.sim_ema_alpha,
                    allow_votes=cfg.allow_votes,
                    vote_window=cfg.vote_window,
                )
                for bbox, sim in unmatched:
                    tracks.append(
                        init_track(
                            bbox,
                            sim,
                            allow_thresh=cfg.similarity_allow,
                            blur_thresh=cfg.similarity_blur,
                            instant_allow=cfg.instant_allow,
                            ema_alpha=cfg.sim_ema_alpha,
                            allow_votes=cfg.allow_votes,
                            vote_window=cfg.vote_window,
                        )
                    )
            else:
                for tr in tracks:
                    tr.missed += 1
                    x1, y1, x2, y2 = tr.predict()
                    tr.bbox = (x1, y1, x2, y2)

            tracks = [t for t in tracks if t.missed <= cfg.max_missed]
            apply_blur_for_tracks(
                frame_bgr, tracks, cfg, use_predicted=not do_detect
            )
            out.write(frame_bgr)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        total_flips = sum(t.allow_flips for t in tracks)
        print(f"\nProcessed {count} frame(s).  allow-state flips={total_flips}")
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"Done → {output_path}")
    return {"frames": count, "allow_flips": total_flips, "config": cfg.slug()}


def ensure_gallery(gallery_path: str, faces_dir: str, model: str, rebuild: bool) -> np.ndarray:
    if os.path.isfile(gallery_path) and not rebuild:
        gallery, sources = load_gallery(gallery_path)
        print(f"Loaded gallery '{gallery_path}' ({len(sources)} embedding(s))")
        return gallery

    print(f"Building gallery from '{faces_dir}' ...")
    app = create_face_app(model)
    embeddings, sources = build_gallery_from_dir(app, faces_dir)
    save_gallery(gallery_path, embeddings, sources)
    return embeddings


def config_from_args(args) -> BlurConfig:
    blur_thresh = max(0.15, args.threshold - args.blur_margin)
    return BlurConfig(
        similarity_allow=args.threshold,
        similarity_blur=blur_thresh,
        instant_allow=args.instant_allow,
        detect_every=args.detect_every,
        allow_votes=args.allow_votes,
        vote_window=args.vote_window,
        max_missed=args.max_missed,
        blur_kernel=args.blur,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blur all faces except allowlisted people (Faces/ gallery)."
    )
    parser.add_argument("--input", "-i", default="try.mp4")
    parser.add_argument("--output", "-o", default="")
    parser.add_argument("--gallery", default="gallery.npz")
    parser.add_argument("--faces-dir", default="Faces")
    parser.add_argument("--rebuild-gallery", action="store_true")
    parser.add_argument("--model", default="buffalo_sc")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIG.similarity_allow,
        help="Cosine sim to allow (keep sharp)",
    )
    parser.add_argument(
        "--blur-margin",
        type=float,
        default=DEFAULT_CONFIG.similarity_allow - DEFAULT_CONFIG.similarity_blur,
        help="allow - blur_thresh hysteresis gap",
    )
    parser.add_argument(
        "--instant-allow",
        type=float,
        default=DEFAULT_CONFIG.instant_allow,
        help="Immediate allow if sim >= this",
    )
    parser.add_argument("--detect-every", type=int, default=DEFAULT_CONFIG.detect_every)
    parser.add_argument("--allow-votes", type=int, default=DEFAULT_CONFIG.allow_votes)
    parser.add_argument("--vote-window", type=int, default=DEFAULT_CONFIG.vote_window)
    parser.add_argument("--max-missed", type=int, default=DEFAULT_CONFIG.max_missed)
    parser.add_argument("--blur", type=int, default=DEFAULT_CONFIG.blur_kernel)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--config",
        default="",
        help="JSON file from tune_blur.py (e.g. tune_results/best_config.json)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or f"{os.path.splitext(args.input)[0]} blurred.mp4"
    faces_dir = resolve_faces_dir(args.faces_dir)
    gallery = ensure_gallery(args.gallery, faces_dir, args.model, args.rebuild_gallery)

    if args.config:
        with open(args.config) as f:
            cfg = BlurConfig(**json.load(f))
        print(f"Loaded config from {args.config}")
    else:
        cfg = config_from_args(args)

    print("Loading face model ...")
    app = create_face_app(args.model)
    run_video(app, gallery, args.input, output, cfg, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

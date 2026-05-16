#!/usr/bin/env python3
"""
Sweep blur parameters on sample frames and rank configs by stability + accuracy.

  python tune_blur.py --input try.mp4
  python tune_blur.py --input try.mp4 --full-grid --render-top 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List

import cv2
import numpy as np

from blur import process_detections, run_video
from blur_config import BlurConfig, all_tuning_configs, iterate_configs
from face_blur_utils import (
    FaceTrack,
    create_face_app,
    init_track,
    load_gallery,
    match_detections_to_tracks,
)


def simulate_config(
    app,
    gallery: np.ndarray,
    frames: List[np.ndarray],
    cfg: BlurConfig,
) -> Dict[str, float]:
    """
    Run pipeline on frames without writing video.
    Scores (lower is better except friend_sharp_mean):
      - flicker: allow-state changes per track per frame
      - friend_false_blur: high-sim track blurred (sim_ema >= allow)
      - stranger_false_clear: low-sim track kept sharp
    """
    FaceTrack.reset_ids()
    tracks: List[FaceTrack] = []
    frame_idx = 0

    flicker = 0
    friend_frames = 0
    friend_blurred_frames = 0
    stranger_frames = 0
    stranger_clear_frames = 0
    for frame_bgr in frames:
        frame_idx += 1
        do_detect = (frame_idx % cfg.detect_every == 1) or not tracks

        if do_detect:
            detections = process_detections(app, frame_bgr, gallery, cfg)
            prev_flips = sum(t.allow_flips for t in tracks)
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
            flicker += sum(t.allow_flips for t in tracks) - prev_flips
        else:
            for tr in tracks:
                tr.missed += 1
                x1, y1, x2, y2 = tr.predict()
                tr.bbox = (x1, y1, x2, y2)

        tracks = [t for t in tracks if t.missed <= cfg.max_missed]

        for tr in tracks:
            if tr.sim_ema >= cfg.similarity_allow:
                friend_frames += 1
                if not tr.allow:
                    friend_blurred_frames += 1
            elif tr.sim_ema < cfg.similarity_blur:
                stranger_frames += 1
                if tr.allow:
                    stranger_clear_frames += 1

    n_frames = len(frames)
    return {
        "flicker": float(flicker),
        "flicker_per_frame": flicker / max(1, n_frames),
        "friend_false_blur_rate": friend_blurred_frames / max(1, friend_frames),
        "stranger_false_clear_rate": stranger_clear_frames / max(1, stranger_frames),
        "friend_frames": float(friend_frames),
        "stranger_frames": float(stranger_frames),
    }


def composite_score(metrics: Dict[str, float]) -> float:
    """Lower is better."""
    return (
        metrics["flicker_per_frame"] * 3.0
        + metrics["friend_false_blur_rate"] * 8.0
        + metrics["stranger_false_clear_rate"] * 4.0
    )


def load_sample_frames(path: str, max_frames: int, stride: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"No frames in {path}")

    indices = list(range(0, total, stride))[:max_frames]
    frames: List[np.ndarray] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} sample frame(s) from {path} (stride={stride})")
    return frames


def iterate_and_score(
    app,
    gallery: np.ndarray,
    frames: List[np.ndarray],
    configs: List[BlurConfig],
) -> List[dict]:
    results: List[dict] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.slug()}")
        metrics = simulate_config(app, gallery, frames, cfg)
        score = composite_score(metrics)
        row = {
            "slug": cfg.slug(),
            "score": score,
            **metrics,
            **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
        }
        results.append(row)
    results.sort(key=lambda r: r["score"])
    return results


def save_results(results: List[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tune_results.csv")
    json_path = os.path.join(out_dir, "tune_results.json")

    if not results:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def print_top(results: List[dict], n: int = 5) -> None:
    print(f"\n{'='*60}\nTop {n} configurations (lower score = better)\n{'='*60}")
    for i, r in enumerate(results[:n], 1):
        print(
            f"\n#{i} score={r['score']:.4f}  {r['slug']}\n"
            f"   flicker/frame={r['flicker_per_frame']:.3f}  "
            f"friend_false_blur={r['friend_false_blur_rate']:.3f}  "
            f"stranger_false_clear={r['stranger_false_clear_rate']:.3f}\n"
            f"   allow={r['cfg_similarity_allow']:.2f}  blur_thresh={r['cfg_similarity_blur']:.2f}  "
            f"instant={r['cfg_instant_allow']:.2f}  detect_every={r['cfg_detect_every']}  "
            f"votes={r['cfg_allow_votes']}/{r['cfg_vote_window']}  missed={r['cfg_max_missed']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune blur parameters on sample frames.")
    parser.add_argument("--input", "-i", default="try.mp4")
    parser.add_argument("--gallery", default="gallery.npz")
    parser.add_argument("--model", default="buffalo_sc")
    parser.add_argument("--sample-frames", type=int, default=240, help="Max frames to evaluate")
    parser.add_argument("--stride", type=int, default=2, help="Sample every Nth frame")
    parser.add_argument("--full-grid", action="store_true", help="Include cartesian grid")
    parser.add_argument("--out-dir", default="tune_results")
    parser.add_argument("--render-top", type=int, default=0, help="Render top N configs to MP4")
    parser.add_argument("--render-frames", type=int, default=0, help="Max frames for render (0=all sample)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.gallery):
        print(f"Run build_gallery.py first. Missing: {args.gallery}", file=sys.stderr)
        sys.exit(1)

    configs = all_tuning_configs(include_full_grid=args.full_grid)
    print(f"Testing {len(configs)} configuration(s) ...\n")

    gallery, _ = load_gallery(args.gallery)
    app = create_face_app(args.model)
    frames = load_sample_frames(args.input, args.sample_frames, args.stride)

    results = iterate_and_score(app, gallery, frames, configs)
    save_results(results, args.out_dir)
    print_top(results, 8)

    best = results[0]
    best_cfg = BlurConfig(
        **{k: best[f"cfg_{k}"] for k in asdict(BlurConfig()).keys()},
    )
    best_path = os.path.join(args.out_dir, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(asdict(best_cfg), f, indent=2)
    print(f"\nBest config saved → {best_path}")
    print(
        f"\nRun full video with:\n"
        f"  python blur.py -i {args.input} -o output.mp4 "
        f"--threshold {best_cfg.similarity_allow} "
        f"--blur-margin {best_cfg.similarity_allow - best_cfg.similarity_blur:.2f} "
        f"--instant-allow {best_cfg.instant_allow} "
        f"--detect-every {best_cfg.detect_every} "
        f"--allow-votes {best_cfg.allow_votes} --vote-window {best_cfg.vote_window} "
        f"--max-missed {best_cfg.max_missed}"
    )

    if args.render_top > 0:
        os.makedirs(os.path.join(args.out_dir, "renders"), exist_ok=True)
        render_n = min(args.render_top, len(results))
        max_f = args.render_frames or len(frames)
        for rank in range(render_n):
            r = results[rank]
            cfg = BlurConfig(**{k: r[f"cfg_{k}"] for k in asdict(BlurConfig()).keys()})
            out_mp4 = os.path.join(
                args.out_dir, "renders", f"rank{rank+1}_{cfg.slug()}.mp4"
            )
            print(f"\nRendering rank #{rank+1} → {out_mp4}")
            run_video(app, gallery, args.input, out_mp4, cfg, max_frames=max_f, print_every=50)


if __name__ == "__main__":
    main()

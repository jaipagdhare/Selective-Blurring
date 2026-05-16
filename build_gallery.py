#!/usr/bin/env python3
"""Build face embedding gallery from Faces/ for selective blur."""

import argparse

import numpy as np

from face_blur_utils import (
    build_gallery_from_dir,
    create_face_app,
    resolve_faces_dir,
    save_gallery,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build allowlist embedding gallery.")
    parser.add_argument(
        "--faces-dir",
        default="Faces",
        help="Folder with reference face images (default: Faces)",
    )
    parser.add_argument(
        "--output",
        default="gallery.npz",
        help="Output .npz path (default: gallery.npz)",
    )
    parser.add_argument(
        "--model",
        default="buffalo_sc",
        help="InsightFace pack name (default: buffalo_sc)",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.35,
        help="Face detector threshold for gallery images (default: 0.35)",
    )
    args = parser.parse_args()

    faces_dir = resolve_faces_dir(args.faces_dir)
    print(f"Building gallery from '{faces_dir}' ...")
    app = create_face_app(args.model, det_thresh=args.det_thresh)
    embeddings, sources = build_gallery_from_dir(app, faces_dir)
    save_gallery(args.output, embeddings, sources)

    sims = embeddings @ embeddings.T
    n = len(sources)
    print(f"\nGallery stats: {n} vectors, dim={embeddings.shape[1]}")
    if n > 1:
        off_diag = sims[~np.eye(n, dtype=bool)]
        print(
            f"  Pairwise similarity: min={off_diag.min():.3f}  "
            f"mean={off_diag.mean():.3f}  max={off_diag.max():.3f}"
        )
    print("Done. Run: python blur.py --input try.mp4")


if __name__ == "__main__":
    main()

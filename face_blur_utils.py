"""Shared helpers for InsightFace gallery + selective video blur."""

from __future__ import annotations

import os
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def resolve_faces_dir(preferred: str = "Faces") -> str:
    if os.path.isdir(preferred):
        return preferred
    if preferred:
        alt = preferred[0].swapcase() + preferred[1:]
        if os.path.isdir(alt):
            return alt
    for name in ("Faces", "faces", "FACES"):
        if os.path.isdir(name):
            return name
    raise FileNotFoundError(
        f"Allowlist folder not found. Create '{preferred}' with face photos."
    )


def list_face_images(faces_dir: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(faces_dir)):
        if name.lower().endswith(IMAGE_EXTS):
            paths.append(os.path.join(faces_dir, name))
    return paths


def onnx_providers() -> Tuple[List[str], int]:
    """Return (providers, ctx_id). ctx_id 0 = GPU when CUDA is available."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime-gpu or onnxruntime.") from exc

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return (["CUDAExecutionProvider", "CPUExecutionProvider"], 0)
    return (["CPUExecutionProvider"], -1)


def create_face_app(model_name: str = "buffalo_sc", det_thresh: float = 0.5):
    from insightface.app import FaceAnalysis

    providers, ctx_id = onnx_providers()
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=det_thresh)
    device = "GPU (CUDA)" if ctx_id == 0 else "CPU"
    print(f"InsightFace model={model_name}  device={device}  providers={providers}")
    return app


def largest_face(faces) -> Optional[object]:
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(emb)
    if n < 1e-6:
        return emb
    return emb / n


def build_gallery_from_dir(app, faces_dir: str) -> Tuple[np.ndarray, List[str]]:
    embeddings: List[np.ndarray] = []
    sources: List[str] = []

    for path in list_face_images(faces_dir):
        img = cv2.imread(path)
        if img is None:
            print(f"  [warn] Could not read {path}")
            continue
        faces = app.get(img)
        face = largest_face(faces)
        if face is None or face.normed_embedding is None:
            print(f"  [warn] No face in {os.path.basename(path)}")
            continue
        embeddings.append(normalize_embedding(face.normed_embedding))
        sources.append(path)
        print(f"  + {os.path.basename(path)}")

    if not embeddings:
        raise ValueError(f"No usable face embeddings from '{faces_dir}'")
    return np.stack(embeddings, axis=0), sources


def save_gallery(path: str, embeddings: np.ndarray, sources: Sequence[str]) -> None:
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        sources=np.array(sources, dtype=object),
    )
    print(f"Saved gallery: {path}  ({len(sources)} embedding(s))")


def load_gallery(path: str) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    emb = emb / norms
    sources = [str(s) for s in data["sources"].tolist()]
    return emb, sources


def max_similarity(gallery: np.ndarray, embedding: np.ndarray) -> float:
    emb = normalize_embedding(embedding)
    return float(np.max(gallery @ emb))


def expand_bbox(
    x1: float, y1: float, x2: float, y2: float, pad: float, w: int, h: int
) -> Tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * pad, bh * pad
    return (
        max(0, int(x1 - mx)),
        max(0, int(y1 - my)),
        min(w, int(x2 + mx)),
        min(h, int(y2 + my)),
    )


def gaussian_blur_roi(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int, kernel: int = 95
) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img))
    y2 = max(0, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    k = max(3, int(kernel) | 1)
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return image


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceTrack:
    _next_id = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        FaceTrack._next_id += 1
        self.id = FaceTrack._next_id
        self.bbox = bbox
        self.velocity = (0.0, 0.0)
        self.allow = False
        self.sim_ema = 0.0
        self.votes: Deque[bool] = deque(maxlen=12)
        self.missed = 0
        self.allow_flips = 0

    def predict(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    def update_bbox(self, bbox: Tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = self.bbox
        nx1, ny1, nx2, ny2 = bbox
        self.velocity = ((nx1 + nx2 - x1 - x2) * 0.5, (ny1 + ny2 - y1 - y2) * 0.5)
        self.bbox = bbox
        self.missed = 0

    def update_similarity(
        self,
        sim: float,
        *,
        allow_thresh: float,
        blur_thresh: float,
        instant_allow: float,
        ema_alpha: float,
        allow_votes: int,
        vote_window: int,
    ) -> None:
        """Hysteresis + EMA: reduces flicker; high sim unlocks allow quickly."""
        if self.sim_ema <= 0:
            self.sim_ema = sim
        else:
            self.sim_ema = ema_alpha * sim + (1.0 - ema_alpha) * self.sim_ema

        vote_allow = self.sim_ema >= allow_thresh
        self.votes.append(vote_allow)

        prev = self.allow
        if sim >= instant_allow or self.sim_ema >= instant_allow:
            self.allow = True
        elif self.sim_ema >= allow_thresh:
            self.allow = True
        elif self.sim_ema < blur_thresh:
            recent = list(self.votes)[-vote_window:]
            if len(recent) >= vote_window and sum(recent) == 0:
                self.allow = False
            elif not self.allow:
                self.allow = False
        else:
            if self.allow:
                pass
            else:
                recent = list(self.votes)[-vote_window:]
                if sum(recent) >= allow_votes:
                    self.allow = True

        if self.allow != prev:
            self.allow_flips += 1

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 0


def match_detections_to_tracks(
    detections: List[Tuple[Tuple[float, float, float, float], float]],
    tracks: List[FaceTrack],
    *,
    iou_thresh: float = 0.25,
    allow_thresh: float = 0.36,
    blur_thresh: float = 0.30,
    instant_allow: float = 0.52,
    ema_alpha: float = 0.45,
    allow_votes: int = 1,
    vote_window: int = 5,
) -> List[Tuple[Tuple[float, float, float, float], float]]:
    """Update tracks from (bbox, similarity) detections; return unmatched."""
    if not detections:
        return []

    unmatched: List[Tuple[Tuple[float, float, float, float], float]] = []
    used_tracks: set[int] = set()

    for det_bbox, sim in detections:
        best_iou, best_idx = 0.0, -1
        for i, tr in enumerate(tracks):
            if i in used_tracks:
                continue
            score = iou(det_bbox, tr.bbox)
            if score > best_iou:
                best_iou, best_idx = score, i

        if best_idx >= 0 and best_iou >= iou_thresh:
            tr = tracks[best_idx]
            tr.update_bbox(det_bbox)
            tr.update_similarity(
                sim,
                allow_thresh=allow_thresh,
                blur_thresh=blur_thresh,
                instant_allow=instant_allow,
                ema_alpha=ema_alpha,
                allow_votes=allow_votes,
                vote_window=vote_window,
            )
            used_tracks.add(best_idx)
        else:
            unmatched.append((det_bbox, sim))

    return unmatched


def init_track(
    bbox: Tuple[float, float, float, float],
    sim: float,
    *,
    allow_thresh: float,
    blur_thresh: float,
    instant_allow: float,
    ema_alpha: float,
    allow_votes: int,
    vote_window: int,
) -> FaceTrack:
    tr = FaceTrack(bbox)
    tr.update_similarity(
        sim,
        allow_thresh=allow_thresh,
        blur_thresh=blur_thresh,
        instant_allow=instant_allow,
        ema_alpha=ema_alpha,
        allow_votes=allow_votes,
        vote_window=vote_window,
    )
    return tr

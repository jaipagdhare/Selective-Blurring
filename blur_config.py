"""Blur pipeline settings and preset grids for tuning."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import product
from typing import Iterator, List


@dataclass(frozen=True)
class BlurConfig:
    """Parameters that control detection, identity, tracking, and blur."""

    # Tuned on try.mp4 sample (tune_blur.py); lower allow + hysteresis reduces flicker.
    similarity_allow: float = 0.34
    similarity_blur: float = 0.28
    instant_allow: float = 0.50
    sim_ema_alpha: float = 0.45
    detect_every: int = 2
    min_face_px: int = 36
    min_det_score: float = 0.45
    max_missed: int = 12
    allow_votes: int = 1
    vote_window: int = 5
    iou_match: float = 0.25
    blur_kernel: int = 95
    bbox_pad: float = 0.18

    def slug(self) -> str:
        return (
            f"allow{self.similarity_allow:.2f}_blur{self.similarity_blur:.2f}"
            f"_det{self.detect_every}_votes{self.allow_votes}w{self.vote_window}"
            f"_miss{self.max_missed}"
        )

    def summary(self) -> str:
        parts = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)]
        return ", ".join(parts)


# Hand-picked grid focused on flicker + false blur on friends.
TUNING_GRID: List[BlurConfig] = [
    BlurConfig(
        similarity_allow=0.34,
        similarity_blur=0.28,
        instant_allow=0.50,
        detect_every=2,
        allow_votes=1,
        vote_window=5,
        max_missed=12,
    ),
    BlurConfig(
        similarity_allow=0.36,
        similarity_blur=0.30,
        instant_allow=0.52,
        detect_every=2,
        allow_votes=1,
        vote_window=5,
        max_missed=10,
    ),
    BlurConfig(
        similarity_allow=0.36,
        similarity_blur=0.28,
        instant_allow=0.55,
        detect_every=2,
        allow_votes=1,
        vote_window=7,
        max_missed=12,
    ),
    BlurConfig(
        similarity_allow=0.38,
        similarity_blur=0.30,
        instant_allow=0.55,
        detect_every=2,
        allow_votes=2,
        vote_window=5,
        max_missed=10,
    ),
    BlurConfig(
        similarity_allow=0.38,
        similarity_blur=0.32,
        instant_allow=0.58,
        detect_every=3,
        allow_votes=1,
        vote_window=5,
        max_missed=10,
    ),
    BlurConfig(
        similarity_allow=0.40,
        similarity_blur=0.32,
        instant_allow=0.60,
        detect_every=3,
        allow_votes=2,
        vote_window=5,
        max_missed=8,
    ),
    BlurConfig(
        similarity_allow=0.35,
        similarity_blur=0.29,
        instant_allow=0.50,
        detect_every=2,
        allow_votes=1,
        vote_window=7,
        max_missed=14,
        sim_ema_alpha=0.35,
    ),
    BlurConfig(
        similarity_allow=0.37,
        similarity_blur=0.31,
        instant_allow=0.54,
        detect_every=2,
        allow_votes=2,
        vote_window=7,
        max_missed=12,
        sim_ema_alpha=0.40,
    ),
]


def iterate_configs(
    *,
    allow_range: tuple[float, ...] = (0.34, 0.36, 0.38),
    blur_delta: float = 0.06,
    detect_every_range: tuple[int, ...] = (2, 3),
    vote_pairs: tuple[tuple[int, int], ...] = ((1, 5), (1, 7), (2, 5)),
    max_missed_range: tuple[int, ...] = (10, 12),
    instant_allow_range: tuple[float, ...] = (0.50, 0.55),
) -> Iterator[BlurConfig]:
    """Cartesian sweep over core stability parameters."""
    seen: set[str] = set()
    for allow, det_every, (votes, window), missed, instant in product(
        allow_range,
        detect_every_range,
        vote_pairs,
        max_missed_range,
        instant_allow_range,
    ):
        blur = max(0.20, allow - blur_delta)
        cfg = BlurConfig(
            similarity_allow=allow,
            similarity_blur=blur,
            instant_allow=instant,
            detect_every=det_every,
            allow_votes=votes,
            vote_window=window,
            max_missed=missed,
        )
        if cfg.slug() not in seen:
            seen.add(cfg.slug())
            yield cfg


def all_tuning_configs(include_full_grid: bool = False) -> List[BlurConfig]:
    configs = list(TUNING_GRID)
    if include_full_grid:
        for cfg in iterate_configs():
            if cfg.slug() not in {c.slug() for c in configs}:
                configs.append(cfg)
    return configs

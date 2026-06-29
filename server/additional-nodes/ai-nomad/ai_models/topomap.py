"""
topomap I/O + 상태유지 주행기 (ROS 무관).

- 스캔 단계: save_frame_to_topomap 로 LiveKit/녹화 프레임을 {dir}/{idx}.png 로 저장.
- 주행 단계: TopomapNavigator 가 topomap 디렉터리를 로드(사전 전처리)하고, 매 틱 현재 관측
  context 로 closest_node 를 갱신하며 subgoal 을 추종(navigate.py 의 closest/subgoal 로직).

topomap 형식: 디렉터리 안 0.png, 1.png, 2.png, ... (deployment/topomaps/images 와 동일 규칙).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from .nomad_infer import Frame, NavResult, NoMaDPolicy


def _numeric_key(fname: str) -> int:
    stem = os.path.splitext(os.path.basename(fname))[0]
    try:
        return int(stem)
    except ValueError:
        return 1 << 30


def list_topomap_files(topomap_dir: str, exts=(".png", ".jpg", ".jpeg")) -> List[str]:
    if not os.path.isdir(topomap_dir):
        return []
    files = [f for f in os.listdir(topomap_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=_numeric_key)
    return [os.path.join(topomap_dir, f) for f in files]


def _imread_bgr(path: str) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(path)
        if img is not None:
            return img
    except Exception:
        pass
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def load_topomap_frames(topomap_dir: str) -> List[np.ndarray]:
    """topomap 디렉터리 → BGR ndarray 리스트 (파일명 숫자 오름차순)."""
    return [_imread_bgr(p) for p in list_topomap_files(topomap_dir)]


def save_frame_to_topomap(topomap_dir: str, index: int, frame_bgr: np.ndarray) -> str:
    """스캔: 프레임 1장을 {dir}/{index}.png 로 저장. frame_bgr=(H,W,3) BGR."""
    os.makedirs(topomap_dir, exist_ok=True)
    path = os.path.join(topomap_dir, f"{index}.png")
    try:
        import cv2
        cv2.imwrite(path, frame_bgr)
    except Exception:
        from PIL import Image
        Image.fromarray(frame_bgr[:, :, ::-1]).save(path)  # BGR→RGB 저장
    return path


class TopomapNavigator:
    """topomap 로 NoMaD 주행 — closest_node 상태를 틱 간 유지."""

    def __init__(
        self,
        policy: NoMaDPolicy,
        topomap_frames: Sequence[np.ndarray],
        goal_node: int = -1,
        radius: int = 4,
        close_threshold: float = 0.2,   # dist_pred 정규화(0~1) 스케일 — 원본 navigate.py 의 3.0(timestep)과 다름
        num_samples: int = 8,
        bgr_to_rgb: bool = True,
    ) -> None:
        if len(topomap_frames) == 0:
            raise ValueError("topomap이 비어있음 — 스캔 먼저 수행 필요")
        self.policy = policy
        # topomap 프레임을 한 번만 전처리해 GPU 텐서로 보관 (M,3,H,W)
        self.goals = policy.preprocess_goals(list(topomap_frames), bgr_to_rgb=bgr_to_rgb)
        self.num_nodes = self.goals.shape[0]
        self.goal_node = (self.num_nodes - 1) if goal_node < 0 else min(goal_node, self.num_nodes - 1)
        self.radius = radius
        self.close_threshold = close_threshold
        self.num_samples = num_samples
        self.bgr_to_rgb = bgr_to_rgb
        self.closest_node = 0

    def reset(self) -> None:
        self.closest_node = 0

    def step(self, context_frames: Sequence[Frame]) -> NavResult:
        res = self.policy.navigate_topomap(
            context_frames=context_frames,
            topomap_goals=self.goals,
            closest_node=self.closest_node,
            goal_node=self.goal_node,
            radius=self.radius,
            close_threshold=self.close_threshold,
            num_samples=self.num_samples,
            bgr_to_rgb=self.bgr_to_rgb,
        )
        self.closest_node = res.closest_node   # 상태 유지(단조 전진 경향)
        return res

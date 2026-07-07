"""
NoMaD 추론 코어 (전송 무관 · ROS 무관) — ai-nomad 노드용.

이 repo에서 학습한 **velocity 예측형 NoMaD_pose**(train/config/nomad.yaml, predict_velocity=True)를
서버/ROS 독립으로 로드·추론한다. 기본 추론 모드는 단일 최종 goal 이미지 조건 추론이다.

학습 파이프라인(vint_train/data/vint_dataset_episode.py)과 동등해야 하는 핵심:
  1. 이미지 전처리: uint8(3,H,W) → /255 → resize[H,W]. **ImageNet 정규화 없음**.
  2. context: (context_size+1)장 채널 concat → (1, 3*(ctx+1), H, W).
  3. goal mask: 0=goal-conditioned, 1=undirected. 주행은 0.
  4. 출력 denormalize:
     - action: (v,w)=(norm+1)/2*scale+min
     - pose: (x,y,theta)=(norm+1)/2*scale+min
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_import_paths() -> None:
    """vint_train, diffusion_policy 를 path에 (repo 안 실행 시 자동; docker는 PYTHONPATH/마운트)."""
    here = os.path.abspath(__file__)
    candidates = []
    if os.environ.get("NOMAD_TRAIN_DIR"):
        candidates.append(os.environ["NOMAD_TRAIN_DIR"])
    if os.environ.get("DIFFUSION_POLICY_DIR"):
        candidates.append(os.environ["DIFFUSION_POLICY_DIR"])
    d = here
    for _ in range(8):
        d = os.path.dirname(d)
        candidates.append(os.path.join(d, "train"))
        candidates.append(os.path.join(d, "diffusion_policy"))
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


_ensure_import_paths()


# ── 학습 데이터 기준 정규화 통계 (train 로그 출력과 동일하게 직접 기입) ──
DEFAULT_ACTION_MIN = np.array([-0.17029296, -0.3425848], dtype=np.float32)
DEFAULT_ACTION_MAX = np.array([0.16505694, 0.30039853], dtype=np.float32)

# TODO: train 로그의 [Global pose normalization] min/max [x,y,theta] 값으로 교체.
DEFAULT_POSE_MIN = np.array([-0.29979697, -0.49998415, -2.8248427], dtype=np.float32)
DEFAULT_POSE_MAX = np.array([0.29997006, 0.03298904, -0.11712503], dtype=np.float32)



@dataclass
class NoMaDParams:
    vision_encoder: str = "nomad_vint"
    encoding_size: int = 256
    context_size: int = 5
    mha_num_attention_heads: int = 4
    mha_num_attention_layers: int = 4
    mha_ff_dim_factor: int = 4
    len_traj_pred: int = 60
    num_diffusion_iters: int = 100
    image_width: int = 320       # nomad.yaml image_size = [W, H]
    image_height: int = 240
    action_dim: int = 2
    use_encoder: bool = True
    use_imu: bool = False
    use_lidar: bool = False
    encoder_imu_context_size: int = 30
    lidar_context_size: int = 5
    num_image_keys: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "NoMaDParams":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        img = raw.get("image_size", [320, 240])
        return cls(
            vision_encoder=raw.get("vision_encoder", "nomad_vint"),
            encoding_size=int(raw.get("encoding_size", 256)),
            context_size=int(raw.get("context_size", 5)),
            mha_num_attention_heads=int(raw.get("mha_num_attention_heads", 4)),
            mha_num_attention_layers=int(raw.get("mha_num_attention_layers", 4)),
            mha_ff_dim_factor=int(raw.get("mha_ff_dim_factor", 4)),
            len_traj_pred=int(raw.get("len_traj_pred", 60)),
            num_diffusion_iters=int(raw.get("num_diffusion_iters", 10)),
            image_width=int(img[0]),
            image_height=int(img[1]),
            use_encoder=bool(raw.get("use_encoder", True)),
            use_imu=bool(raw.get("use_imu", False)),
            use_lidar=bool(raw.get("use_lidar", False)),
            encoder_imu_context_size=int(raw.get("encoder_imu_context_size", 30)),
            lidar_context_size=int(raw.get("lidar_context_size", 5)),
            num_image_keys=len(raw.get("data_image_keys", ["image_bottom"])),
        )


Frame = Union[np.ndarray, "torch.Tensor"]


def preprocess_frame(frame: Frame, image_h: int, image_w: int, bgr_to_rgb: bool = False) -> torch.Tensor:
    """단일 프레임 → (3,H,W) float[0,1]. 학습 _load_image 동등(/255 + resize, 정규화 없음)."""
    if isinstance(frame, torch.Tensor):
        t = frame.detach().float()
    else:
        t = torch.as_tensor(np.ascontiguousarray(frame), dtype=torch.float32)
    if t.ndim == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"프레임 형태가 (3,H,W)/(H,W,3)이 아님: {tuple(t.shape)}")
    if bgr_to_rgb:
        t = t[[2, 1, 0], :, :]
    if t.max() > 1.5:
        t = t / 255.0
    return F.interpolate(t.unsqueeze(0), size=(image_h, image_w), mode="bilinear", align_corners=False).squeeze(0)


@dataclass
class NavResult:
    velocities: np.ndarray      # (T,2) denormalize된 (v,w)
    closest_node: int           # 갱신된 현재 위치 노드
    subgoal_node: int           # 이번에 추종한 subgoal 노드
    closest_distance: float     # closest 노드까지 정규화 temporal distance
    reached_goal: bool


class NoMaDPolicy:
    """학습된 velocity-NoMaD 로더 + 추론기."""

    def __init__(
        self,
        ckpt_path: str,
        params: Optional[NoMaDParams] = None,
        device: Optional[torch.device] = None,
        action_min: np.ndarray = DEFAULT_ACTION_MIN,
        action_max: np.ndarray = DEFAULT_ACTION_MAX,
        pose_min: np.ndarray = DEFAULT_POSE_MIN,
        pose_max: np.ndarray = DEFAULT_POSE_MAX,
    ) -> None:
        self.params = params or NoMaDParams()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_min = np.asarray(action_min, dtype=np.float32)
        self.action_max = np.asarray(action_max, dtype=np.float32)
        self.pose_min = np.asarray(pose_min, dtype=np.float32)
        self.pose_max = np.asarray(pose_max, dtype=np.float32)
        self.model = self._build_and_load(ckpt_path)
        self.noise_scheduler = self._build_scheduler()

    def _build_and_load(self, ckpt_path: str) -> torch.nn.Module:
        from vint_train.models.nomad.nomad import NoMaD_pose, PoseNetwork
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

        p = self.params
        if p.vision_encoder != "nomad_vint":
            raise ValueError(f"이 코어는 nomad_vint만 지원 (got {p.vision_encoder})")
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=p.encoding_size, context_size=p.context_size,
            mha_num_attention_heads=p.mha_num_attention_heads,
            mha_num_attention_layers=p.mha_num_attention_layers,
            mha_ff_dim_factor=p.mha_ff_dim_factor,
            sensor_context_sizes={
                "encoder": p.encoder_imu_context_size,
                "imu": p.encoder_imu_context_size,
                "lidar": p.lidar_context_size,
            },
            use_encoder=p.use_encoder,
            use_imu=p.use_imu,
            use_lidar=p.use_lidar,
            num_image_keys=p.num_image_keys,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        noise_pred_net = TransformerForDiffusion(
            input_dim=p.action_dim,
            output_dim=p.action_dim,
            horizon=p.len_traj_pred,
            cond_dim=p.encoding_size,
            n_obs_steps=1,
            n_layer=12,
            n_head=6,
            n_emb=384,
        )

        pose_pred_net = PoseNetwork(embedding_dim=p.encoding_size)
        
        model = NoMaD_pose(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            pose_pred_net=pose_pred_net,
        )

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[NoMaDPolicy] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        model.to(self.device).eval()
        return model

    def _build_scheduler(self):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers import DPMSolverMultistepScheduler

        
        return DDPMScheduler(
            num_train_timesteps=self.params.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon",
        )
        """
        return DPMSolverMultistepScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
        )
        """

    # ── 전처리 ──
    def build_obs_tensor(self, context_frames: Sequence[Frame], bgr_to_rgb: bool = False) -> torch.Tensor:
        need = self.params.context_size + 1
        if len(context_frames) < need:
            raise ValueError(f"context 프레임 부족: {len(context_frames)} < {need}")
        frames = list(context_frames)[-need:]
        chans = [preprocess_frame(f, self.params.image_height, self.params.image_width, bgr_to_rgb) for f in frames]
        return torch.cat(chans, dim=0).unsqueeze(0).to(self.device)

    def preprocess_goals(self, goal_frames: Sequence[Frame], bgr_to_rgb: bool = False) -> torch.Tensor:
        gs = [preprocess_frame(g, self.params.image_height, self.params.image_width, bgr_to_rgb) for g in goal_frames]
        return torch.stack(gs, dim=0).to(self.device)  # (N,3,H,W)

    # ── 추론 빌딩블록 ──
    def _default_sensor_hist(self, batch_size: int):
        p = self.params
        encoder_hist = (
            torch.zeros((batch_size, p.encoder_imu_context_size, 2), device=self.device)
            if p.use_encoder
            else None
        )
        imu_hist = (
            torch.zeros((batch_size, p.encoder_imu_context_size, 6), device=self.device)
            if p.use_imu
            else None
        )
        lidar_hist = (
            torch.zeros((batch_size, p.lidar_context_size, 360), device=self.device)
            if p.use_lidar
            else None
        )
        return encoder_hist, imu_hist, lidar_hist

    def _encode(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        mask: torch.Tensor,
        encoder_hist: Optional[torch.Tensor] = None,
        imu_hist: Optional[torch.Tensor] = None,
        lidar_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        default_encoder, default_imu, default_lidar = self._default_sensor_hist(batch_size)
        return self.model(
            "vision_encoder",
            obs_img=obs,
            goal_img=goal,
            input_goal_mask=mask,
            encoder_hist=encoder_hist.to(self.device) if encoder_hist is not None else default_encoder,
            imu_hist=imu_hist.to(self.device) if imu_hist is not None else default_imu,
            lidar_hist=lidar_hist.to(self.device) if lidar_hist is not None else default_lidar,
        )

    def _dist(self, obsgoal_cond: torch.Tensor) -> torch.Tensor:
        return torch.full((obsgoal_cond.shape[0],), float("nan"), device=self.device)

    def _pose(self, obsgoal_cond: torch.Tensor) -> torch.Tensor:
        return self.model("pose_pred_net", obsgoal_cond=obsgoal_cond)

    def _run_diffusion(self, obs_cond: torch.Tensor, num_samples: int) -> np.ndarray:
        """obs_cond(1,enc) 또는 (n,enc) → denormalize된 (num_samples, T, 2)."""
        p = self.params
        cond = obs_cond.repeat(num_samples, 1) if obs_cond.ndim == 2 and obs_cond.shape[0] == 1 else obs_cond
        naction = torch.randn((cond.shape[0], p.len_traj_pred, p.action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(p.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.model("noise_pred_net", sample=naction, timestep=k, global_cond=cond)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        norm = naction.cpu().numpy().reshape(cond.shape[0], p.len_traj_pred, p.action_dim)
        return self._denormalize_action(norm)

    @staticmethod
    def _denormalize_data(norm: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        norm = np.asarray(norm, dtype=np.float32)
        min_val = np.asarray(min_val, dtype=np.float32)
        max_val = np.asarray(max_val, dtype=np.float32)
        return ((norm + 1.0) / 2.0) * (max_val - min_val) + min_val

    def _denormalize_action(self, norm: np.ndarray) -> np.ndarray:
        return ((norm + 1.0) / 2.0) * (self.action_max - self.action_min) + self.action_min

    def _denormalize_pose(self, norm_pose: np.ndarray) -> np.ndarray:
        return self._denormalize_data(norm_pose, self.pose_min, self.pose_max)

    # ── 단일 goal 추론 ──
    @torch.no_grad()
    def infer(
        self,
        context_frames,
        goal_frame,
        masked=False,
        num_samples=8,
        bgr_to_rgb=False,
        encoder_hist: Optional[torch.Tensor] = None,
        imu_hist: Optional[torch.Tensor] = None,
        lidar_hist: Optional[torch.Tensor] = None,
    ) -> dict:
        p = self.params
        obs = self.build_obs_tensor(context_frames, bgr_to_rgb)
        goal = (self.preprocess_goals([goal_frame], bgr_to_rgb) if goal_frame is not None
                else torch.zeros((1, 3, p.image_height, p.image_width), device=self.device))
        mask = (torch.ones if (masked or goal_frame is None) else torch.zeros)(1, device=self.device).long()
        cond = self._encode(obs, goal, mask, encoder_hist=encoder_hist, imu_hist=imu_hist, lidar_hist=lidar_hist)
        pose_norm = self._pose(cond).detach().cpu().numpy()
        pose = self._denormalize_pose(pose_norm)
        vel_all = self._run_diffusion(cond, num_samples)
        return {
            "velocities": vel_all.mean(axis=0),
            "velocities_all": vel_all,
            "pose": pose[0],
            "pose_all": pose,
            "pose_norm": pose_norm[0],
            "pose_norm_all": pose_norm,
            "distance": float("nan"),
        }

    @torch.no_grad()
    def infer_tensors(
        self,
        obs_tensor,
        goal_tensor,
        masked=False,
        num_samples=8,
        encoder_hist: Optional[torch.Tensor] = None,
        imu_hist: Optional[torch.Tensor] = None,
        lidar_hist: Optional[torch.Tensor] = None,
    ) -> dict:
        """사전 전처리된 텐서 입력 추론 (metric/평가용 — 학습 dataset 의 obs/goal 텐서를 그대로 사용).

        obs_tensor : (1, 3*(ctx+1), H, W), goal_tensor : (1, 3, H, W). 반환은 infer 와 동일 +
        dist_pred 의 raw 정규화값(distance).
        """
        obs = obs_tensor.to(self.device)
        goal = goal_tensor.to(self.device)
        mask = (torch.ones if masked else torch.zeros)(obs.shape[0], device=self.device).long()
        cond = self._encode(obs, goal, mask, encoder_hist=encoder_hist, imu_hist=imu_hist, lidar_hist=lidar_hist)
        pose_norm = self._pose(cond).detach().cpu().numpy()
        pose = self._denormalize_pose(pose_norm)
        vel_all = self._run_diffusion(cond, num_samples)   # (num_samples, T, 2) denorm
        return {
            "velocities": vel_all.mean(axis=0),
            "velocities_all": vel_all,
            "pose": pose[0],
            "pose_all": pose,
            "pose_norm": pose_norm[0],
            "pose_norm_all": pose_norm,
            "distance": float("nan"),
        }

    # ── topomap 주행 (진짜 NoMaD; navigate.py 로직 ROS-free 이식) ──
    @torch.no_grad()
    def navigate_topomap(
        self,
        context_frames: Sequence[Frame],
        topomap_goals: torch.Tensor,        # (M,3,H,W) 사전 전처리된 topomap (preprocess_goals)
        closest_node: int,
        goal_node: int,
        radius: int = 4,
        close_threshold: float = 0.2,
        num_samples: int = 8,
        bgr_to_rgb: bool = False,
    ) -> NavResult:
        """현재 관측 context + topomap → closest node 갱신 + subgoal 추종 속도 궤적."""
        p = self.params
        m = topomap_goals.shape[0]
        goal_node = min(goal_node, m - 1)
        obs = self.build_obs_tensor(context_frames, bgr_to_rgb)        # (1,C,H,W)

        start = max(closest_node - radius, 0)
        end = min(closest_node + radius + 1, goal_node)
        cand = topomap_goals[start:end + 1]                            # (n,3,H,W)
        n = cand.shape[0]
        mask = torch.zeros(n, device=self.device).long()              # goal-conditioned
        cond = self._encode(obs.repeat(n, 1, 1, 1), cand, mask)        # (n,enc)
        dists = self._dist(cond).detach().cpu().numpy()               # (n,)
        min_idx = int(np.argmin(dists))
        new_closest = start + min_idx
        # 충분히 가까우면 다음 노드를 subgoal로
        sg_local = min(min_idx + int(dists[min_idx] < close_threshold), n - 1)
        obs_cond = cond[sg_local].unsqueeze(0)                        # (1,enc)
        velocities = self._run_diffusion(obs_cond, num_samples)[0]    # (T,2)
        return NavResult(
            velocities=velocities,
            closest_node=new_closest,
            subgoal_node=start + sg_local,
            closest_distance=float(dists[min_idx]),
            reached_goal=(new_closest >= goal_node),
        )


def velocities_to_trajectory(velocities: np.ndarray, dt: float = 0.0333) -> np.ndarray:
    """(T,2)(v,w) → RK4 적분 (T+1,3)[x,y,theta]. 시각화/검증용."""
    traj = np.zeros((len(velocities) + 1, 3), dtype=np.float64)
    q = np.zeros(3)

    def f(qq, v, w):
        return np.array([v * np.cos(qq[2]), v * np.sin(qq[2]), w])

    for i, (v, w) in enumerate(velocities):
        k1 = f(q, v, w); k2 = f(q + 0.5 * dt * k1, v, w)
        k3 = f(q + 0.5 * dt * k2, v, w); k4 = f(q + dt * k3, v, w)
        q = q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q[2] = (q[2] + np.pi) % (2 * np.pi) - np.pi
        traj[i + 1] = q
    return traj

"""AINomadConfig — BasePluginConfig(common) + ai-nomad 전용 필드 (NoMaD topomap-goal 주행/스캔)."""
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from node_sdk.config import BasePluginConfig, _coerce_bool, load_config


@dataclass
class AINomadConfig(BasePluginConfig):
    # ── 출력 ──
    enable_send: bool = False                 # True=속도 궤적 MQTT 발행 + robot_control 점유
    ai_command_topic: str = ""                # __post_init__: robot/{id}/ai-command

    # ── 모드 ──
    nomad_mode: str = "navigate"              # navigate | scan

    # ── 모델 / topomap goal image ──
    nomad_ckpt: str = "/models/ai-nomad/ema_18.pth"
    nomad_params_yaml: str = ""               # 비면 NoMaDParams 기본값(nomad.yaml 동일)
    topomap_dir: str = "/models/ai-nomad/topomap" # 첫 번째 이미지를 최종 goal image로 사용
    nomad_radius: int = 4
    nomad_close_threshold: float = 0.2   # dist_pred 정규화(0~1) 스케일
    nomad_goal_node: int = -1                 # -1 = topomap 마지막 노드
    nomad_num_samples: int = 1
    nomad_obs_stream: str = "video1"          # 학습 image_bottom 에 해당하는 LiveKit 트랙
    nomad_context_stride: int = 1
    nomad_bgr_to_rgb: bool = True

    # ── 스캔 ──
    scan_interval_sec: float = 1.0            # topomap 노드 저장 간격(초)
    scan_reset: bool = False                  # 시작 시 topomap_dir 비우기

    # ── 실환경 평가 trace (navigate 시 추론 기록 CSV) ──
    nomad_trace_path: str = ""                # 비면 off. 예: /models/ai-nomad/trace.csv

    def __post_init__(self):
        super().__post_init__()
        if not self.ai_command_topic:
            self.ai_command_topic = f"robot/{self.robot_id}/ai-command"


# nomad 전용 env (docker-compose 가 주입; load_config 의 공통 _ENV_OVERRIDES 외 추가분)
_NOMAD_ENV = {
    "NOMAD_MODE": ("nomad_mode", str),
    "NOMAD_CKPT": ("nomad_ckpt", str),
    "NOMAD_PARAMS_YAML": ("nomad_params_yaml", str),
    "NOMAD_TOPOMAP_DIR": ("topomap_dir", str),
    "NOMAD_RADIUS": ("nomad_radius", int),
    "NOMAD_CLOSE_THRESHOLD": ("nomad_close_threshold", float),
    "NOMAD_GOAL_NODE": ("nomad_goal_node", int),
    "NOMAD_NUM_SAMPLES": ("nomad_num_samples", int),
    "NOMAD_OBS_STREAM": ("nomad_obs_stream", str),
    "NOMAD_CONTEXT_STRIDE": ("nomad_context_stride", int),
    "NOMAD_BGR_TO_RGB": ("nomad_bgr_to_rgb", _coerce_bool),
    "SCAN_INTERVAL_SEC": ("scan_interval_sec", float),
    "SCAN_RESET": ("scan_reset", _coerce_bool),
    "NOMAD_TRACE_PATH": ("nomad_trace_path", str),
}


def load_ai_nomad_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AINomadConfig:
    cfg = load_config(config_path, overrides, config_cls=AINomadConfig)
    # nomad 전용 env 최우선 적용 (load_config 가 모르는 키들)
    for env_key, (field, caster) in _NOMAD_ENV.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, field, caster(raw))
        except (ValueError, TypeError):
            pass
    cfg.__post_init__()
    return cfg

"""
ai-nomad P2 inference 플러그인 — topomap 기반 NoMaD 주행 (진짜 NoMaD 흐름).

window_snapshot 의 관측 카메라(context) + 마운트된 topomap 디렉터리 → TopomapNavigator 가
closest node 갱신 + subgoal 추종 속도 궤적 → CommandStep 리스트.

config(AINomadConfig) 에서 읽음: nomad_ckpt, topomap_dir, nomad_radius, nomad_close_threshold,
nomad_goal_node, nomad_num_samples, nomad_obs_stream, nomad_context_stride, nomad_bgr_to_rgb.
"""
import logging
import os
import time

import numpy as np

from node_sdk import CommandStep

from ai_models.nomad_infer import NoMaDParams, NoMaDPolicy
from ai_models.topomap import TopomapNavigator, load_topomap_frames

logger = logging.getLogger(__name__)

_NAV = None  # TopomapNavigator (lazy, 상태 유지)


def _cfg(config, name, default):
    return getattr(config, name, default)


def _load_navigator_once(config):
    global _NAV
    if _NAV is not None:
        return _NAV
    params_yaml = _cfg(config, "nomad_params_yaml", "") or os.environ.get("NOMAD_PARAMS_YAML", "")
    params = NoMaDParams.from_yaml(params_yaml) if params_yaml else NoMaDParams()
    ckpt = _cfg(config, "nomad_ckpt", "") or os.environ.get("NOMAD_CKPT", "/models/ai-nomad/ema_18.pth")
    logger.info("NoMaD 로드: ckpt=%s", ckpt)
    policy = NoMaDPolicy(ckpt, params=params)

    topomap_dir = _cfg(config, "topomap_dir", "/models/ai-nomad/topomap")
    frames = load_topomap_frames(topomap_dir)
    if not frames:
        raise FileNotFoundError(
            f"topomap 비어있음: {topomap_dir} — 먼저 스캔(NOMAD_MODE=scan)으로 경로를 기록하세요.")
    _NAV = TopomapNavigator(
        policy, frames,
        goal_node=int(_cfg(config, "nomad_goal_node", -1)),
        radius=int(_cfg(config, "nomad_radius", 4)),
        close_threshold=float(_cfg(config, "nomad_close_threshold", 3.0)),
        num_samples=int(_cfg(config, "nomad_num_samples", 1)),
        bgr_to_rgb=bool(_cfg(config, "nomad_bgr_to_rgb", True)),
    )
    logger.info("topomap 로드 완료: %d 노드 (goal=%d)", _NAV.num_nodes, _NAV.goal_node)
    return _NAV


def _zero_horizon(config):
    n = int(_cfg(config, "inference_size", 16))
    step_dt = 1.0 / max(1, int(_cfg(config, "inference_fps", 30)))
    return [CommandStep(0.0, 0.0, 0.0, (i + 1) * step_dt) for i in range(n)]


def _extract_context(window_snapshot, need, stride, stream):
    vid = window_snapshot.get(stream, ([], []))
    frames = vid[0] if isinstance(vid, tuple) else vid
    if not frames:
        return None
    idxs = [len(frames) - 1 - i * max(1, stride) for i in range(need)]
    idxs = [i for i in idxs if i >= 0]
    if len(idxs) < need:
        idxs = idxs + [0] * (need - len(idxs))
    return [frames[i] for i in reversed(idxs)]


_TRACE_INIT = False


def _append_trace(config, res):
    """실환경 평가용: navigate 추론 1틱을 CSV 한 줄로 기록 (nomad_trace_path 설정 시)."""
    global _TRACE_INIT
    path = _cfg(config, "nomad_trace_path", "") or os.environ.get("NOMAD_TRACE_PATH", "")
    if not path:
        return
    try:
        v = res.velocities
        if not _TRACE_INIT and not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write("ts,closest_node,subgoal_node,closest_distance,reached,v0,w0,v_mean,w_mean\n")
        _TRACE_INIT = True
        with open(path, "a") as f:
            f.write(f"{time.time():.3f},{res.closest_node},{res.subgoal_node},"
                    f"{res.closest_distance:.4f},{int(res.reached_goal)},"
                    f"{float(v[0,0]):.4f},{float(v[0,1]):.4f},"
                    f"{float(np.mean(v[:,0])):.4f},{float(np.mean(v[:,1])):.4f}\n")
    except Exception:
        logger.debug("trace 기록 실패", exc_info=True)


def nomad_topomap_inference(window_snapshot, latency_marks, config):
    nav = _load_navigator_once(config)
    need = nav.policy.params.context_size + 1
    stride = int(_cfg(config, "nomad_context_stride", 1))
    stream = _cfg(config, "nomad_obs_stream", "video1")

    context = _extract_context(window_snapshot, need, stride, stream)
    if context is None:
        logger.warning("context 프레임 없음(stream=%s) → 정지", stream)
        return _zero_horizon(config)

    try:
        res = nav.step(context)
    except Exception:
        logger.exception("NoMaD topomap 추론 실패 → 정지")
        return _zero_horizon(config)

    _append_trace(config, res)

    if res.reached_goal:
        logger.info("goal 도달(node %d) → 정지", res.closest_node)
        return _zero_horizon(config)

    velocities = res.velocities
    dt = 1.0 / max(1, int(_cfg(config, "inference_fps", 30)))
    n = min(velocities.shape[0], int(_cfg(config, "inference_size", 16)))
    steps = []
    for i in range(n):
        v, w = float(velocities[i, 0]), float(velocities[i, 1])
        steps.append(CommandStep(dx=v * dt, dy=0.0, dtheta=w * dt, dt=(i + 1) * dt))
    return steps


inference_fn = nomad_topomap_inference

"""
ai-nomad P2 inference plugin.

The file name is kept for plugin compatibility, but inference no longer uses a
topomap. Each tick runs NoMaD_pose with the current observation context and one
final goal image.

The topomap directory is still used as the goal-image source. Since every
topomap capture is expected to be taken at the final goal position, inference
always uses the first topomap frame as the single goal image.
"""
import logging
import os
import time

import numpy as np
import torch

from node_sdk import CommandStep

from ai_models.nomad_infer import NoMaDParams, NoMaDPolicy
from ai_models.topomap import load_topomap_frames

logger = logging.getLogger(__name__)

_POLICY = None
_GOAL_TOPOMAP_DIR = None
_GOAL_TOPOMAP_FRAME = None
_TRACE_INIT = False


def _cfg(config, name, default):
    return getattr(config, name, default)


def _load_policy_once(config):
    global _POLICY
    if _POLICY is not None:
        return _POLICY

    params_yaml = _cfg(config, "nomad_params_yaml", "") or os.environ.get("NOMAD_PARAMS_YAML", "")
    params = NoMaDParams.from_yaml(params_yaml) if params_yaml else NoMaDParams()
    ckpt = _cfg(config, "nomad_ckpt", "") or os.environ.get("NOMAD_CKPT", "/models/ai-nomad/ema_latest.pth")
    logger.info("NoMaD_pose load: ckpt=%s", ckpt)
    _POLICY = NoMaDPolicy(ckpt, params=params)
    return _POLICY


def _zero_horizon(config):
    n = int(_cfg(config, "inference_size", 16))
    step_dt = 1.0 / max(1, int(_cfg(config, "inference_fps", 30)))
    return [CommandStep(0.0, 0.0, 0.0, (i + 1) * step_dt) for i in range(n)]


def _extract_frames(window_snapshot, stream):
    vid = window_snapshot.get(stream, ([], []))
    return vid[0] if isinstance(vid, tuple) else vid


def _extract_series(window_snapshot, stream):
    data = window_snapshot.get(stream, ([], []))
    return data[0] if isinstance(data, tuple) else data


def _extract_context(window_snapshot, need, stride, stream):
    frames = _extract_frames(window_snapshot, stream)
    if not frames:
        return None
    idxs = [len(frames) - 1 - i * max(1, stride) for i in range(need)]
    idxs = [i for i in idxs if i >= 0]
    if len(idxs) < need:
        idxs = idxs + [0] * (need - len(idxs))
    return [frames[i] for i in reversed(idxs)]


def _as_encoder_row(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for keys in (
            ("vx", "wz"),
            ("left", "right"),
            ("l", "r"),
            ("left_delta", "right_delta"),
            ("linear", "angular"),
            ("v", "w"),
        ):
            if all(k in value for k in keys):
                return [float(value[keys[0]]), float(value[keys[1]])]
        numeric = [v for v in value.values() if isinstance(v, (int, float, np.number))]
        if len(numeric) >= 2:
            return [float(numeric[0]), float(numeric[1])]

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return [float(arr[0]), 0.0]
    return [float(arr[0]), float(arr[1])]


def _extract_encoder_hist(window_snapshot, need, config):
    stream = (
        _cfg(config, "nomad_encoder_stream", "")
        or os.environ.get("NOMAD_ENCODER_STREAM", "")
        or "encoder"
    )
    candidates = [stream, "encoder_hist", "encoder", "wheel_encoder", "odom_encoder"]
    series = None
    used_key = None
    for key in dict.fromkeys(candidates):
        series = _extract_series(window_snapshot, key)
        if series is not None and len(series) > 0:
            used_key = key
            break
    if series is None or len(series) == 0:
        logger.debug("encoder history missing(keys=%s); using policy default", candidates)
        return None

    stride = 1
    idxs = [len(series) - 1 - i * max(1, stride) for i in range(need)]
    idxs = [i for i in idxs if i >= 0]
    if len(idxs) < need:
        idxs = idxs + [0] * (need - len(idxs))

    rows = [_as_encoder_row(series[i]) for i in reversed(idxs)]
    if any(row is None for row in rows):
        logger.debug("encoder history has unsupported row(key=%s); using policy default", used_key)
        return None

    hist = torch.as_tensor(rows, dtype=torch.float32).unsqueeze(0)
    logger.debug("encoder history loaded key=%s shape=%s", used_key, tuple(hist.shape))
    return hist


def _load_goal_from_topomap_once(config):
    global _GOAL_TOPOMAP_DIR, _GOAL_TOPOMAP_FRAME
    topomap_dir = _cfg(config, "topomap_dir", "/models/ai-nomad/topomap") or os.environ.get(
        "NOMAD_TOPOMAP_DIR",
        "/models/ai-nomad/topomap",
    )
    if _GOAL_TOPOMAP_FRAME is not None and _GOAL_TOPOMAP_DIR == topomap_dir:
        return _GOAL_TOPOMAP_FRAME

    frames = load_topomap_frames(topomap_dir)
    if not frames:
        raise FileNotFoundError(f"topomap is empty: {topomap_dir}")

    _GOAL_TOPOMAP_DIR = topomap_dir
    _GOAL_TOPOMAP_FRAME = frames[0]
    logger.info("topomap goal loaded: %s first frame", topomap_dir)
    return _GOAL_TOPOMAP_FRAME


def _append_trace(config, velocities, pose):
    """Write one inference tick to CSV when nomad_trace_path is set."""
    global _TRACE_INIT
    path = _cfg(config, "nomad_trace_path", "") or os.environ.get("NOMAD_TRACE_PATH", "")
    if not path:
        return
    try:
        if not _TRACE_INIT and not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write("ts,v0,w0,v_mean,w_mean,pose\n")
        _TRACE_INIT = True
        pose_str = " ".join(f"{float(x):.5f}" for x in np.asarray(pose).reshape(-1))
        with open(path, "a") as f:
            f.write(
                f"{time.time():.3f},"
                f"{float(velocities[0, 0]):.4f},{float(velocities[0, 1]):.4f},"
                f"{float(np.mean(velocities[:, 0])):.4f},{float(np.mean(velocities[:, 1])):.4f},"
                f"{pose_str}\n"
            )
    except Exception:
        logger.debug("trace write failed", exc_info=True)


def nomad_topomap_inference(window_snapshot, latency_marks, config):
    policy = _load_policy_once(config)
    need = policy.params.context_size + 1
    frame_stride = int(_cfg(config, "nomad_context_stride", 1))
    stream = _cfg(config, "nomad_obs_stream", "video1")

    context = _extract_context(window_snapshot, need, frame_stride, stream)
    if context is None:
        logger.warning("context frames missing(stream=%s); stop", stream)
        return _zero_horizon(config)
    encoder_hist = _extract_encoder_hist(window_snapshot, policy.params.encoder_imu_context_size, config)

    try:
        goal_frame = _load_goal_from_topomap_once(config)
    except Exception:
        logger.exception("topomap goal image missing; stop")
        return _zero_horizon(config)

    try:
        out = policy.infer(
            context,
            goal_frame,
            num_samples=int(_cfg(config, "nomad_num_samples", 1)),
            bgr_to_rgb=bool(_cfg(config, "nomad_bgr_to_rgb", True)),
            encoder_hist=encoder_hist,
        )
        velocities = np.asarray(out.get("velocities", []), dtype=np.float32)
        logger.info("NoMaD pose raw=%s", out.get("pose"))
        if velocities.size:
            logger.info(
                "velocity first=(%.4f, %.4f) mean=(%.4f, %.4f)",
                float(velocities[0, 0]),
                float(velocities[0, 1]),
                float(np.mean(velocities[:, 0])),
                float(np.mean(velocities[:, 1])),
            )
        else:
            logger.info("velocity empty")
        
    except Exception:
        logger.exception("NoMaD_pose inference failed; stop")
        return _zero_horizon(config)

    velocities = out["velocities"]
    _append_trace(config, velocities, out.get("pose"))

    dt = 1.0 / max(1, int(_cfg(config, "inference_fps", 30)))
    n = min(velocities.shape[0], int(_cfg(config, "inference_size", 16)))
    steps = []
    for i in range(n):
        v, w = float(velocities[i, 0]), float(velocities[i, 1])
        steps.append(CommandStep(dx=v * dt, dy=0.0, dtheta=w * dt, dt=(i + 1) * dt))
    return steps


inference_fn = nomad_topomap_inference

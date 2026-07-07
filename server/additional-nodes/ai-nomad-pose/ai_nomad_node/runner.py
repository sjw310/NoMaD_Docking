"""
AINomadRunner — RunnerBase + 두 모드.

navigate : P2(topomap NoMaD 추론)가 준 horizon → 속도 궤적 보정 → AiCommand(MQTT).
           트레젝토리 보정(seq/anchor/stride/padding/is_final)은 ai-control runner 와 byte 동등.
scan     : LiveKit 관측 프레임을 scan_interval 마다 topomap_dir 에 저장(P2 추론 안 함).
           프레임 픽셀은 P1 의 SHM ring 에서 attach_frames 로 읽는다(LiveKit 수신이 P1 ring 에 write).
"""
import logging
import os
import time
from typing import List, Optional

from node_sdk import CommandStep
from node_sdk.runner_base import RunnerBase

from ai_models.topomap import save_frame_to_topomap
from .config import AINomadConfig
from .trajectory_payload import ai_command_velocity_steps_dict, steps_to_velocity_steps

logger = logging.getLogger(__name__)

_STREAM_TO_CHANNEL = {"video1": 0, "video2": 1}


class AINomadRunner(RunnerBase):
    def __init__(self, config: AINomadConfig, hub, control_q, result_q, shm_ring=None, livekit_receiver=None):
        super().__init__(config, hub, control_q, result_q, shm_ring, livekit_receiver)
        # 트레젝토리 보정 상태 (navigate)
        self._steps_buffer: List[CommandStep] = []
        self._steps_index = 0
        self._seq_num = 0
        self._trajectory_anchor = 0.0
        self._robot_send_count = 0
        self._has_new_inference = False
        self._last_observation_time = 0.0
        # 스캔 상태
        self._scan_channel = _STREAM_TO_CHANNEL.get(config.nomad_obs_stream, 0)
        self._scan_index = 0
        self._scan_last_ts = 0.0           # time.monotonic 기준 (프레임 ts 아님)
        self._scan_seen_channels: set = set()
        if config.nomad_mode == "scan" and getattr(config, "scan_reset", False):
            self._reset_topomap_dir()

    # ── register ──
    def get_register_payload(self) -> dict:
        cfg = self.config
        rid = cfg.robot_id
        sending = cfg.enable_send and cfg.nomad_mode == "navigate"
        return {
            "type": "register",
            "plugin_id": cfg.plugin_id or "ai-nomad",
            "instance_id": cfg.instance_id or (cfg.plugin_id or "ai-nomad"),
            "robot_id": rid,
            "robot_scope": "single",
            "run_mode": "on_demand",
            "scale_mode": "fan_out",
            "trigger": {"type": "stream"},
            "required_data": {"sensors": []},          # NoMaD는 카메라(LiveKit)만 사용
            "output_mode": ["mqtt"] if sending else [],
            "managed_resources": ["robot_control"] if sending else [],
            "heartbeat_interval_sec": 5,
            "mqtt_topics": [f"robot/{rid}/ai-command"] if (rid and sending) else [],
            "livekit_tracks": [
                {"track_name": cfg.livekit_tracks_video1, "label": cfg.livekit_tracks_video1, "allowed_modes": ["subscribe"], "default_mode": "subscribe"},
                {"track_name": cfg.livekit_tracks_video2, "label": cfg.livekit_tracks_video2, "allowed_modes": ["subscribe"], "default_mode": "subscribe"},
            ],
        }

    # ── 스캔: P2 추론 요청 안 함 ──
    def _maybe_request_inference(self) -> None:
        if self.config.nomad_mode == "scan":
            return
        super()._maybe_request_inference()

    # ── 스캔: LiveKit 프레임 저장 ──
    def on_video_frame(self, channel: int, slot: int, ts_sec: float) -> None:
        super().on_video_frame(channel, slot, ts_sec)   # windows.push_video (정상 흐름 유지)
        if self.config.nomad_mode != "scan":
            return
        # 진단: 채널별 첫 프레임 수신 1회 로깅 (프레임이 실제로 들어오는지 확인)
        if channel not in self._scan_seen_channels:
            self._scan_seen_channels.add(channel)
            logger.info("scan: 프레임 수신 채널=%d (저장 대상=%d)", channel, self._scan_channel)
        if channel != self._scan_channel:
            return
        # 간격은 프레임 ts(robot timestamp_us=0이면 0)가 아니라 **벽시계(monotonic)** 로 잰다.
        now = time.monotonic()
        if self._scan_last_ts > 0.0 and now - self._scan_last_ts < max(0.05, float(self.config.scan_interval_sec)):
            return
        if self.shm_ring is None:
            return
        try:
            frame = self.shm_ring.attach_frames(channel, [slot])[0]   # BGR (H,W,3)
            path = save_frame_to_topomap(self.config.topomap_dir, self._scan_index, frame)
            logger.info("scan: topomap 노드 %d 저장 → %s", self._scan_index, path)
            self._scan_index += 1
            self._scan_last_ts = now
        except Exception:
            logger.exception("scan 프레임 저장 실패")

    def _reset_topomap_dir(self) -> None:
        d = self.config.topomap_dir
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass
        logger.info("scan: topomap_dir 초기화 %s", d)

    # ── navigate: send_tick ──
    async def on_send_tick(self) -> None:
        if self.config.nomad_mode == "scan":
            return
        res = self.consume_result()
        if res is not None:
            horizon = [CommandStep.from_dict(s) for s in res.get("horizon", [])][: self.config.inference_size]
            if horizon:
                self._steps_buffer = horizon
                self._steps_index = 0
                obs = res.get("observation_time")
                self._last_observation_time = obs if obs is not None else self.robot_time_sec(0.0)
                self._has_new_inference = True
        outputs = self._build_send_tick_outputs()
        if outputs:
            await self.hub.send_output(outputs)

    def _anchor_now(self) -> float:
        return self.robot_time_sec(self.config.prediction_delay_sec)

    def _make_outputs(self, ts, velocity_steps, seq_num, is_final) -> List[dict]:
        cfg = self.config
        if cfg.enable_send and velocity_steps:
            return [{
                "delivery": "mqtt",
                "topic": cfg.ai_command_topic,
                "payload": ai_command_velocity_steps_dict(ts, velocity_steps, seq_num, is_final),
            }]
        return []

    def _build_send_tick_outputs(self) -> List[dict]:
        """ai-control _send_tick byte 동등 (overlay 제외, mqtt만)."""
        cfg = self.config
        send_interval = max(0.01, cfg.send_interval_sec)
        fps = max(1, cfg.inference_fps)
        step_dt = 1.0 / fps
        action_horizon = getattr(cfg, "action_horizon", None)
        if action_horizon is None or action_horizon < 1:
            action_horizon = max(1, round(send_interval * fps))
        stride_steps = max(1, round(send_interval * fps))

        now_robot = self.robot_time_sec(0.0)
        horizon = list(self._steps_buffer) if self._steps_buffer else []
        has_new = self._has_new_inference
        last_obs = self._last_observation_time
        idx = self._steps_index

        if not horizon:
            if getattr(cfg, "no_send_when_no_horizon", False):
                return []
            ts = self._anchor_now()
            zero_steps = [{"vx": 0.0, "wz": 0.0, "dt": step_dt} for _ in range(action_horizon)]
            outs = self._make_outputs(ts, zero_steps, self._seq_num, True)
            self._seq_num = (self._seq_num + 1) & 0xFFFF
            return outs

        if has_new:
            fixed_pad = getattr(cfg, "observation_send_padding_sec", None)
            if fixed_pad is not None:
                padding_sec = max(0.0, float(fixed_pad))
            else:
                padding_sec = max(0.0, now_robot - last_obs + 0.1)
            cap_mult = max(0.0, float(getattr(cfg, "send_padding_cap_interval_multiplier", 2.0)))
            padding_sec = min(padding_sec, send_interval * cap_mult)
            reflect_small = bool(getattr(cfg, "send_padding_reflect_small_latency", False))
            if padding_sec < send_interval and not reflect_small:
                padding_steps = 0
            else:
                padding_steps = min(int(round(padding_sec / step_dt)), len(horizon) - 1)
                padding_steps = max(0, padding_steps)
            start_index = padding_steps
        else:
            start_index = idx

        start_index = min(start_index, max(0, len(horizon) - action_horizon))

        if start_index >= len(horizon):
            self._last_observation_time = now_robot
            ts = self._anchor_now()
            zero_steps = [{"vx": 0.0, "wz": 0.0, "dt": step_dt} for _ in range(action_horizon)]
            outs = self._make_outputs(ts, zero_steps, self._seq_num, True)
            self._seq_num = (self._seq_num + 1) & 0xFFFF
            return outs

        self._has_new_inference = False
        self._steps_index = start_index + stride_steps

        velocity_steps = steps_to_velocity_steps(horizon, step_dt, start_index, action_horizon)
        while len(velocity_steps) < action_horizon:
            velocity_steps.append({"vx": 0.0, "wz": 0.0, "dt": step_dt})

        init_pad = getattr(cfg, "initial_send_padding_sec", 0.0)
        if init_pad > 0:
            remaining_sec = max(0.0, init_pad - (self._robot_send_count * send_interval))
            if remaining_sec > 0:
                pad_steps = min(int(round(remaining_sec / step_dt)), action_horizon - 1)
                if pad_steps > 0:
                    zero_pad = [{"vx": 0.0, "wz": 0.0, "dt": step_dt} for _ in range(pad_steps)]
                    velocity_steps = zero_pad + velocity_steps[: action_horizon - pad_steps]

        all_zero = all(s.get("vx", 0.0) == 0.0 and s.get("wz", 0.0) == 0.0 for s in velocity_steps)
        next_index = start_index + stride_steps
        is_final = all_zero or (next_index >= len(horizon))

        if has_new:
            self._trajectory_anchor = self._anchor_now()
        ts = self._trajectory_anchor + start_index * step_dt

        outs = self._make_outputs(ts, velocity_steps, self._seq_num, is_final)
        if cfg.enable_send and velocity_steps:
            self._seq_num = (self._seq_num + 1) & 0xFFFF
            self._robot_send_count += 1
        return outs

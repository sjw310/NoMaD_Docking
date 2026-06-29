"""
ai-nomad 전용 — ai-command MQTT 페이로드 빌더 (ai-control trajectory_payload 와 동일 로직).

velocity CommandStep horizon → AiCommand 메시지의 byte 동등성을 보존한다 (로봇 도킹 펌웨어 호환).
"""
from typing import List

from node_sdk import CommandStep

PROTOCOL_VERSION = 1
TRAJ_MAX_STEPS = 150
IS_FINAL_FLAG = 0x01


def steps_to_velocity(horizon: List[CommandStep], index: int, step_dt: float):
    """horizon[index] 구간 속도 (vx, wz). velocity 인코딩(dx=vx*dt, dtheta=wz*dt) 또는 누적위치 모두 처리."""
    if not horizon or index < 0 or index >= len(horizon):
        return (0.0, 0.0)
    step = horizon[index]
    t = step.dt if step.dt > 1e-9 else step_dt
    if t < 1e-9:
        return (0.0, 0.0)
    if index == 0:
        return (step.dx / t, step.dtheta / t)
    prev = horizon[index - 1]
    dt_delta = step.dt - prev.dt
    if abs(dt_delta) < 1e-9:
        return (step.dx / t, step.dtheta / t)
    return ((step.dx - prev.dx) / dt_delta, (step.dtheta - prev.dtheta) / dt_delta)


def steps_to_velocity_steps(horizon: List[CommandStep], step_dt: float, start_index: int = 0, count=None) -> List[dict]:
    if not horizon or start_index < 0 or start_index >= len(horizon):
        return []
    end = start_index + (count if count is not None else len(horizon) - start_index)
    end = min(end, len(horizon))
    out = []
    for i in range(start_index, end):
        vx, wz = steps_to_velocity(horizon, i, step_dt)
        step = horizon[i]
        dt_sec = step.dt if i == 0 else (step.dt - horizon[i - 1].dt)
        if dt_sec < 1e-9:
            dt_sec = step_dt
        out.append({"vx": vx, "wz": wz, "dt": dt_sec})
    return out


def ai_command_velocity_steps_dict(anchor_time: float, steps_list: List[dict], seq_num: int = 0, is_final: bool = False) -> dict:
    n = min(len(steps_list), TRAJ_MAX_STEPS)
    return {
        "version": PROTOCOL_VERSION,
        "num_steps": n,
        "seq_num": seq_num & 0xFFFF,
        "flags": IS_FINAL_FLAG if is_final else 0,
        "anchor_time": anchor_time,
        "steps": steps_list[:n],
    }

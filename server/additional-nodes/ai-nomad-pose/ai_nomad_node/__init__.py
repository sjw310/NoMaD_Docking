"""ai-nomad 노드 — 학습된 velocity-NoMaD 를 rc 노드 그래프로 (topomap 주행 + 스캔).

ai-control 과 **독립**(별도 컨테이너·노드 정체성 "ai-nomad"). 전송계층/2-프로세스/SHM ring/궤적보정은
node_sdk + ai-control 패턴을 그대로 따르고, 모델 경계만 NoMaD topomap 추론으로 채운다.

모드(config.nomad_mode):
  - scan     : LiveKit 관측 프레임을 topomap_dir 에 저장(경로 1회 기록). 명령 미발행.
  - navigate : topomap + dist_pred 로 subgoal 추종 → 속도 궤적 → AiCommand(MQTT robot/{id}/ai-command).

비교 운영: enable_send=false 로 띄우면 제어권 미점유·명령 미발행(추론만) → ai-control(다른 모델)과 동시 구동 가능.
"""
import os

from ai_nomad_node.ai_process import ai_main
from ai_nomad_node.config import load_ai_nomad_config
from ai_nomad_node.runner import AINomadRunner

from node_sdk import NodeSpec, run_node_main


def build_spec(config) -> NodeSpec:
    sending = config.enable_send and config.nomad_mode == "navigate"
    outputs = []
    if sending:
        outputs = [{
            "name": "ai_command",
            "topic": "robot.{robot_id}.ai_command",
            "type": "AiCommand@1",
            "requires_keys": ["robot_id"],
        }]
    return NodeSpec(
        name="ai-nomad",
        label="AI NoMaD (도킹/주행)",
        tags=["ai", "control", "nomad"],
        sensor_inputs=[],                 # NoMaD는 카메라(LiveKit)만 사용 — 버스 센서 구독 없음
        outputs=outputs,
        control_lease=sending,            # 명령 보낼 때만 제어권 lease
        control_owner_key="ai-nomad",
    )


def main() -> None:
    config = load_ai_nomad_config(config_path=os.environ.get("CONFIG_PATH", "config.yml"))
    run_node_main(
        runner_factory=lambda c, cq, rq, ring: AINomadRunner(
            c, hub=None, control_q=cq, result_q=rq, shm_ring=ring
        ),
        config=config,
        ai_entry=ai_main,
        spec=build_spec(config),
    )

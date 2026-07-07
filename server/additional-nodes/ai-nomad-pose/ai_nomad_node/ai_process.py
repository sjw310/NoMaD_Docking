"""ai-nomad P2 진입점. 공통 run_ai_main + horizon build_result (ai-control ai_process 동등)."""
from node_sdk.ai_runtime import run_ai_main

from .config import AINomadConfig


def make_build_result(inference_fn):
    def build_result(snapshot, latency_marks, config):
        horizon = inference_fn(snapshot, latency_marks, config)
        return {"horizon": [s.to_dict() for s in horizon]}
    return build_result


def ai_main(config_dict, control_q, result_q, ring_meta, lock):
    run_ai_main(AINomadConfig(**config_dict), control_q, result_q, ring_meta, lock, make_build_result)

# ai-nomad — 학습된 velocity-NoMaD 노드 (topomap 주행 + 스캔)

이 repo에서 학습한 **velocity 예측형 NoMaD**(`train/config/nomad.yaml`, `predict_velocity: True`)를
rc 노드 그래프에 붙인 **독립 노드**. `ai-control`(다른 도킹 모델)과 분리되어 있어 **성능 비교**에
적합하다 — 같은 LiveKit 데이터로 동시 구동 가능(추론 전용 모드).

진짜 NoMaD 흐름(topomap)을 ROS 없이 이식: [navigate.py](../../../deployment/src/navigate.py) ·
[create_topomap.py](../../../deployment/src/create_topomap.py) 로직 기반.

## 두 모드

| 모드 | 하는 일 | 출력 |
|---|---|---|
| **scan** | 경로를 한 번 주행하며 LiveKit 관측 프레임을 `topomap_dir`에 `0.png,1.png,...` 저장 | 없음(파일만) |
| **navigate** | topomap + `dist_pred_net`으로 현재 위치(closest node) 찾고 subgoal 추종 → 속도 궤적 | MQTT `robot/{id}/ai-command` |

## 데이터 파이프라인 (navigate)

```
[로봇] ─LiveKit 영상─> connection ─버스─> [ai-nomad 노드]
  P1: 관측 프레임 → windows (스캔이면 여기서 topomap 저장)
  P2: run_nomad_topomap.inference_fn
       TopomapNavigator(NoMaDPolicy + topomap)
        → 후보 goal(closest±radius) encode → dist → closest 갱신 → subgoal diffusion
        → (60,2) 속도 (v,w)
  P1: CommandStep → 궤적 보정(ai-control 동등) → AiCommand
        ─버스 robot.{id}.ai_command─> connection ─MQTT robot/{id}/ai-command─> [로봇 도킹]
```

## 구조

```
ai_nomad_node/      # 노드(전송/2-프로세스/궤적보정) — node_sdk + ai-control 패턴
  __init__.py         build_spec(name="ai-nomad") + main
  config.py           AINomadConfig (모드/모델/topomap/스캔 필드)
  runner.py           AINomadRunner — navigate(궤적) + scan(프레임 저장, SHM ring read)
  ai_process.py       P2 진입 (run_ai_main)
  trajectory_payload.py  속도 → AiCommand (byte 동등)
ai_models/          # 모델 경계 (전송 무관)
  nomad_infer.py      NoMaDPolicy: 로드/전처리/diffusion/navigate_topomap
  topomap.py          topomap I/O + TopomapNavigator(closest_node 상태 유지)
  plugins/run_nomad_topomap.py  inference_fn (navigate)
  scripts/run_navigate_standalone.py  오프라인 검증 (서버 불필요)
config.yml · docker-compose.yml
```

## 실행

### 0) 오프라인 검증 (서버/도커 없이 — 권장 first)
```bash
conda run -n nomad_train python \
  server/additional-nodes/ai-nomad/ai_models/scripts/run_navigate_standalone.py
# h5로 topomap 만들고 주행 → closest_node 가 0→끝(reached)로 전진하면 정상.
```

### 1) 스캔 (경로 1회 기록)
```bash
cd server/additional-nodes/ai-nomad
NOMAD_MODE=scan ENABLE_SEND=false docker compose up -d --build && docker compose logs -f
# 로봇을 도킹 경로로 주행 → ./topomap/0.png,1.png,... 생성. 충분하면 docker compose down.
```

### 2) 주행 (도킹)
```bash
NOMAD_MODE=navigate ENABLE_SEND=true docker compose up -d --build
mosquitto_sub -h <mqtt_host> -t 'robot/+/ai-command' -v   # 발행 확인
```

### 비교 (ai-control 과 동시)
`ENABLE_SEND=false` 로 띄우면 제어권 미점유·명령 미발행(추론만) → ai-control이 실제 조종하는 동안
NoMaD 출력을 로그로 비교. 전제: rc 스택 가동(`cd ../../rc_server && ./scripts/local.sh up`).

## 학습과 맞춘 핵심 (정확도)
- 이미지: `/255 + resize[240,320]`, **ImageNet 정규화 없음** (학습 `_load_image` 동등).
- context: `context_size+1=6`장 채널 concat.
- 역정규화: `(v,w)=(norm+1)/2·scale+min`, min=[-0.2649721,-0.7315362] max=[0.2977743,0.30039853]
  (학습 로그 `[Action normalization]` 동일). 재학습 시 `nomad_infer.DEFAULT_ACTION_*` 갱신.

## 운영 확인 (env)
- `NOMAD_OBS_STREAM`(video1|video2): 학습 `image_bottom` 카메라에 해당하는 LiveKit 트랙.
- `NOMAD_BGR_TO_RGB`(기본 true): LiveKit BGR↔학습 RGB. 색 이상하면 false.
- `NOMAD_GOAL_NODE`(기본 -1=마지막), `NOMAD_RADIUS`, `nomad_close_threshold`: topomap 추종 튜닝.

## 이미지 의존 (도커)
`diffusers`, `efficientnet_pytorch`, `vint_train`, `diffusion_policy` 필요. 뒤 둘은 compose가
`/opt/nomad`로 마운트. 앞 둘이 `ai-node-base`에 없으면 `additional-nodes/Dockerfile`에 추가 후 재빌드.

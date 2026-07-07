"""
ai-nomad 주행 로직 오프라인 검증 + 결과 아티팩트 저장 (서버/ROS 불필요).

h5 한 에피소드로 topomap(이미지 시퀀스)을 만든 뒤, 에피소드를 따라 이동하는 관측을 흘려보내
TopomapNavigator.closest_node 가 0→끝으로 단조 전진하는지(=localization 동작) + subgoal 속도를 확인하고,
--out-dir 에 결과(plot/montage/json)를 저장한다.

  conda run -n nomad_train python \
    server/additional-nodes/ai-nomad/ai_models/scripts/run_navigate_standalone.py --threads 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_THIS = os.path.abspath(__file__)
_AI_NOMAD = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, _AI_NOMAD)

from ai_models.nomad_infer import NoMaDParams, NoMaDPolicy, velocities_to_trajectory  # noqa: E402
from ai_models.topomap import TopomapNavigator, save_frame_to_topomap, load_topomap_frames  # noqa: E402


def _repo_root() -> str:
    d = _THIS
    for _ in range(10):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "train")) and os.path.isdir(os.path.join(d, "dataset")):
            return d
    return os.getcwd()


def chw_rgb_to_hwc_bgr(chw_rgb: np.ndarray) -> np.ndarray:
    """h5 (3,H,W) RGB → (H,W,3) BGR (LiveKit/디스크 색순서 시뮬레이션)."""
    return chw_rgb.transpose(1, 2, 0)[:, :, ::-1].copy()


def _bgr_to_rgb_disp(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, ::-1]


def save_topomap_montage(topo_frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(topo_frames)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.4))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(_bgr_to_rgb_disp(topo_frames[i]))
            ax.set_title(f"node {i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"topomap ({n} nodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_navigation_plots(records, out_dir, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) closest_node 진행 그래프
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fracs = [r["frac"] for r in records]
    ax[0].plot(fracs, [r["closest_node"] for r in records], "-o", label="closest")
    ax[0].plot(fracs, [r["subgoal_node"] for r in records], "--s", label="subgoal", alpha=0.6)
    ax[0].set_xlabel("관측 위치(경로 fraction)"); ax[0].set_ylabel("node")
    ax[0].set_title("localization: closest/subgoal node"); ax[0].grid(True); ax[0].legend()
    ax[1].plot(fracs, [r["closest_distance"] for r in records], "-o", color="tab:red")
    ax[1].set_xlabel("관측 위치(경로 fraction)"); ax[1].set_ylabel("dist (norm)")
    ax[1].set_title("dist_pred (closest까지)"); ax[1].grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "localization.png"), dpi=120); plt.close(fig)

    # (2) 관측별 속도 horizon + 적분 경로
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.4 * n))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(records):
        v = np.array(r["velocities"])
        axes[i, 0].plot(v[:, 0], label="v[m/s]"); axes[i, 0].plot(v[:, 1], label="w[rad/s]")
        axes[i, 0].set_title(f"frac {r['frac']:.2f}: velocity horizon", fontsize=8)
        axes[i, 0].grid(True); axes[i, 0].legend(fontsize=7)
        traj = velocities_to_trajectory(v, dt)
        axes[i, 1].plot(traj[:, 0], traj[:, 1], "-o", ms=2)
        axes[i, 1].set_title(f"frac {r['frac']:.2f}: integrated path", fontsize=8)
        axes[i, 1].axis("equal"); axes[i, 1].grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "velocity_paths.png"), dpi=110); plt.close(fig)


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        root, "train/logs/nomad_train_after_0328/nomad_dist_max_norm_p60_a100_d100_2026_06_18_20_42_07/ema_18.pth"))
    ap.add_argument("--h5", default=os.path.join(root, "dataset/after_0328.h5"))
    ap.add_argument("--image-key", default="image_bottom")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--topomap-spacing", type=int, default=40)
    ap.add_argument("--context-spacing", type=int, default=6)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--device", default=None, help="cpu | cuda (기본: 자동)")
    ap.add_argument("--threads", type=int, default=0, help="CPU 스레드 상한(0=기본). CPU 추론 시 멈춤 방지")
    ap.add_argument("--out-dir", default=None, help="결과 저장 폴더 (기본: nomad_eval/standalone_<ts>)")
    args = ap.parse_args()

    import torch
    import h5py
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 1.0 / 30.0

    out_dir = args.out_dir or os.path.join(os.getcwd(), "nomad_eval", f"standalone_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    topodir = os.path.join(out_dir, "topomap")

    params = NoMaDParams()
    print(f"[1/4] 모델 로드: {args.ckpt}")
    print(f"      device={device}  (cuda 가용: {torch.cuda.is_available()}, threads={torch.get_num_threads()})")
    if device.type == "cpu":
        print("      ⚠️ CPU 모드 — 느리고 모든 코어 점유 가능. --threads 4 권장, 또는 GPU 머신에서 실행.")
    policy = NoMaDPolicy(args.ckpt, params=params, device=device)

    with h5py.File(args.h5, "r") as f:
        images = f[args.image_key]
        ends = f["episode_ends"][:]
        ep_start = 0 if args.episode == 0 else int(ends[args.episode - 1])
        ep_end = int(ends[args.episode])
        print(f"[2/4] 에피소드 {args.episode}: frames [{ep_start}, {ep_end}) (len {ep_end-ep_start})")

        node_times = list(range(ep_start, ep_end, args.topomap_spacing))
        for i, t in enumerate(node_times):
            save_frame_to_topomap(topodir, i, chw_rgb_to_hwc_bgr(np.asarray(images[t])))
        print(f"[3/4] topomap {len(node_times)} 노드 저장: {topodir}")

        topo_frames = load_topomap_frames(topodir)
        nav = TopomapNavigator(policy, topo_frames, goal_node=-1, radius=args.radius,
                               num_samples=args.num_samples, bgr_to_rgb=True)

        print(f"[4/4] 주행 시뮬레이션 (goal_node={nav.goal_node}):")
        fracs = [0.0, 0.25, 0.5, 0.75, 0.95]
        cs = params.context_size
        records = []
        for fr in fracs:
            cur = int(ep_start + fr * (ep_end - ep_start - 1))
            ctx_times = [max(ep_start, cur - i * args.context_spacing) for i in range(cs, -1, -1)]
            context = [chw_rgb_to_hwc_bgr(np.asarray(images[t])) for t in ctx_times]
            res = nav.step(context)
            v = res.velocities
            print(f"  obs@frac={fr:.2f} (frame {cur:6d}) → closest={res.closest_node:2d}/{nav.goal_node} "
                  f"subgoal={res.subgoal_node:2d} dist={res.closest_distance:.3f} "
                  f"reached={res.reached_goal} | v_mean={v[:,0].mean():+.3f} w_mean={v[:,1].mean():+.3f}")
            records.append({
                "frac": fr, "frame": cur, "closest_node": int(res.closest_node),
                "subgoal_node": int(res.subgoal_node), "closest_distance": res.closest_distance,
                "reached_goal": bool(res.reached_goal),
                "v_mean": float(v[:, 0].mean()), "w_mean": float(v[:, 1].mean()),
                "velocities": v.tolist(),
            })

    # ── 아티팩트 저장 ──
    try:
        save_topomap_montage(topo_frames, os.path.join(out_dir, "topomap_montage.png"))
        save_navigation_plots(records, out_dir, dt)
    except Exception as e:
        print(f"  (plot 저장 실패, 무시): {e}")
    summary = {
        "ckpt": args.ckpt, "h5": args.h5, "episode": args.episode,
        "device": str(device), "num_nodes": len(topo_frames), "goal_node": nav.goal_node,
        "params": {"context_size": params.context_size, "len_traj_pred": params.len_traj_pred,
                   "num_samples": args.num_samples, "radius": args.radius,
                   "topomap_spacing": args.topomap_spacing, "context_spacing": args.context_spacing},
        "monotonic_closest": [r["closest_node"] for r in records],
        "records": [{k: r[k] for k in r if k != "velocities"} for r in records],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 결과 저장: {out_dir}")
    print("  - topomap/ (노드 이미지)   - topomap_montage.png")
    print("  - localization.png (closest/subgoal/dist)   - velocity_paths.png   - results.json")
    cl = summary["monotonic_closest"]
    print(f"  closest_node 진행: {cl}  →  {'단조 증가(정상)' if all(b>=a for a,b in zip(cl,cl[1:])) else '비단조(점검)'}")


if __name__ == "__main__":
    main()

"""
성능 분석 metric — 학습된 NoMaD의 행동 예측을 **녹화 데이터의 정답(encoder)** 과 비교.

학습 dataset(vint_dataset_episode.ViNT_H5_Action_Dataset)을 그대로 써서 (obs, goal, 정답 actions)
삼중쌍을 학습과 **동일 전처리**로 뽑고, 모델을 goal-conditioned 로 돌려 예측 (v,w) 궤적을 비교한다.

리포트:
  - Velocity MAE : v[m/s], w[rad/s]  (mean-sample, best-of-K)
  - Trajectory   : ADE/FDE [m]       (속도 RK4 적분 경로, mean-sample, best-of-K)
  - Distance     : dist_pred vs 정답 temporal distance
  - Baseline 비교: 정지(0속도) / 데이터평균속도 예측 → NoMaD 가 naive 대비 우월한지

  conda run -n nomad_train python \
    server/additional-nodes/ai-nomad/ai_models/scripts/eval_metrics.py --num 80 --diffusion-samples 4

다른 모델과 비교: --out result.json 으로 저장 후 두 모델 결과를 diff. (다른 모델은 자체 추론으로
같은 (obs,goal,정답)에 대해 같은 지표를 계산하면 동일 축으로 비교 가능.)
"""
import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.abspath(__file__)
_AI_NOMAD = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, _AI_NOMAD)


def _repo_root() -> str:
    d = _THIS
    for _ in range(10):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "train")) and os.path.isdir(os.path.join(d, "dataset")):
            return d
    return os.getcwd()


ROOT = _repo_root()
sys.path.insert(0, os.path.join(ROOT, "train"))

from ai_models.nomad_infer import NoMaDParams, NoMaDPolicy, velocities_to_trajectory  # noqa: E402


def trajectory_errors(pred_vel: np.ndarray, gt_vel: np.ndarray, dt: float):
    """예측/정답 속도 → RK4 적분 경로 → ADE(평균), FDE(끝점) [m]."""
    pt = velocities_to_trajectory(pred_vel, dt)[1:, :2]
    gt = velocities_to_trajectory(gt_vel, dt)[1:, :2]
    d = np.linalg.norm(pt - gt, axis=1)
    return float(d.mean()), float(d[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "train/logs/nomad_train_after_0328/nomad_dist_max_norm_p60_a100_d100_2026_06_18_20_42_07/ema_18.pth"))
    ap.add_argument("--h5", default=os.path.join(ROOT, "dataset/after_0328.h5"))
    ap.add_argument("--image-key", default="image_bottom")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--num", type=int, default=80, help="평가 샘플 수")
    ap.add_argument("--diffusion-samples", type=int, default=4, help="샘플당 diffusion 추출 수 K")
    ap.add_argument("--max-dist-cat", type=int, default=100)
    ap.add_argument("--context-spacing", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="요약 JSON 저장")
    args = ap.parse_args()

    import torch
    from vint_train.data.vint_dataset_episode import ViNT_H5_Action_Dataset

    params = NoMaDParams()
    dt = 1.0 / 30.0
    rng = np.random.RandomState(args.seed)

    print(f"[1/3] 모델 로드: {args.ckpt}")
    policy = NoMaDPolicy(args.ckpt, params=params)

    print(f"[2/3] dataset 구성: {args.h5} (split={args.split})")
    ds = ViNT_H5_Action_Dataset(
        h5_path=args.h5, image_key=args.image_key, action_key="encoder",
        image_size=(params.image_width, params.image_height), split=args.split,
        context_spacing=args.context_spacing, waypoint_spacing=1,
        min_dist_cat=0, max_dist_cat=args.max_dist_cat,
        min_action_distance=0, max_action_distance=args.max_dist_cat,
        negative_mining=False, len_traj_pred=params.len_traj_pred,
        context_size=params.context_size, normalize=True, predict_velocity=True,
    )
    # 모델과 동일한 action_stats 로 정답 정규화 일치 보장
    ds.action_stats = {"min": policy.action_min, "max": policy.action_max,
                       "scale": (policy.action_max - policy.action_min)}
    n_total = len(ds)
    idxs = rng.choice(n_total, size=min(args.num, n_total), replace=False)
    print(f"      total {n_total} 샘플 중 {len(idxs)}개 평가, K={args.diffusion_samples}")

    # 누적기
    acc = {k: [] for k in ["v_mae", "w_mae", "ade", "fde",
                           "v_mae_best", "w_mae_best", "ade_best", "fde_best",
                           "dist_abs", "zero_fde", "mean_fde"]}
    dmin, dscale = policy.action_min, (policy.action_max - policy.action_min)
    mean_vel = ((np.array([0.0, 0.0]) + 1) / 2) * dscale + dmin  # norm 0 → 평균근방(참고 baseline)

    print(f"[3/3] 평가 중...")
    used = 0
    for i in idxs:
        obs_img, goal_img, gt_actions, gt_dist, *_rest, action_mask, ep_idx, curr_time = ds[int(i)]
        if float(action_mask) < 0.5:
            continue
        gt_norm = gt_actions.numpy()                      # (T,2) [-1,1]
        gt_vel = policy._denormalize(gt_norm)             # (T,2) physical
        out = policy.infer_tensors(obs_img.unsqueeze(0), goal_img.unsqueeze(0),
                                   masked=False, num_samples=args.diffusion_samples)
        pred_all = out["velocities_all"]                  # (K,T,2) physical

        # mean-sample 예측
        pred_mean = pred_all.mean(axis=0)
        acc["v_mae"].append(np.abs(pred_mean[:, 0] - gt_vel[:, 0]).mean())
        acc["w_mae"].append(np.abs(pred_mean[:, 1] - gt_vel[:, 1]).mean())
        ade, fde = trajectory_errors(pred_mean, gt_vel, dt)
        acc["ade"].append(ade); acc["fde"].append(fde)

        # best-of-K (FDE 최소 샘플)
        per = [trajectory_errors(pred_all[k], gt_vel, dt) for k in range(pred_all.shape[0])]
        bk = int(np.argmin([p[1] for p in per]))
        acc["v_mae_best"].append(np.abs(pred_all[bk][:, 0] - gt_vel[:, 0]).mean())
        acc["w_mae_best"].append(np.abs(pred_all[bk][:, 1] - gt_vel[:, 1]).mean())
        acc["ade_best"].append(per[bk][0]); acc["fde_best"].append(per[bk][1])

        # distance
        gt_dist_norm = float(gt_dist) / args.max_dist_cat
        if np.isfinite(out["distance"]):
            acc["dist_abs"].append(abs(out["distance"] - gt_dist_norm))

        # baselines
        zero_vel = np.zeros_like(gt_vel)
        acc["zero_fde"].append(trajectory_errors(zero_vel, gt_vel, dt)[1])
        mv = np.tile(mean_vel, (gt_vel.shape[0], 1))
        acc["mean_fde"].append(trajectory_errors(mv, gt_vel, dt)[1])
        used += 1

    def stat(name):
        a = np.array(acc[name]) if acc[name] else np.array([np.nan])
        return float(a.mean()), float(a.std())

    print(f"\n==== 평가 결과 ({used} 샘플, horizon {params.len_traj_pred} @30Hz) ====")
    print(f"{'metric':<26}{'mean':>10}{'std':>10}")
    rows = [
        ("Velocity v MAE [m/s]", "v_mae"), ("Velocity w MAE [rad/s]", "w_mae"),
        ("Traj ADE [m]", "ade"), ("Traj FDE [m]", "fde"),
        ("— best-of-K v MAE", "v_mae_best"), ("— best-of-K w MAE", "w_mae_best"),
        ("— best-of-K ADE [m]", "ade_best"), ("— best-of-K FDE [m]", "fde_best"),
        ("Distance |pred-gt| (norm)", "dist_abs"),
        ("[baseline] zero-vel FDE [m]", "zero_fde"),
        ("[baseline] mean-vel FDE [m]", "mean_fde"),
    ]
    summary = {}
    for label, key in rows:
        m, s = stat(key)
        summary[key] = {"mean": m, "std": s}
        print(f"{label:<26}{m:>10.4f}{s:>10.4f}")

    # 해석 도움말
    fde_m = summary["fde"]["mean"]; zero_m = summary["zero_fde"]["mean"]
    print(f"\n해석: NoMaD FDE {fde_m:.3f} m vs 정지 baseline {zero_m:.3f} m "
          f"→ {'개선(우월)' if fde_m < zero_m else '개선 없음(점검 필요)'}.")
    print("best-of-K 가 mean-sample 보다 크게 낮으면 모델이 multimodal(여러 그럴듯한 경로) 임.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "h5": args.h5, "split": args.split,
                       "used": used, "diffusion_samples": args.diffusion_samples,
                       "summary": summary}, f, indent=2)
        print(f"\n요약 저장: {args.out}")


if __name__ == "__main__":
    main()

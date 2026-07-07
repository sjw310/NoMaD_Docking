import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DPMSolverMultistepScheduler

# ROS
#import rospy
#from sensor_msgs.msg import Image
#from std_msgs.msg import Bool, Float32MultiArray
from utils import to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
import h5py

MODEL_CONFIG_PATH = "../config/models.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 학습 데이터 기준 정규화 통계 (train 로그 출력과 동일하게 직접 기입) ──
DEFAULT_ACTION_MIN = np.array([-0.17029296, -0.3425848], dtype=np.float32)
DEFAULT_ACTION_MAX = np.array([0.16505694, 0.30039853], dtype=np.float32)
DEFAULT_ACTION_SCALE = DEFAULT_ACTION_MAX - DEFAULT_ACTION_MIN

# TODO: train 로그의 [Global pose normalization] min/max [x,y,theta] 값으로 교체.
DEFAULT_POSE_MIN = np.array([-0.29979697, -0.49998415, -2.8248427], dtype=np.float32)
DEFAULT_POSE_MAX = np.array([0.29997006, 0.03298904, -0.11712503], dtype=np.float32)
DEFAULT_POSE_SCALE = DEFAULT_POSE_MAX - DEFAULT_POSE_MIN
"""
crop_x_min = -0.3
crop_x_max = 0.3
crop_y_min = -0.5
crop_y_max = 0.1

"""
crop_x_min = -100
crop_x_max = 100
crop_y_min = -100
crop_y_max = 100


# GLOBALS
context_queue = []
context_size = None

def main(args: argparse.Namespace):
    global context_size # 전역 변수 context_size -> main() 안에서 직접 수정

     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"] 
    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    action_key = "encoder"

    h5_path = "/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/train_episode_postech_260330_dock.h5"
    episode_num = 74
    image_key = "image_bottom"
    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "outputs",
            "navigation_simulation",
            f"episode_{episode_num:03d}",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        episode_ends = f["episode_ends"][:]
        ep_idx = episode_num - 1
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]

        goal_img_np = f[image_key][end_idx - 1]
        if goal_img_np.shape[0] == 3:
            goal_img_np = np.transpose(goal_img_np, (1, 2, 0))
        goal_img = PILImage.fromarray(goal_img_np.astype(np.uint8))

    reached_goal = False

    """
    # ROS
    # 현재 프로그램을 ROS에 EXPLORATION이라는 이름으로 등록
    rospy.init_node("EXPLORATION", anonymous=False)
    # 실행 주기 설정
    rate = rospy.Rate(RATE)

    # 토픽 구독, 새 Image 오면 callback_obs 실행
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")
    """

    # DDPMScheduler 만 여기서 설정, 나머지는 load_model에서 이미 설정됨
    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        """
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
        )
        """
    
    def sync_time():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()


    def unnormalize_action(norm_actions):
        return (
            (norm_actions + 1.0) / 2.0
            * DEFAULT_ACTION_SCALE
            + DEFAULT_ACTION_MIN
        )

    def unnormalize_pose(norm_pose):
        return (
            (norm_pose + 1.0) / 2.0
            * DEFAULT_POSE_SCALE
            + DEFAULT_POSE_MIN
        )

    def wrap_angle(theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi

    def align_final_pose_to_origin_y(traj):
        final_x, final_y, final_theta = traj[-1]
        rot_angle = -np.pi / 2 - final_theta
        c = np.cos(rot_angle)
        s = np.sin(rot_angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        xy = traj[:, :2] - np.array([final_x, final_y], dtype=np.float32)

        aligned = np.zeros_like(traj)
        aligned[:, :2] = xy @ rot.T
        aligned[:, 2] = wrap_angle(traj[:, 2] + rot_angle)
        return aligned

    def get_crop_start_offset(aligned_traj, episode_vel):
        x = aligned_traj[:, 0]
        y = aligned_traj[:, 1]
        outside = (
            (x < crop_x_min) | (x > crop_x_max) |
            (y < crop_y_min) | (y > crop_y_max)
        )
        outside_idx = np.where(outside)[0]

        if len(outside_idx) == 0:
            crop_start_offset = 0
        else:
            if len(outside_idx) == len(aligned_traj):
                return None
            crop_start_offset = int(outside_idx[-1] + 1)

        if crop_start_offset >= len(episode_vel):
            return None
        return crop_start_offset

    def extract_encoder_hist(actions, idx, start_idx):
        need = int(model_params.get("encoder_imu_context_size", 30))
        spacing = int(model_params.get("encoder_imu_context_spacing", 1))
        idxs = [idx - i * max(1, spacing) for i in range(need)]
        idxs = [i for i in idxs if i >= start_idx]
        if len(idxs) < need:
            idxs = idxs + [start_idx] * (need - len(idxs))
        rows = actions[list(reversed(idxs))].astype(np.float32)
        return torch.as_tensor(rows, dtype=torch.float32, device=device).unsqueeze(0)

    def traj_reconstruct_pose_rk4(
            linear_vels,
            angular_vels,
            dt=0.0333,
            initial_pose=(0.0, 0.0, 0.0),
        ):
            n_steps = len(linear_vels)
            trajectory = np.zeros((n_steps + 1, 3), dtype=np.float32)
            trajectory[0] = initial_pose

            def f(q, v, w):
                return np.array(
                    [v * np.cos(q[2]), v * np.sin(q[2]), w],
                    dtype=np.float32,
                )

            curr_q = np.array(initial_pose, dtype=np.float32)

            for i in range(n_steps):
                v, w = linear_vels[i], angular_vels[i]

                k1 = f(curr_q, v, w)
                k2 = f(curr_q + 0.5 * dt * k1, v, w)
                k3 = f(curr_q + 0.5 * dt * k2, v, w)
                k4 = f(curr_q + dt * k3, v, w)

                curr_q += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                curr_q[2] = (curr_q[2] + np.pi) % (2 * np.pi) - np.pi

                trajectory[i + 1] = curr_q

            return trajectory



    def reconstruct_pose_rk4(linear_vels, angular_vels, dt=0.0333, initial_pose=(0.0, 0.0, 0.0)):
        n_steps = len(linear_vels)
        trajectory = np.zeros((n_steps + 1, 3))
        trajectory[0] = initial_pose

        def f(q, v, w):
            v = -v   # <-- 부호 반전 (-선속이 전진)
            return np.array([v * np.cos(q[2]), v * np.sin(q[2]), w])

        curr_q = np.array(initial_pose, dtype=float)
        for i in range(n_steps):
            v, w = linear_vels[i], angular_vels[i]
            k1 = f(curr_q, v, w)
            k2 = f(curr_q + 0.5 * dt * k1, v, w)
            k3 = f(curr_q + 0.5 * dt * k2, v, w)
            k4 = f(curr_q + dt * k3, v, w)
            curr_q += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            curr_q[2] = (curr_q[2] + np.pi) % (2 * np.pi) - np.pi
            trajectory[i + 1] = curr_q
        return trajectory
    
    
    receding_horizon = model_params["len_traj_pred"] # 6*0.03 = 0.18

    # navigation loop
    with h5py.File(h5_path, "r") as f:

        images = f[image_key]
        actions = f[action_key]
        episode_ends = f["episode_ends"][:]

        ep_idx = episode_num - 1

        if ep_idx == 0:
            start_idx = 0
        else:
            start_idx = episode_ends[ep_idx - 1]

        end_idx = episode_ends[ep_idx]
        episode_vel = actions[start_idx:end_idx].astype(np.float32)
        episode_traj = traj_reconstruct_pose_rk4(
            linear_vels=episode_vel[:, 0],
            angular_vels=episode_vel[:, 1],
            dt=0.0333,
            initial_pose=(0.0, 0.0, 0.0),
        )
        aligned_episode_traj = align_final_pose_to_origin_y(episode_traj)
        crop_start_offset = get_crop_start_offset(aligned_episode_traj, episode_vel)
        if crop_start_offset is None:
            print(
                "No valid cropped segment for "
                f"x=[{crop_x_min}, {crop_x_max}], y=[{crop_y_min}, {crop_y_max}]"
            )
            return

        sim_start_idx = start_idx + crop_start_offset
        print(
            f"[Crop] episode={episode_num} "
            f"original=({start_idx}, {end_idx}) "
            f"cropped=({sim_start_idx}, {end_idx}) "
            f"offset={crop_start_offset}"
        )
        context_queue.clear()

        context_spacing = model_params["datasets"]["episode_postech_dock"]["context_spacing"]
        
        for t, img_idx in enumerate(range(sim_start_idx, end_idx, context_spacing)):

            img = images[img_idx]

            # CHW -> HWC
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            img = img.astype(np.uint8)
            curr_img = PILImage.fromarray(img)

            # ROS callback 대신 여기서 context_queue 업데이트
            context_queue.append(curr_img)

            # context size 유지
            if len(context_queue) > model_params["context_size"] + 1:
                context_queue.pop(0)

            if len(context_queue) > model_params["context_size"]:
                if model_params["model_type"] == "nomad":
                    obs_images = transform_images(
                        context_queue,
                        model_params["image_size"],
                        center_crop=False,
                    )

                    obs_images = torch.split(obs_images, 3, dim=1)
                    obs_images = torch.cat(obs_images, dim=1)
                    obs_images = obs_images.to(device)

                    mask = torch.zeros(1).long().to(device)
                    goal_mask = torch.ones(1).long().to(device)

                    goal_image = transform_images(
                        goal_img,
                        model_params["image_size"],
                        center_crop=False,
                    ).to(device)
                    encoder_hist = extract_encoder_hist(actions, img_idx, sim_start_idx)

                    with torch.no_grad():

                        t0 = sync_time()

                        obsgoal_cond = model(
                            "vision_encoder",
                            obs_img=obs_images,
                            goal_img=goal_image,
                            input_goal_mask=mask,
                            encoder_hist=encoder_hist,
                        )
                        obs_cond_uc = model(
                            "vision_encoder",
                            obs_img=obs_images,
                            goal_img=goal_image,
                            input_goal_mask=goal_mask,
                            encoder_hist=encoder_hist,
                        )

                        t1 = sync_time()

                        pose_norm = model(
                            "pose_pred_net",
                            obsgoal_cond=obsgoal_cond,
                        )
                        pose_raw = unnormalize_pose(to_numpy(pose_norm))[0]
                        gt_pose_raw = aligned_episode_traj[img_idx - start_idx]

                        t2 = sync_time()

                        obs_cond_gc = obsgoal_cond

                        if len(obs_cond_gc.shape) == 2:
                            obs_cond_gc = obs_cond_gc.repeat(args.num_samples, 1)
                        else:
                            obs_cond_gc = obs_cond_gc.repeat(args.num_samples, 1, 1)

                        if len(obs_cond_uc.shape) == 2:
                            obs_cond_uc = obs_cond_uc.repeat(args.num_samples, 1)
                        else:
                            obs_cond_uc = obs_cond_uc.repeat(args.num_samples, 1, 1)

                        t3 = sync_time()

                        episode_step = img_idx - sim_start_idx
                        
                        t4 = sync_time()

                        naction_gc = torch.randn(
                            (
                                args.num_samples,
                                model_params["len_traj_pred"],
                                2,
                            ),
                            device=device,
                        )

                        noise_scheduler.set_timesteps(num_diffusion_iters)

                        for k in noise_scheduler.timesteps:
                            noise_pred = model(
                                "noise_pred_net",
                                sample=naction_gc,
                                timestep=k,
                                global_cond=obs_cond_gc,
                            )

                            naction_gc = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction_gc,
                            ).prev_sample

                        t5 = sync_time()

                        vision_time_ms = (t1 - t0) * 1000
                        pose_time_ms = (t2 - t1) * 1000
                        select_time_ms = (t3 - t2) * 1000
                        diffusion_time_ms = (t5 - t4) * 1000

                        total_infer_time_ms = (
                            vision_time_ms
                            + pose_time_ms
                            + select_time_ms
                            + diffusion_time_ms
                        )
                        print(
                            f"[Timing] "
                            f"vision={vision_time_ms:.1f} ms | "
                            f"pose={pose_time_ms:.1f} ms | "
                            f"select={select_time_ms:.1f} ms | "
                            f"diffusion={diffusion_time_ms:.1f} ms | "
                            f"total={total_infer_time_ms:.1f} ms"
                        )
                        print(f"[Pose] raw=({pose_raw[0]:.4f}, {pose_raw[1]:.4f}, {pose_raw[2]:.4f})")

                        naction_uc = torch.randn(
                            (
                                args.num_samples,
                                model_params["len_traj_pred"],
                                2,
                            ),
                            device=device,
                        )


                        noise_scheduler.set_timesteps(num_diffusion_iters)

                        for k in noise_scheduler.timesteps:

                            noise_pred = model(
                                "noise_pred_net",
                                sample=naction_uc,
                                timestep=k,
                                global_cond=obs_cond_uc,
                            )

                            naction_uc = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction_uc,
                            ).prev_sample

                    naction_gc = to_numpy(get_action(naction_gc))
                    print("naction_gc.shape =", naction_gc.shape)

                    # normalized action [-1, 1] -> real velocity [v, w]
                    pred_actions = unnormalize_action(naction_gc)   # (num_samples, 60, 2)
                    
                    mean_pred_vel_reced= pred_actions.mean(axis=0)[:receding_horizon]  # (4, 2)

                    mean_pred_traj_reced = reconstruct_pose_rk4(
                        linear_vels=mean_pred_vel_reced[:, 0],
                        angular_vels=mean_pred_vel_reced[:, 1],
                        dt=0.0333,
                    )

                    pred_trajs = []

                    for i in range(args.num_samples):
                        pred_vel = pred_actions[i]

                        pred_traj = reconstruct_pose_rk4(
                            linear_vels=pred_vel[:, 0],
                            angular_vels=pred_vel[:, 1],
                            dt=0.0333,
                        )

                        pred_trajs.append(pred_traj)

                    naction_uc = to_numpy(get_action(naction_uc))
                    print("naction_uc.shape =", naction_uc.shape)
                    pred_actions_uc = unnormalize_action(naction_uc)
                    
                    mean_pred_vel_uc_reced = pred_actions_uc.mean(axis=0)[:receding_horizon]
                    mean_pred_traj_uc_reced = reconstruct_pose_rk4(
                        linear_vels=mean_pred_vel_uc_reced[:, 0],
                        angular_vels=mean_pred_vel_uc_reced[:, 1],
                        dt=0.0333,
                    )

                    pred_trajs_uc = []

                    for i in range(args.num_samples):
                        pred_vel_uc = pred_actions_uc[i]

                        pred_traj_uc = reconstruct_pose_rk4(
                            pred_vel_uc[:, 0],
                            pred_vel_uc[:, 1],
                            dt=0.0333,
                        )

                        pred_trajs_uc.append(pred_traj_uc)

                    # GT velocity 가져오기
                    pred_len = model_params["len_traj_pred"]
                    gt_end = min(img_idx + pred_len, end_idx)

                    gt_vel = f[action_key][img_idx:gt_end].astype(np.float32)

                    # episode 끝 근처에서 길이 부족하면 zero padding
                    if len(gt_vel) < pred_len:
                        pad = np.zeros((pred_len - len(gt_vel), 2), dtype=np.float32)
                        gt_vel = np.concatenate([gt_vel, pad], axis=0)

                    gt_traj = reconstruct_pose_rk4(
                        linear_vels=gt_vel[:, 0],
                        angular_vels=gt_vel[:, 1],
                        dt=0.0333,
                        initial_pose=(0.0, 0.0, 0.0),
                    )

                    fig = plt.figure(figsize=(12, 8))

                    gs = fig.add_gridspec(
                        3,
                        3,
                        width_ratios=[2.4, 1, 1],
                        height_ratios=[1, 1, 1],
                    )

                    ax_curr_gt   = fig.add_subplot(gs[0, 0])
                    ax_curr_pred = fig.add_subplot(gs[1, 0])

                    ax_goal = fig.add_subplot(gs[0:2, 1:3])

                    ax_reced_traj = fig.add_subplot(gs[2, 0])
                    ax_reced_v = fig.add_subplot(gs[2, 1])
                    ax_reced_w = fig.add_subplot(gs[2, 2])

                    ax_curr_pred.imshow(curr_img)
                    ax_curr_pred.set_title(f"GC Pred + GT\nstep={episode_step}\ntotal={total_infer_time_ms:.1f} ms")

                    ax_curr_gt.imshow(curr_img)
                    ax_curr_gt.set_title(f"UC Pred + GT\nstep={episode_step}")

                    ax_goal.imshow(goal_img)
                    ax_goal.set_title(
                        "Episode Final Goal Image\n"
                        f"Pred pose=({pose_raw[0]:.2f}, {pose_raw[1]:.2f}, {pose_raw[2]:.2f})"
                    )
                    ax_goal.axis("off")

                    fig.suptitle(
                        f"Episode={episode_num} | "
                        f"Pred pose=({pose_raw[0]:.2f}, {pose_raw[1]:.2f}, {pose_raw[2]:.2f}) | "
                        f"GT pose=({gt_pose_raw[0]:.2f}, {gt_pose_raw[1]:.2f}, {gt_pose_raw[2]:.2f})",
                        fontsize=12,
                    )

                    # axes[0] 현재 이미지 위에 pred / gt trajectory overlay
                    H, W = np.array(curr_img).shape[:2]
                    cx, cy = W // 2, H - 30   # 로봇이 이미지 하단 중앙에 있다고 가정
                    scale = 500             # meter -> pixel, 필요하면 조절

                    for pred_traj in pred_trajs:

                        pred_px_x = cx - pred_traj[:, 1] * scale
                        pred_px_y = cy - pred_traj[:, 0] * scale

                        ax_curr_pred.plot(
                            pred_px_x,
                            pred_px_y,
                            linewidth=1,
                            linestyle="--",
                            color="green",
                            alpha=0.5,
                        )

                        ax_curr_pred.scatter(
                            pred_px_x[-1],
                            pred_px_y[-1],
                            s=20,
                            marker="x",
                            color="green",
                            alpha=0.5,
                        )

                    for pred_traj in pred_trajs_uc:

                        pred_px_x = cx - pred_traj[:, 1] * scale
                        pred_px_y = cy - pred_traj[:, 0] * scale

                        ax_curr_gt.plot(
                            pred_px_x,
                            pred_px_y,
                            linewidth=1,
                            linestyle="--",
                            color="red",
                            alpha=0.5,
                        )

                        ax_curr_gt.scatter(
                            pred_px_x[-1],
                            pred_px_y[-1],
                            s=20,
                            marker="x",
                            color="red",
                            alpha=0.5,
                        )

                    gt_px_x = cx - gt_traj[:, 1] * scale
                    gt_px_y = cy - gt_traj[:, 0] * scale

                    
                    ax_curr_pred.plot(gt_px_x, gt_px_y, linewidth=2, color="blue" ,label="GT traj")
                    ax_curr_pred.scatter(gt_px_x[-1], gt_px_y[-1], s=30, marker="x", color="blue",label="GT end")
                    
                    ax_curr_pred.scatter(cx,cy, s=20, color="green",label="Start")
               
                    ax_curr_gt.plot(gt_px_x,gt_px_y,linewidth=2,color="blue",label="GT traj")
                    ax_curr_gt.scatter(gt_px_x[-1],gt_px_y[-1],s=30,marker="x",color="blue",label="GT end")
                    
                    ax_curr_gt.scatter(cx,cy,s=20,color="green",label="Start")

                    mean_uc_px_x = cx - mean_pred_traj_uc_reced[:, 1] * scale
                    mean_uc_px_y = cy - mean_pred_traj_uc_reced[:, 0] * scale

                    ax_curr_gt.plot(
                        mean_uc_px_x,
                        mean_uc_px_y,
                        linewidth=3,
                        color="red",
                        label=f"Mean UC traj {receding_horizon}-step",
                    )

                    ax_curr_gt.scatter(
                        mean_uc_px_x[-1],
                        mean_uc_px_y[-1],
                        s=40,
                        marker="x",
                        color="red",
                        label=f"Mean UC {receding_horizon}-step end",
                    )

                    ax_curr_gt.legend(fontsize=8)
                    ax_curr_gt.axis("off")

                    mean_px_x = cx - mean_pred_traj_reced[:, 1] * scale
                    mean_px_y = cy - mean_pred_traj_reced[:, 0] * scale

                    ax_curr_pred.plot(
                        mean_px_x,
                        mean_px_y,
                        linewidth=3,
                        color="green",
                        label=f"Mean GC traj {receding_horizon}-step",
                    )

                    ax_curr_pred.scatter(
                        mean_px_x[-1],
                        mean_px_y[-1],
                        s=40,
                        marker="x",
                        color="green",
                        label=f"Mean GC {receding_horizon}-step end",
                    )
                         
                    ax_curr_pred.legend(fontsize=8)
                    ax_curr_pred.axis("off")

                    # =========================
                    # 3rd row: receding horizon 4-step action analysis
                    # =========================
                    reced_steps = np.arange(receding_horizon)
                    reced_time = reced_steps * 0.0333

                    gt_vel_reced = gt_vel[:receding_horizon]  

                    # 이동거리 계산
                    dx = np.diff(mean_pred_traj_reced[:, 0])
                    dy = np.diff(mean_pred_traj_reced[:, 1])
                    step_dists = np.sqrt(dx**2 + dy**2)
                    total_dist = step_dists.sum()

                    final_x = mean_pred_traj_reced[-1, 0]
                    final_y = mean_pred_traj_reced[-1, 1]
                    final_disp = np.sqrt(final_x**2 + final_y**2)

                    # 1열: 4-step 적분 trajectory
                    ax_reced_traj.plot(
                        -mean_pred_traj_reced[:, 1],
                        mean_pred_traj_reced[:, 0],
                        linewidth=1,
                        color = "green",
                        #marker="o",
                        label=f"Mean GC {receding_horizon}-step integrated traj",
                    )

                    ax_reced_traj.plot(
                        -mean_pred_traj_uc_reced[:, 1],
                        mean_pred_traj_uc_reced[:, 0],
                        linewidth=1,
                        color="red",
                        label=f"Mean UC {receding_horizon}-step integrated traj",
                    )

                    ax_reced_traj.scatter(
                        -mean_pred_traj_reced[0, 1],
                        mean_pred_traj_reced[0, 0],
                        s=30,
                        color = "green",
                        marker="o",
                        label="Start",
                    )

                    ax_reced_traj.scatter(
                        -mean_pred_traj_reced[-1, 1],
                        mean_pred_traj_reced[-1, 0],
                        s=30,
                        color = "green",
                        marker="x",
                        label="GC End",
                    )

                    ax_reced_traj.scatter(
                        -mean_pred_traj_uc_reced[-1, 1],
                        mean_pred_traj_uc_reced[-1, 0],
                        s=30,
                        color="red",
                        marker="x",
                        label="UC End",
                    )

                    ax_reced_traj.set_title(
                        f"Mean Receding Traj ({receding_horizon} steps)\n"
                        f"path={total_dist:.4f} m, disp={final_disp:.4f} m"
                    )
                    ax_reced_traj.set_xlabel("y [m]")
                    ax_reced_traj.set_ylabel("x [m]")
                    ax_reced_traj.axis("equal")
                    ax_reced_traj.grid(True)
                    ax_reced_traj.legend(fontsize=8)

                    # 2열: 선속 4개
                    ax_reced_v.plot(
                        reced_steps,
                        mean_pred_vel_reced[:, 0],
                        color = "green",
                        #marker="o",
                        linewidth=1,
                        label="GC v",
                    )

                    ax_reced_v.plot(
                        reced_steps,
                        mean_pred_vel_uc_reced[:, 0],
                        color="red",
                        linewidth=1,
                        label="UC v",
                    )

                    ax_reced_v.plot(
                        reced_steps,
                        gt_vel_reced[:, 0],
                        color="blue",
                        #marker="x",
                        linewidth=1,
                        label="GT v",
                    )

                    ax_reced_v.set_title("Mean linear velocity")
                    ax_reced_v.set_xlabel("step")
                    ax_reced_v.set_ylabel("v [m/s]")
                    ax_reced_v.grid(True)
                    ax_reced_v.legend(fontsize=8)

                    # 3열: 각속 4개
                    ax_reced_w.plot(
                        reced_steps,
                        mean_pred_vel_reced[:, 1],
                        color = "green",
                        #marker="o",
                        linewidth=1,
                        label="GC w",
                    )

                    ax_reced_w.plot(
                        reced_steps,
                        mean_pred_vel_uc_reced[:, 1],
                        color="red",
                        linewidth=1,
                        label="UC w",
                    )

                    ax_reced_w.plot(
                        reced_steps,
                        gt_vel_reced[:, 1],
                        color="blue",
                        #marker="x",
                        linewidth=1,
                        label="GT w",
                    )
                    ax_reced_w.set_title("Mean angular velocity")
                    ax_reced_w.set_xlabel("step")
                    ax_reced_w.set_ylabel("w [rad/s]")
                    ax_reced_w.grid(True)
                    ax_reced_w.legend(fontsize=8)

                    plt.tight_layout()
                    save_path = os.path.join(output_dir, f"step_{episode_step:06d}.png")
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"[Saved] {save_path}")
                    continue


                    """
                    # ROS publish 대신 결과 저장
                    sampled_actions = naction.copy()

                    naction = naction[0]
                    chosen_waypoint = naction[args.waypoint]
                    """
                else:
                    raise NotImplementedError("Only nomad simulation path is handled here.")

            reached_goal = img_idx >= end_idx - 1
            
            if reached_goal:
                print("Reached goal! Stopping simulation...")
                break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", default="nomad", type=str)
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)

"""
python navigation_simulation.py \
  --num-samples 8
"""

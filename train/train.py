import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DPMSolverMultistepScheduler

from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, NoMaD_pose, DenseNetwork, PoseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.vint_dataset_episode import ViNT_H5_Action_Dataset

from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)

import h5py


def dock_crop_condition(traj):
    x = traj[:, 0]
    y = traj[:, 1]

    return (
        (x >= -0.3) & (x <= 0.3) &
        (y >= -0.5) & (y <= 0.1)
    )


def reconstruct_pose_rk4_for_stats(
    linear_vels,
    angular_vels,
    dt=0.0333,
    initial_pose=(0.0, 0.0, 0.0),
):
    n_steps = len(linear_vels)
    trajectory = np.zeros((n_steps + 1, 3), dtype=np.float32)
    trajectory[0] = np.array(initial_pose, dtype=np.float32)

    def f(q, v, w):
        return np.array(
            [
                v * np.cos(q[2]),
                v * np.sin(q[2]),
                w,
            ],
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

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def align_final_pose_to_origin_y_for_stats(traj):
    final_x, final_y, final_theta = traj[-1]

    rot_angle = -np.pi / 2 - final_theta

    c = np.cos(rot_angle)
    s = np.sin(rot_angle)

    R = np.array([
        [c, -s],
        [s,  c],
    ], dtype=np.float32)

    xy = traj[:, :2] - np.array([final_x, final_y], dtype=np.float32)
    aligned_xy = xy @ R.T

    aligned_theta = wrap_angle(traj[:, 2] + rot_angle)

    aligned_traj = np.zeros_like(traj)
    aligned_traj[:, :2] = aligned_xy
    aligned_traj[:, 2] = aligned_theta

    return aligned_traj


def get_valid_action_start_by_condition_for_stats(
    actions,
    crop_condition_fn,
    dt=0.0333,
):
    if crop_condition_fn is None:
        return 0

    traj = reconstruct_pose_rk4_for_stats(
        linear_vels=actions[:, 0],
        angular_vels=actions[:, 1],
        dt=dt,
        initial_pose=(0.0, 0.0, 0.0),
    )

    traj = align_final_pose_to_origin_y_for_stats(traj)

    valid = crop_condition_fn(traj)
    valid = np.asarray(valid, dtype=bool)

    if len(valid) != len(traj):
        raise ValueError(
            f"crop_condition_fn must return bool array with len(traj). "
            f"got {len(valid)}, expected {len(traj)}"
        )

    outside = ~valid
    outside_idx = np.where(outside)[0]

    if len(outside_idx) == 0:
        return 0

    if len(outside_idx) == len(traj):
        return None

    crop_local_start = int(outside_idx[-1]) + 1

    if crop_local_start >= len(actions):
        return None

    return crop_local_start


def compute_global_h5_action_stats(
    h5_datasets,
    action_key_default="encoder",
    percent_99=True,
    crop_condition_fn=None,
    crop_dt=0.0333,
):
    all_actions = []

    for h5_cfg in h5_datasets:
        h5_path = h5_cfg["h5_path"]
        action_key = h5_cfg.get("action_key", action_key_default)

        with h5py.File(h5_path, "r") as h5:
            actions = h5[action_key][:].astype(np.float32)
            episode_ends = h5["episode_ends"][:]

        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        selected_actions = []

        for ep_idx, (ep_start, ep_end) in enumerate(
            zip(episode_starts, episode_ends)
        ):
            ep_start = int(ep_start)
            ep_end = int(ep_end)

            ep_actions = actions[ep_start:ep_end]

            crop_local_start = get_valid_action_start_by_condition_for_stats(
                ep_actions,
                crop_condition_fn=crop_condition_fn,
                dt=crop_dt,
            )

            if crop_local_start is None:
                continue

            ep_actions = ep_actions[crop_local_start:]

            if len(ep_actions) > 0:
                selected_actions.append(ep_actions)

        if len(selected_actions) == 0:
            print("[Global stats load]")
            print("  h5_path:", h5_path)
            print("  action_key:", action_key)
            print("  selected actions: 0")
            continue

        selected_actions = np.concatenate(selected_actions, axis=0)
        all_actions.append(selected_actions)

        print("[Global stats load]")
        print("  h5_path:", h5_path)
        print("  action_key:", action_key)
        print("  original actions:", actions.shape)
        print("  selected actions:", selected_actions.shape)

    if len(all_actions) == 0:
        raise RuntimeError(
            "No actions selected for global stats. "
            "Check crop_condition_fn or dataset paths."
        )

    all_actions = np.concatenate(all_actions, axis=0)

    if percent_99:
        action_min = np.percentile(all_actions, 1, axis=0)
        action_max = np.percentile(all_actions, 99, axis=0)
    else:
        action_min = np.min(all_actions, axis=0)
        action_max = np.max(all_actions, axis=0)

    action_scale = action_max - action_min
    action_scale = np.maximum(action_scale, 1e-6)

    action_stats = {
        "min": action_min.astype(np.float32),
        "max": action_max.astype(np.float32),
        "scale": action_scale.astype(np.float32),
    }

    print("[Global action normalization]")
    print("total selected actions:", all_actions.shape)
    print("min:", action_stats["min"])
    print("max:", action_stats["max"])
    print("scale:", action_stats["scale"])

    return action_stats

def compute_global_h5_pose_stats(
    h5_datasets,
    action_key_default="encoder",
    percent_99=True,
    crop_condition_fn=None,
    crop_dt=0.0333,
):
    all_pose = []

    for h5_cfg in h5_datasets:
        h5_path = h5_cfg["h5_path"]
        action_key = h5_cfg.get("action_key", action_key_default)

        with h5py.File(h5_path, "r") as h5:
            actions = h5[action_key][:].astype(np.float32)
            episode_ends = h5["episode_ends"][:]

        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        selected_pose = []

        for ep_start, ep_end in zip(episode_starts, episode_ends):
            ep_start = int(ep_start)
            ep_end = int(ep_end)

            ep_actions = actions[ep_start:ep_end]

            traj = reconstruct_pose_rk4_for_stats(
                linear_vels=ep_actions[:, 0],
                angular_vels=ep_actions[:, 1],
                dt=crop_dt,
                initial_pose=(0.0, 0.0, 0.0),
            )

            traj = align_final_pose_to_origin_y_for_stats(traj)

            if crop_condition_fn is not None:
                valid = crop_condition_fn(traj)
                valid = np.asarray(valid, dtype=bool)

                outside_idx = np.where(~valid)[0]

                if len(outside_idx) == len(traj):
                    continue

                crop_start = 0 if len(outside_idx) == 0 else int(outside_idx[-1]) + 1

                if crop_start >= len(traj):
                    continue

                traj = traj[crop_start:]

            pose = np.stack(
                [traj[:, 0],
                 traj[:, 1], 
                 traj[:, 2]],
                axis=1,
            ).astype(np.float32)

            if len(pose) > 0:
                selected_pose.append(pose)

        if len(selected_pose) == 0:
            print("[Global pose stats load]")
            print("  h5_path:", h5_path)
            print("  selected pose: 0")
            continue

        selected_pose = np.concatenate(selected_pose, axis=0)
        all_pose.append(selected_pose)

        print("[Global pose stats load]")
        print("  h5_path:", h5_path)
        print("  selected pose:", selected_pose.shape)

    if len(all_pose) == 0:
        raise RuntimeError(
            "No pose selected for global pose stats. "
            "Check crop_condition_fn or dataset paths."
        )

    all_pose = np.concatenate(all_pose, axis=0)

    if percent_99:
        pose_min = np.percentile(all_pose, 1, axis=0)
        pose_max = np.percentile(all_pose, 99, axis=0)
    else:
        pose_min = np.min(all_pose, axis=0)
        pose_max = np.max(all_pose, axis=0)

    pose_scale = pose_max - pose_min
    pose_scale = np.maximum(pose_scale, 1e-6)

    pose_stats = {
        "min": pose_min.astype(np.float32),
        "max": pose_max.astype(np.float32),
        "scale": pose_scale.astype(np.float32),
    }

    print("[Global pose normalization]")
    print("total selected pose:", all_pose.shape)
    print("min [x, theta]:", pose_stats["min"])
    print("max [x, theta]:", pose_stats["max"])
    print("scale [x, theta]:", pose_stats["scale"])

    return pose_stats


def visualize_h5_crop_preview(
    h5_datasets,
    save_path,
    action_key_default="encoder",
    crop_condition_fn=None,
    crop_dt=0.0333,
):
    import matplotlib.pyplot as plt

    #os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["red", "blue", "green", "orange", "purple"]

    for h5_i, h5_cfg in enumerate(h5_datasets):
        h5_path = h5_cfg["h5_path"]
        action_key = h5_cfg.get("action_key", action_key_default)

        with h5py.File(h5_path, "r") as h5:
            actions = h5[action_key][:].astype(np.float32)
            episode_ends = h5["episode_ends"][:]

        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        color = colors[h5_i % len(colors)]

        for ep_idx, (ep_start, ep_end) in enumerate(zip(episode_starts, episode_ends)):
            ep_start = int(ep_start)
            ep_end = int(ep_end)

            ep_actions = actions[ep_start:ep_end]

            traj = reconstruct_pose_rk4_for_stats(
                linear_vels=ep_actions[:, 0],
                angular_vels=ep_actions[:, 1],
                dt=crop_dt,
                initial_pose=(0.0, 0.0, 0.0),
            )

            traj = align_final_pose_to_origin_y_for_stats(traj)

            if crop_condition_fn is not None:
                valid = crop_condition_fn(traj)
                valid = np.asarray(valid, dtype=bool)

                outside_idx = np.where(~valid)[0]

                if len(outside_idx) == 0:
                    crop_start = 0
                elif len(outside_idx) == len(traj):
                    continue
                else:
                    crop_start = int(outside_idx[-1]) + 1

                if crop_start >= len(traj):
                    continue

                plot_traj = traj[crop_start:]
                start_xy = plot_traj[0, :2]
            else:
                plot_traj = traj
                start_xy = plot_traj[0, :2]

            ax.plot(
                plot_traj[:, 0],
                plot_traj[:, 1],
                color=color,
                alpha=0.35,
                linewidth=1,
            )

            ax.scatter(
                start_xy[0],
                start_xy[1],
                color=color,
                s=12,
                alpha=0.8,
            )

    # crop box
    ax.axvline(-0.3, linestyle="--", linewidth=1)
    ax.axvline(0.3, linestyle="--", linewidth=1)
    ax.axhline(-0.5, linestyle="--", linewidth=1)
    ax.axhline(0.1, linestyle="--", linewidth=1)

    ax.scatter(0, 0, marker="x", color="black", s=60, label="aligned final pose")
    ax.set_title("Aligned crop preview used for training")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()
    #fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print("[Crop preview saved]", save_path)

def main(config):
    # assert : 조건이 거짓이면 raise error
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        # PyTorch(CUDA)가 GPU 번호(cuda:0, cuda:1 …)를 어떤 기준으로 매길지 정하는 환경변수
        # GPU를 PCI 버스 번호 순서대로 정렬해서 GPU ID를 부여
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"]) # PyTorch 랜덤 고정
        cudnn.deterministic = True # GPU 연산(CuDNN)을 deterministic 하게 강제

    # CuDNN이 현재 입력 크기에 가장 빠른 연산 알고리즘을 자동 탐색해서 사용하도록 허용
    # 입력 크기가 자주 변하지 않으면 성능 향상에 좋다
    cudnn.benchmark = True  # good if input sizes don't vary
    
    # ImageNet dataset normalization statistics
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:

        data_config = config["datasets"][dataset_name]

        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True

        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1

        if "end_slack" not in data_config:
            data_config["end_slack"] = 0

        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        dataset_format = data_config.get("format", "vint") #get() 은 딕셔너리(dictionary)에서 값을 꺼내는 함수
        # dict.get(key, default) key가 있으면 그 값을 반환, 없으면 default 반환
        
        """ ===================================================== """
        # =========================================================
        # H5 DATASET
        # =========================================================

   
        if dataset_format == "h5":

            h5_datasets = data_config.get("h5_datasets", None)

            if h5_datasets is None:
                h5_datasets = [
                    {
                        "h5_path": data_config["h5_path"],
                        "image_keys": data_config.get("image_keys", ["image_bottom"]),
                        "action_key": data_config.get("action_key", "encoder"),
                        "encoder_key": data_config.get("encoder_key", None),
                        "imu_key": data_config.get("imu_key", None),
                        "lidar_key": data_config.get("lidar_key", None),
                    }
                ]

            if isinstance(h5_datasets, dict):
                h5_datasets = [h5_datasets]

            train_action_stats = compute_global_h5_action_stats(
                h5_datasets=h5_datasets,
                action_key_default=data_config.get("action_key", "encoder"),
                percent_99=config.get("action_stats_percent_99", True),
                crop_condition_fn=dock_crop_condition,
                crop_dt=0.0333,
            )
            train_pose_stats = compute_global_h5_pose_stats(
                h5_datasets=h5_datasets,
                action_key_default=data_config.get("action_key", "encoder"),
                percent_99=config.get("pose_stats_percent_99", True),
                crop_condition_fn=dock_crop_condition,
                crop_dt=0.0333,
            )

            visualize_h5_crop_preview(
                h5_datasets=h5_datasets,
                save_path=os.path.join(
                    config["project_folder"],
                    "crop_preview",
                    f"{dataset_name}_aligned_crop_preview.png",
                ),
                action_key_default=data_config.get("action_key", "encoder"),
                crop_condition_fn=dock_crop_condition,
                crop_dt=0.0333,
            )

            # =========================
            # Train h5: all episodes
            # =========================
            for h5_i, h5_cfg in enumerate(h5_datasets):

                dataset = ViNT_H5_Action_Dataset(
                    h5_path=h5_cfg["h5_path"],
                    split="train",
                    test_episode_num=0,
                    seed=config.get("seed", 42),
                    dataset_index=h5_i,
                    image_keys=h5_cfg.get("image_keys", [h5_cfg.get("image_key", "image_bottom")]),
                    action_key=h5_cfg.get("action_key", "encoder"),
                    encoder_key=h5_cfg.get("encoder_key", None),
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    context_size=config["context_size"],
                    context_spacing=data_config["context_spacing"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    normalize=config["normalize"],
                    action_stats=train_action_stats,
                    pose_stats=train_pose_stats,
                    predict_velocity=config["predict_velocity"],
                    use_global_goal_for_test=False,
                    imu_key=h5_cfg.get("imu_key", None),
                    lidar_key=h5_cfg.get("lidar_key", None),
                    return_sensor_history=config.get("return_sensor_history", False),
                    crop_condition_fn=dock_crop_condition,
                    encoder_imu_context_size=config["encoder_imu_context_size"],
                    encoder_imu_context_spacing=config["encoder_imu_context_spacing"],
                    lidar_context_size=config["lidar_context_size"],
                    lidar_context_spacing=config["lidar_context_spacing"],
                    chunk_size=config.get("chunk_size", 30)
                )

                train_dataset.append(dataset)

            # =========================
            # Test h5: fixed episode
            # =========================
            dataset = ViNT_H5_Action_Dataset(
                h5_path=data_config["test_h5_path"],
                split="test",
                test_episode_num=data_config["test_episode_num"],
                seed=config.get("seed", 42),
                dataset_index=0,
                image_keys=data_config.get(
                    "test_image_keys",
                    [data_config.get("test_image_key", data_config.get("image_key", "image_bottom"))],
                ),
                action_key=data_config.get(
                    "test_action_key",
                    data_config.get("action_key", "encoder"),
                ),
                encoder_key=data_config.get(
                    "test_encoder_key", None),
                image_size=config["image_size"],
                waypoint_spacing=data_config["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=data_config["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                context_spacing=data_config["context_spacing"],
                context_type=config["context_type"],
                end_slack=data_config["end_slack"],
                normalize=config["normalize"],
                action_stats=train_action_stats,
                pose_stats=train_pose_stats,
                predict_velocity=config["predict_velocity"],
                use_global_goal_for_test=data_config["use_global_goal_for_test"],
                imu_key=data_config.get("test_imu_key", None),
                lidar_key=data_config.get("test_lidar_key", None),
                return_sensor_history=config.get("return_sensor_history", False),
                crop_condition_fn=dock_crop_condition,
                encoder_imu_context_size=config["encoder_imu_context_size"],
                encoder_imu_context_spacing=config["encoder_imu_context_spacing"],
                lidar_context_size=config["lidar_context_size"],
                lidar_context_spacing=config["lidar_context_spacing"],
                chunk_size=config.get("chunk_size", 30)
            )

            dataset_type = f"{dataset_name}_test"
            test_dataloaders[dataset_type] = dataset


        # =========================================================
        # ORIGINAL ViNT DATASET
        # =========================================================
        else:

            for data_split_type in ["train", "test"]:

                if data_split_type in data_config:

                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        context_spacing=data_config["context_spacing"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        test_dataloaders[dataset_type] = dataset
                        
    """ ===================================================== """

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    print("num train datasets:", len(train_dataset.datasets))

    for i, d in enumerate(train_dataset.datasets):
        print(
            f"[dataset {i}]",
            "len =", len(d),
            "h5_path =", getattr(d, "h5_path", None),
            "num episodes =", len(getattr(d, "episode_ends", [])),
        )

    print("concat len:", len(train_dataset))
    print("batch_size:", config["batch_size"])
    print("expected batches:", int(np.ceil(len(train_dataset) / config["batch_size"])))


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False, # 마지막 batch 크기가 부족해도 버리지 말고 사용
        persistent_workers=False, # True : epoch 끝나도 DataLoader worker 프로세스를 종료하지 말고 계속 유지
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    # =============================================================

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            predict_velocity=config.get("predict_velocity", True),
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                sensor_context_sizes={
                    "encoder": config["encoder_imu_context_size"],
                    "imu": config["encoder_imu_context_size"],
                    "lidar": config["lidar_context_size"],
                },
                use_encoder=config["use_encoder"],
                use_imu=config["use_imu"],
                use_lidar=config["use_lidar"],
                num_image_keys=len(config.get("data_image_keys", ["image_bottom"])),
            )


            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            

        """
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        """
        noise_pred_net = TransformerForDiffusion(
            input_dim=2,
            output_dim=2,
            horizon=config["len_traj_pred"],
            cond_dim=config["encoding_size"],
            n_obs_steps=1,
            n_layer=12,
            n_head=6,
            n_emb=384,
        )

        pose_pred_network = PoseNetwork(embedding_dim=config["encoding_size"])
        # dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD_pose(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            pose_pred_net=pose_pred_network,
        )
        """
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
        """
        
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        """
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
        )
        """

    else:
        raise ValueError(f"Model {config['model']} not supported")

    # =============================================================


    # Gradient Clipping:
    # 역전파(backpropagation)에서 계산된 gradient가 너무 커지는 것(exploding gradients)을 막기 위해 gradient 크기를 제한하는 기법
    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        
        for p in model.parameters():
            if not p.requires_grad:
                continue
            
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")


    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        
        # Cyclic LR → 미리 정해진 규칙으로 LR을 올렸다 내렸다 반복
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )

        # Plateau → 성능이 멈추면 LR 줄임
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    ema_checkpoint_path = None

    # 이전 학습 이어하기 (Resume) 
    if "load_run" in config:
        # 로그 폴더 찾기
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        ema_checkpoint_path = os.path.join(load_project_folder, "ema_latest.pth")
         # 숫자 이름의 pth 파일만 찾기
        pth_files = [
            f for f in os.listdir(load_project_folder)
            if re.match(r"^\d+\.pth$", f)
        ]

        if pth_files:
            max_epoch = max(
                int(os.path.splitext(f)[0])
                for f in pth_files
            )

            current_epoch = max_epoch + 1
            print(f"Resume from epoch {current_epoch}")


    if "test_load_model" in config:
        load_project_folder = os.path.join("logs", config["test_load_model"])
        print("Loading test model from ", load_project_folder)

        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path)

        load_model(model, config["model_type"], latest_checkpoint)
        ema_checkpoint_path = os.path.join(load_project_folder, "ema_latest.pth")

        pth_files = [
            f for f in os.listdir(load_project_folder)
            if re.match(r"^\d+\.pth$", f)
        ]

        if pth_files:
            max_epoch = max(
                int(os.path.splitext(f)[0])
                for f in pth_files
            )
            current_epoch = max_epoch
            print(f"Test model trained epoch {current_epoch}")


    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        # batch -> GPU0, GPU1, ...
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if config["model_type"] == "nomad":
            optimizer_path = os.path.join(load_project_folder, "optimizer_latest.pth")
            scheduler_path = os.path.join(load_project_folder, "scheduler_latest.pth")

            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                print("Loaded optimizer from ", optimizer_path)
            else:
                print("Optimizer checkpoint not found: ", optimizer_path)

            if scheduler is not None:
                if os.path.exists(scheduler_path):
                    scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
                    print("Loaded scheduler from ", scheduler_path)
                else:
                    print("Scheduler checkpoint not found: ", scheduler_path)
        else:
            # optimizer 복원
            if "optimizer" in latest_checkpoint:
                optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
            # scheduler 복원
            if scheduler is not None and "scheduler" in latest_checkpoint:
                scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())


    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )

    else:
        # NoMaD
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            predict_velocity=config["predict_velocity"],
            ACTION_STATS=train_action_stats,
            POSE_STATS = train_pose_stats,
            max_distance=config["distance"]["max_dist_cat"],
            ema_checkpoint_path=ema_checkpoint_path,
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":

    # PyTorch 멀티프로세싱(worker)를 새 프로세스를 생성(spawn)하는 방식으로 시작
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    config_save_path = os.path.join(
        config["project_folder"],
        "config.yaml"
    )

    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="orothy579-postech", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)

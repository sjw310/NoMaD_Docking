import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE # (160, 120)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import psutil
import gc

"""
# LOAD DATA CONFIG
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
# POPULATE ACTION STATS
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])
"""
    
# Train utils for ViNT and GNM

def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
    max_dist = 20.0
):
    """
    Compute losses for distance and action prediction.
    """

    if not hasattr(_compute_losses, "debug_count"):
        _compute_losses.debug_count = 0

    debug_now = (_compute_losses.debug_count % 100 == 0) # False


    dist_pred_norm = dist_pred.squeeze(-1)
    dist_pred_scaled = dist_pred_norm * max_dist
    dist_label_norm = dist_label.float() / max_dist

    dist_loss = F.mse_loss(dist_pred_norm, dist_label_norm)

    if debug_now:
        print("\n===== DIST DEBUG =====")
        print("dist_pred shape:", dist_pred.shape)
        print("dist_label shape:", dist_label.shape)

        print("pred_scaled[:8]:", np.round(dist_pred_scaled[:8].detach().cpu().numpy(), 3))
        print("pred_norm[:8]:", np.round(dist_pred_norm[:8].detach().cpu().numpy(), 3))
        print("label[:8]:", dist_label[:8].detach().cpu().numpy())
        print("label_norm[:8]:", np.round(dist_label_norm[:8].detach().cpu().numpy(), 3))

        print("scaled_mse[:8]:",np.round((dist_pred_scaled[:8] - dist_label[:8].float()).pow(2).detach().cpu().numpy(),3,),)
        print("norm_mse[:8]:",np.round((dist_pred_norm[:8] - dist_label_norm[:8]).pow(2).detach().cpu().numpy(),6,),)

        print(
            "pred_scaled mean/min/max:",
            dist_pred_scaled.mean().item(),
            dist_pred_scaled.min().item(),
            dist_pred_scaled.max().item(),
        )

        print(
            "label mean/min/max:",
            dist_label.float().mean().item(),
            dist_label.float().min().item(),
            dist_label.float().max().item(),
        )

        print("dist_loss:", dist_loss.item())
        print("======================\n")

    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)

        assert unreduced_loss.shape == action_mask.shape, (
            f"{unreduced_loss.shape} != {action_mask.shape}"
        )

        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    assert action_pred.shape == action_label.shape, (
        f"{action_pred.shape} != {action_label.shape}"
    )

    per_elem_action_mse = F.mse_loss(
        action_pred,
        action_label,
        reduction="none",
    )

    action_loss = action_reduce(per_elem_action_mse)

    if debug_now:
        print("\n===== ACTION DEBUG =====")
        print("action_pred shape:", action_pred.shape)
        print("action_label shape:", action_label.shape)
        print("action_mask shape:", action_mask.shape)

        print("mask[:8]:", action_mask[:8].detach().cpu().numpy())
        print("mask mean:", action_mask.float().mean().item())

        print("pred[0]:")
        print(np.round(action_pred[0].detach().cpu().numpy(), 4))

        print("label[0]:")
        print(np.round(action_label[0].detach().cpu().numpy(), 4))

        print(
            "pred mean/min/max:",
            action_pred.mean().item(),
            action_pred.min().item(),
            action_pred.max().item(),
        )

        print(
            "label mean/min/max:",
            action_label.mean().item(),
            action_label.min().item(),
            action_label.max().item(),
        )

        print("per_elem_mse[0]:")
        print(np.round(per_elem_action_mse[0].detach().cpu().numpy(), 6))

        per_sample_mse = per_elem_action_mse
        while per_sample_mse.dim() > 1:
            per_sample_mse = per_sample_mse.mean(dim=-1)
            
        print(
            "per_sample_mse[:8]:",
            np.round(per_sample_mse[:8].detach().cpu().numpy(), 6),
        )

        print(
            "masked_per_sample_mse[:8]:",
            np.round(
                (per_sample_mse[:8] * action_mask[:8])
                .detach()
                .cpu()
                .numpy(),
                6,
            ),
        )

        print("action_loss:", action_loss.item())
        print("========================\n")

    action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(
            action_pred[:, :, :2],
            action_label[:, :, :2],
            dim=-1,
            eps=1e-8,
        )
    )

    multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
            eps=1e-8,
        )
    )

    results = {
        "dist_loss": dist_loss,
        "action_loss": action_loss,
        # "action_waypts_cos_sim": action_waypts_cos_similairity,
        # "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,
    }

    if learn_angle:
        action_orien_cos_sim = action_reduce(
            F.cosine_similarity(
                action_pred[:, :, 2:],
                action_label[:, :, 2:],
                dim=-1,
                eps=1e-8,
            )
        )

        multi_action_orien_cos_sim = action_reduce(
            F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
                eps=1e-8,
            )
        )

        results["action_orien_cos_sim"] = action_orien_cos_sim
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim

    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    _compute_losses.debug_count += 1

    return results


def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
        use_tqdm: whether to use tqdm
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )

    # Unpacking
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        
        model_outputs = model(obs_image, goal_image)

        dist_label = dist_label.to(device)
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()
      
        dist_pred, action_pred = model_outputs

        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,

):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
        eval_fraction (float): fraction of data to use for evaluation
        use_tqdm (bool): whether to use tqdm for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            # VISUALIZATION_IMAGE_SIZE (160, 120)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            
            model_outputs = model(obs_image, goal_image)

            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            dist_pred, action_pred = model_outputs

            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # Log data to wandb/console, with visualizations selected from the last batch
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    #batch_dist_label: torch.Tensor,
    batch_pose_align_target: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    predict_velocity: bool = True,
    ACTION_STATS = None,
    POSE_STATS=None,
    max_distance = 400,
    encoder_hist=None,
    imu_hist=None,
    lidar_hist=None,
):
    """
    Compute losses for distance and action prediction.
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
        predict_velocity=predict_velocity,
        encoder_hist=encoder_hist,
        imu_hist=imu_hist,
        lidar_hist=lidar_hist,
    )

    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    #gc_distance = model_output_dict['gc_distance']
    gc_pose = model_output_dict["gc_pose"]

    debug_now =True

    if debug_now:
        b = 0  # batch 안에서 볼 샘플 index
        
        np.set_printoptions(suppress=True, precision=5)

        print("\n===== NOMAD ACTION DEBUG =====")
        print("uc_actions shape:", uc_actions.shape)
        print("gc_actions shape:", gc_actions.shape)
        print("label shape:", batch_action_label.shape)
        print("action_mask:", action_mask[b].item())
        #print("distance label:", batch_dist_label[b].item())

        print("pred pose norm:", gc_pose[b].detach().cpu().numpy())
        print("gt pose norm:", batch_pose_align_target[b].detach().cpu().numpy())
        
        gc_pose_raw = unnormalize_pose(gc_pose, POSE_STATS)
        target_pose_raw = unnormalize_pose(batch_pose_align_target, POSE_STATS)

        def unnormalize_action_tensor(action_tensor):
            if ACTION_STATS is None:
                return action_tensor
            scale = torch.as_tensor(
                ACTION_STATS["scale"],
                device=action_tensor.device,
                dtype=action_tensor.dtype,
            )
            min_val = torch.as_tensor(
                ACTION_STATS["min"],
                device=action_tensor.device,
                dtype=action_tensor.dtype,
            )
            return ((action_tensor + 1.0) / 2.0) * scale + min_val

        uc_actions_raw = unnormalize_action_tensor(uc_actions)
        gc_actions_raw = unnormalize_action_tensor(gc_actions)
        batch_action_label_raw = unnormalize_action_tensor(batch_action_label)

        print("\n[UC pred action raw]")
        print(np.round(uc_actions_raw[b][:5].detach().cpu().numpy(), 5))

        print("\n[GC pred action raw]")
        print(np.round(gc_actions_raw[b][:5].detach().cpu().numpy(), 5))

        print("\n[GT action label raw]")
        print(np.round(batch_action_label_raw[b][:5].detach().cpu().numpy(), 5))

        print("\n[GC - GT raw]")
        print(np.round((gc_actions_raw[b][:5] - batch_action_label_raw[b][:5]).detach().cpu().numpy(), 5))

        print("pred pose raw [x, y, theta]:", gc_pose_raw[b].detach().cpu().numpy())
        print("gt pose raw [x, y, theta]:", target_pose_raw[b].detach().cpu().numpy())
        print("pose raw error:", (gc_pose_raw[b] - target_pose_raw[b]).detach().cpu().numpy())
 

        """
        print("\n===== NOMAD DIST DEBUG =====")
        print("gc_distance shape:", gc_distance.shape)
        print("batch_dist_label shape:", batch_dist_label.shape)
        """
        print("\n===== NOMAD POSE DEBUG =====")
        print("gc_pose shape:", gc_pose.shape)
        print("batch_pose_align_target shape:", batch_pose_align_target.shape)

        #gc_distance_scaled = gc_distance * max_distance

        """
        print("gc_distance norm:", gc_distance[b].detach().cpu().numpy())
        print("gc_distance scaled:", gc_distance_scaled[b].detach().cpu().numpy())
        print("dist_label:", batch_dist_label[b].detach().cpu().numpy())

        print("gc_distance[:8]:")
        print(np.round(gc_distance[:8].detach().cpu().numpy().squeeze(), 5))
        print("gc_distance_scaled[:8]:")
        print(np.round(gc_distance_scaled[:8].detach().cpu().numpy().squeeze(), 5))
        print("dist_label[:8]:")
        print(batch_dist_label[:8].detach().cpu().numpy())

        print("dist error[:8]:")
        print(np.round((gc_distance_scaled[:8].squeeze(-1) - batch_dist_label[:8]).detach().cpu().numpy(),5))
        """
        print("==============================\n")

    """
    # loss 계산 시 distance label 0~1 정규화
    dist_target = batch_dist_label.float().unsqueeze(-1) / max_distance
    gc_dist_loss = F.mse_loss(gc_distance,dist_target)
    """

    batch_pose_align_target = batch_pose_align_target.float().to(device)
    gc_pose_loss = F.mse_loss(gc_pose,batch_pose_align_target)


    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        #"gc_dist_loss": gc_dist_loss,
        "gc_pose_loss": gc_pose_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 0.1,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    predict_velocity: bool = True,
    ACTION_STATS =None,
    POSE_STATS=None,
    max_distance = 20
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        ema_model: exponential moving average model
        optimizer: optimizer to use
        dataloader: dataloader for training
        transform: transform to use
        device: device to use
        noise_scheduler: noise scheduler to train with 
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    
    model.train()
    num_batches = len(dataloader)

    train_loggers = {
        "total_loss": Logger("total_loss", "train", window_size=print_log_freq),
        "pose_loss": Logger("pose_loss", "train", window_size=print_log_freq),
        "diffusion_loss": Logger("diffusion_loss", "train", window_size=print_log_freq),
    }

    ema_window_size = 10
    uc_action_loss_logger = Logger("uc_action_loss", "train_ema", window_size=ema_window_size)
    # uc_action_waypts_cos_sim_logger = Logger(
    #     "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    # )
    # uc_multi_action_waypts_cos_sim_logger = Logger(
    #     "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    # )
    
    #gc_dist_loss_logger = Logger("gc_dist_loss", "train_ema", window_size=print_log_freq)
    gc_pose_loss_logger = Logger("gc_pose_loss", "train_ema", window_size=ema_window_size)
    gc_action_loss_logger = Logger("gc_action_loss", "train_ema", window_size=ema_window_size)
    # gc_action_waypts_cos_sim_logger = Logger(
    #     "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    # )
    # gc_multi_action_waypts_cos_sim_logger = Logger(
    #     "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    # )
    ema_loggers = {
        "uc_action_loss": uc_action_loss_logger,
        #"uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger, # for waypoint prediction
        #"uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger, # for waypoint prediction
        #"gc_dist_loss": gc_dist_loss_logger,
        "gc_pose_loss": gc_pose_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        #"gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger, # for waypoint prediction
        #"gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger, # for waypoint prediction
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
                ep_idx,
                curr_time,
                pose_align_target,
                sensor_dict,
            ) = data

            """
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(ep_idx, dtype=torch.int64),
            torch.as_tensor(curr_time, dtype=torch.int64),
            """        
        
            obs_channels = goal_image.shape[1]  # image key 개수 * 3
            obs_images = torch.split(obs_image, obs_channels, dim=1)
            
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            
            action_mask = action_mask.to(device)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            encoder_hist = sensor_dict.get("encoder_hist", None)
            imu_hist = sensor_dict.get("imu_hist", None)
            lidar_hist = sensor_dict.get("lidar_hist", None)

            if encoder_hist is not None:
                encoder_hist = encoder_hist.to(device)
            if imu_hist is not None:
                imu_hist = imu_hist.to(device)
            if lidar_hist is not None:
                lidar_hist = lidar_hist.to(device)

            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
                encoder_hist=encoder_hist,
                imu_hist=imu_hist,
                lidar_hist=lidar_hist,
            )

            pose_align_target = pose_align_target.float().to(device)

            # Get distance label
            distance = distance.float().to(device)
            # normalize
            distance_norm = distance / max_distance
            
            # 이미 Normalized
            ndiffusion_target = get_diffusion_target(actions, predict_velocity)
            
            #ndiffusion_target = normalize_data(diffusion_target, ACTION_STATS)
            
            if isinstance(ndiffusion_target, torch.Tensor):
                naction = ndiffusion_target.float().to(device)
            else:
                naction = from_numpy(ndiffusion_target).to(device)

            assert naction.shape[-1] == 2, "action dim must be 2"

            """
            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance_norm)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())
            """
            pose_pred = model("pose_pred_net", obsgoal_cond=obsgoal_cond)
            pose_loss = F.mse_loss(pose_pred,pose_align_target,reduction="none",).mean(dim=-1)
            pose_loss = (pose_loss * (1 - goal_mask.float())).mean() / (1e-2 + (1 - goal_mask.float()).mean())

            # Sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2 loss
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # Total loss
            #loss = alpha * dist_loss + (1-alpha) * diffusion_loss
            loss = alpha * pose_loss + (1-alpha) * diffusion_loss

            grad_accum_steps = 4

            if i % grad_accum_steps == 0:
                optimizer.zero_grad()

            (loss / grad_accum_steps).backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == num_batches:
                optimizer.step()
                ema_model.step(model)
            
            """
            # Total loss
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Exponential Moving Average of the model weights
            ema_model.step(model)
            """

            # Logging
            loss_cpu = loss.item()
            pose_loss_cpu = pose_loss.item()
            diffusion_loss_cpu = diffusion_loss.item()

            train_loggers["total_loss"].log_data(loss_cpu)
            train_loggers["pose_loss"].log_data(pose_loss_cpu)
            train_loggers["diffusion_loss"].log_data(diffusion_loss_cpu)

            tepoch.set_postfix(loss=loss_cpu)

            ema_data_log = {}

            if print_log_freq != 0 and i % print_log_freq == 0:
                for logger in train_loggers.values():
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                with torch.inference_mode():
                    losses = _compute_losses_nomad(
                                ema_model.averaged_model,
                                noise_scheduler,
                                batch_obs_images,
                                batch_goal_images,
                                #distance.to(device),
                                pose_align_target.to(device),
                                actions.to(device),
                                device,
                                action_mask.to(device),
                                predict_velocity=predict_velocity,
                                ACTION_STATS=ACTION_STATS,
                                POSE_STATS=POSE_STATS,
                                max_distance=max_distance,
                                encoder_hist=encoder_hist,
                                imu_hist=imu_hist,
                                lidar_hist=lidar_hist,
                            )
                
                for key, value in losses.items():
                    if key in ema_loggers:
                        logger = ema_loggers[key]
                        logger.log_data(value.item())
	            
                for key, logger in ema_loggers.items():
                    ema_data_log[logger.full_name()] = logger.latest()
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

            if use_wandb and wandb_log_freq != 0 and i % wandb_log_freq == 0:
                data_log = {}
                for logger in train_loggers.values():
                    data_log[logger.full_name()] = logger.latest()
                    data_log[f"{logger.full_name()} moving_avg"] = logger.moving_average()
                data_log.update(ema_data_log)
                wandb.log(data_log, commit=True)
            

            if image_log_freq != 0 and i % image_log_freq == 0:
                
                with torch.inference_mode():
                    records = visualize_diffusion_action_distribution(
                        ema_model.averaged_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        batch_viz_obs_images,
                        batch_viz_goal_images,
                        actions,
                        distance,
                        goal_pos,
                        device,
                        "train",
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_wandb,
                        predict_velocity,
                        ACTION_STATS,
                        POSE_STATS,
                        global_step=i,
                        ep_idx=ep_idx,
                        curr_time=curr_time,
                        episode_starts=dataloader.dataset.datasets[0].episode_starts, # TODO: 임시방편, concat 때문
                        action_mask=action_mask.cpu(),
                        dataset=dataloader.dataset.datasets[0], # TODO: 임시방편, concat 때문
                        max_distance=max_distance,
                        encoder_hist=encoder_hist,
                        imu_hist=imu_hist,
                        lidar_hist=lidar_hist,
                    )

            # RAM monitor
            # rss : 실제로 현재 점유 중인 RAM
            if i % 100 == 0:
                process = psutil.Process(os.getpid())

                print(
                    f"[RAM] "
                    f"RSS={process.memory_info().rss / 1024**3:.2f} GB | "
                    f"VMS={process.memory_info().vms / 1024**3:.2f} GB"
                )

def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 1.0,
    use_wandb: bool = True,
    predict_velocity = True,
    ACTION_STATS =None,
    POSE_STATS=None,
    max_distance= 400
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    
    ema_model = ema_model.averaged_model
    ema_model.eval()
    
    num_batches = len(dataloader)

    # logger
    eval_window_size = 10
    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=eval_window_size)
    # uc_action_waypts_cos_sim_logger = Logger(
    #     "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    # )
    # uc_multi_action_waypts_cos_sim_logger = Logger(
    #     "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    # )
    #gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_pose_loss_logger = Logger("gc_pose_loss", eval_type, window_size=eval_window_size)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=eval_window_size)
    # gc_action_waypts_cos_sim_logger = Logger(
    #     "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    # )
    # gc_multi_action_waypts_cos_sim_logger = Logger(
    #     "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    # )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        #"uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        #"uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        #"gc_dist_loss": gc_dist_loss_logger,
        "gc_pose_loss": gc_pose_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        #"gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        #"gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)


    traj_error_records = []
    
    # 추론(inference)만 수행, 학습에 필요한 계산 기록을 수행하지 않음
    with torch.inference_mode():
        with tqdm.tqdm(
            itertools.islice(dataloader, num_batches), 
            total=num_batches, 
            dynamic_ncols=True, 
            desc=f"Evaluating {eval_type}", leave=False) as tepoch:
            
            for i, data in enumerate(tepoch):
                (
                    obs_image, 
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    dataset_idx,
                    action_mask, 
                    ep_idx,
                    curr_time,
                    pose_align_target,
                    sensor_dict,
                ) = data
                
                """
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                actions_torch,
                torch.as_tensor(distance, dtype=torch.int64),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(self.dataset_index, dtype=torch.int64),
                torch.as_tensor(action_mask, dtype=torch.float32),
                torch.as_tensor(ep_idx, dtype=torch.int64),
                torch.as_tensor(curr_time, dtype=torch.int64),   
                """
                encoder_hist = sensor_dict.get("encoder_hist", None)
                imu_hist = sensor_dict.get("imu_hist", None)
                lidar_hist = sensor_dict.get("lidar_hist", None)

                if encoder_hist is not None:
                    encoder_hist = encoder_hist.to(device)
                if imu_hist is not None:
                    imu_hist = imu_hist.to(device)
                if lidar_hist is not None:
                    lidar_hist = lidar_hist.to(device)


                obs_channels = goal_image.shape[1]  # image key 개수 * 3
                obs_images = torch.split(obs_image, obs_channels, dim=1)
                batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
                batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
                batch_goal_images = transform(goal_image).to(device)
                action_mask = action_mask.to(device)

                B = actions.shape[0]

                # Generate random goal mask
                rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
                goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
                no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

                rand_mask_cond = ema_model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    goal_img=batch_goal_images,
                    input_goal_mask=rand_goal_mask,
                    encoder_hist=encoder_hist,
                    imu_hist=imu_hist,
                    lidar_hist=lidar_hist,
                )

                obsgoal_cond = ema_model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    goal_img=batch_goal_images,
                    input_goal_mask=no_mask,
                    encoder_hist=encoder_hist,
                    imu_hist=imu_hist,
                    lidar_hist=lidar_hist,
                )

                goal_mask_cond = ema_model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    goal_img=batch_goal_images,
                    input_goal_mask=goal_mask,
                    encoder_hist=encoder_hist,
                    imu_hist=imu_hist,
                    lidar_hist=lidar_hist,
                )

                distance = distance.to(device)
            
                ndiffusion_target = get_diffusion_target(actions, predict_velocity)
                #ndiffusion_target = normalize_data(diffusion_target, ACTION_STATS)
                
                if isinstance(ndiffusion_target, torch.Tensor):
                    naction = ndiffusion_target.float().to(device)
                else:
                    naction = from_numpy(ndiffusion_target).to(device)
                
                assert naction.shape[-1] == 2, "action dim must be 2"

                # Sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # Sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                ### RANDOM MASK ERROR ###
                # Predict the noise residual
                rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
                
                # L2 loss
                rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
                
                ### NO MASK ERROR ###
                # Predict the noise residual
                no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
                
                # L2 loss
                no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

                ### GOAL MASK ERROR ###
                # predict the noise residual
                goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
                
                # L2 loss
                goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
                
                # Logging
                loss_cpu = rand_mask_loss.item()
                tepoch.set_postfix(loss=loss_cpu)
                
                if use_wandb:
                    wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
                    wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
                    wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})

                
                if i % print_log_freq == 0 and print_log_freq != 0:
                    losses = _compute_losses_nomad(
                                ema_model,
                                noise_scheduler,
                                batch_obs_images,
                                batch_goal_images,
                                #distance.to(device),
                                pose_align_target.to(device),
                                actions.to(device),
                                device,
                                action_mask.to(device),
                                predict_velocity=predict_velocity,
                                ACTION_STATS=ACTION_STATS,
                                POSE_STATS=POSE_STATS,
                                max_distance=max_distance,
                                encoder_hist=encoder_hist,
                                imu_hist=imu_hist,
                                lidar_hist=lidar_hist,
                            )
                    
                    for key, value in losses.items():
                        if key in loggers:
                            logger = loggers[key]
                            logger.log_data(value.item())
                
                    data_log = {}
                    for key, logger in loggers.items():
                        data_log[logger.full_name()] = logger.latest()
                        data_log[f"{logger.full_name()} moving_avg"] = logger.moving_average()
                        if i % print_log_freq == 0 and print_log_freq != 0:
                            print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                        wandb.log(data_log, commit=True)

                if image_log_freq != 0 and i % image_log_freq == 0:
                    records = visualize_diffusion_action_distribution(
                        ema_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        batch_viz_obs_images,
                        batch_viz_goal_images,
                        actions,
                        distance,
                        goal_pos,
                        device,
                        eval_type,
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_wandb,
                        predict_velocity,
                        ACTION_STATS,
                        POSE_STATS,
                        global_step=i,
                        ep_idx=ep_idx,
                        curr_time=curr_time,
                        episode_starts=dataloader.dataset.episode_starts, # TODO: 임시방편
                        action_mask=action_mask.cpu(),
                        dataset=dataloader.dataset, # TODO: 임시방편
                        max_distance=max_distance, # TODO: 임시방편
                        encoder_hist=encoder_hist,
                        imu_hist=imu_hist,
                        lidar_hist=lidar_hist,
                    )
                    traj_error_records.extend(records)
    
    if len(traj_error_records) > 0:
        x = np.array([
            r["time_sec"]
            for r in traj_error_records
        ])

        uc_dist = np.array([r["uc_dist"] for r in traj_error_records])
        gc_dist = np.array([r["gc_dist"] for r in traj_error_records])
        uc_ang = np.array([r["uc_ang"] for r in traj_error_records])
        gc_ang = np.array([r["gc_ang"] for r in traj_error_records])

        save_dir = os.path.join(
            project_folder,
            f"visualize",
            f"epoch{epoch}"
        )

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].scatter(x,uc_dist,s=10,label=f"UC (mean={uc_dist.mean():.3f} m)",)
        ax[0].scatter(x,gc_dist,s=10,label=f"GC (mean={gc_dist.mean():.3f} m)",)

        ax[0].axhline(uc_dist.mean(),linestyle="--",alpha=0.7,)
        ax[0].axhline(gc_dist.mean(),linestyle="--",color="orange",alpha=0.7,)

        ax[0].set_title(f"Final position error | episode {traj_error_records[0]['episode']}")
        ax[0].set_xlabel("time [s]")
        ax[0].set_ylabel("distance error [m]")
        ax[0].grid(True)
        ax[0].legend(loc="upper right")

        ax[1].scatter(x,uc_ang,s=10,label=f"UC (mean={uc_ang.mean():.2f}°)",)
        ax[1].scatter(x,gc_ang,s=10,label=f"GC (mean={gc_ang.mean():.2f}°)",)

        ax[1].axhline(uc_ang.mean(),linestyle="--",alpha=0.7,)
        ax[1].axhline(gc_ang.mean(),linestyle="--",alpha=0.7,color="orange")

        ax[1].set_title(f"Final heading error | episode_{traj_error_records[0]['episode']}")
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("heading error [deg]")
        ax[1].grid(True)
        ax[1].legend(loc="upper right")

        fig.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"episode_{traj_error_records[0]['episode']}_traj_error_summary.png",
        )

        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        if use_wandb:
            wandb.log({
                f"{eval_type}_traj_error_summary": wandb.Image(save_path)
            }, commit=False)

"""
# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata
"""

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def unnormalize_pose(norm_pose, POSE_STATS):
    if POSE_STATS is None:
        return norm_pose

    if isinstance(norm_pose, torch.Tensor):
        scale = torch.tensor(
            POSE_STATS["scale"],
            device=norm_pose.device,
            dtype=norm_pose.dtype,
        )
        min_val = torch.tensor(
            POSE_STATS["min"],
            device=norm_pose.device,
            dtype=norm_pose.dtype,
        )
    else:
        scale = np.asarray(POSE_STATS["scale"], dtype=np.float32)
        min_val = np.asarray(POSE_STATS["min"], dtype=np.float32)

    return (norm_pose + 1.0) / 2.0 * scale + min_val


def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_diffusion_target(actions, predict_velocity: bool = False):
    
    if predict_velocity:
        return actions
    else:
        return get_delta(actions)

def get_action(diffusion_output,ACTION_STATS=None,predict_velocity: bool = True):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device

    nout = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    nout = to_numpy(nout)

    if predict_velocity:
        actions = nout
    else:
        out = unnormalize_data(
            nout,
            ACTION_STATS,
        )
        actions = np.cumsum(out, axis=1)

    return from_numpy(actions).to(device)

def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    predict_velocity: bool = True,
    encoder_hist=None,
    imu_hist=None,
    lidar_hist=None,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
    "vision_encoder",
    obs_img=batch_obs_images,
    goal_img=batch_goal_images,
    input_goal_mask=goal_mask,
    encoder_hist=encoder_hist,
    imu_hist=imu_hist,
    lidar_hist=lidar_hist,
)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
        encoder_hist=encoder_hist,
        imu_hist=imu_hist,
        lidar_hist=lidar_hist,
    )
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output


    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output,predict_velocity=predict_velocity)

    # initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    
    gc_actions = get_action(diffusion_output,predict_velocity=predict_velocity)
    
    #gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    gc_pose = model("pose_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        #'gc_distance': gc_distance,
        'gc_pose': gc_pose,
    }

# RK4
def reconstruct_pose_rk4(linear_vels, angular_vels, dt=0.0333, initial_pose=(0.0, 0.0, 0.0)):
    n_steps = len(linear_vels)
    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = initial_pose

    def f(q, v, w):
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

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
    predict_velocity= True,
    ACTION_STATS = None,
    POSE_STATS=None,
    global_step=None,
    ep_idx=None,
    curr_time=None,
    episode_starts=None,
    action_mask=None,
    dataset=None,
    max_distance = 400,
    visualize_detail = False,
    encoder_hist=None,
    imu_hist=None,
    lidar_hist=None,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        f"epoch{epoch}",
        "prediction_detail",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    if encoder_hist is not None:
        encoder_hist = encoder_hist[:num_images_log]

    if imu_hist is not None:
        imu_hist = imu_hist[:num_images_log]

    if lidar_hist is not None:
        lidar_hist = lidar_hist[:num_images_log]
    
    wandb_list = []
    traj_error_records = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    # torch.split(tensor, chunk_size, dim=0)
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
            predict_velocity=predict_velocity,
            encoder_hist=encoder_hist,
            imu_hist=imu_hist,
            lidar_hist=lidar_hist,
        )

        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist)*max_distance for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    def unnormalize_action(data, ACTION_STATS):
        if isinstance(data, torch.Tensor):

            scale = torch.tensor(
                ACTION_STATS["scale"],
                device=data.device,
                dtype=data.dtype,
            )
            min_val = torch.tensor(
                ACTION_STATS["min"],
                device=data.device,
                dtype=data.dtype,
            )
        else:
            scale = np.asarray(
                ACTION_STATS["scale"],
                dtype=np.float32,
            )
            min_val = np.asarray(
                ACTION_STATS["min"],
                dtype=np.float32,
            )
        return (
            (data + 1.0)
            / 2.0
            * scale
            + min_val
        )

    # predict_velocity의 경우 적분하여 경로 생성
    def to_traj_from_velocity(action_seq, dt=0.0333):


        traj = reconstruct_pose_rk4(
            linear_vels=action_seq[:, 0],
            angular_vels=action_seq[:, 1],
            dt=dt,
        )
        return traj[:, :2]  # (T+1, 2)


    for i in range(num_images_log):
        
        # ep_idx, curr_time는 batch tensor
        sample_ep_idx = int(ep_idx[i])
        sample_curr_time = int(curr_time[i])
        ep_start = int(episode_starts[sample_ep_idx])
        local_time = sample_curr_time - ep_start

        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])
        
        # unnormalize
        uc_actions = unnormalize_action(uc_actions, ACTION_STATS)
        gc_actions = unnormalize_action(gc_actions, ACTION_STATS)
        action_label = unnormalize_action(action_label, ACTION_STATS)

        uc_actions_raw = uc_actions.copy()
        gc_actions_raw = gc_actions.copy()
        action_label_raw = action_label.copy()

        # total prediction length
        total_len = action_label.shape[0]

        sample_action_mask = bool(action_mask[i].item())

        if predict_velocity:
            uc_actions = np.stack([
                to_traj_from_velocity(a, dt=0.0333)
                for a in uc_actions
            ], axis=0)

            gc_actions = np.stack([
                to_traj_from_velocity(a, dt=0.0333)
                for a in gc_actions
            ], axis=0)

            action_label = to_traj_from_velocity(action_label, dt=0.0333)

            # Ground Truth distance
            gt_dist = int(np_distance_labels[i])

            if dataset is not None:
                ep_end = int(dataset.episode_ends[sample_ep_idx])

                long_end = min(
                    sample_curr_time + gt_dist,
                    ep_end,
                )

                action_label_long_raw = dataset.actions[
                    sample_curr_time:long_end
                ].astype(np.float32)

                action_label_long = to_traj_from_velocity(
                    action_label_long_raw,
                    dt=0.0333,
                )
            else:
                action_label_long_raw = action_label_raw
                action_label_long = action_label

            gt_max_len = min(gt_dist + 1, action_label.shape[0])

            # truncate separately
            uc_actions = uc_actions[:, :gt_max_len]
            gc_actions = gc_actions[:, :gt_max_len]
            action_label = action_label[:gt_max_len]
            
        else:
            uc_actions = uc_actions
            gc_actions = gc_actions
            action_label = action_label

        if visualize_detail:

            fig, ax = plt.subplots(1, 3)
            goal_pos_vis = action_label_long[-1]

            # action_label[None] : Add a batch dimension at the front
            traj_list = (
                list(uc_actions)
                + list(gc_actions)
                + [action_label_long]
            )

            # traj_labels = ["r", "GC", "GC_mean", "GT"]
            traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
            traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

            # make points numpy array of robot positions (0, 0) and goal positions
            point_list = [np.array([0, 0]), goal_pos_vis]

            point_colors = ["blue", "orange"]
            point_alphas = [1.0, 1.0]
            
            plot_trajs_and_points(
                ax[0],
                traj_list,
                point_list,
                traj_colors,
                point_colors,
                traj_labels=None,
                point_labels=None,
                quiver_freq=0,
                traj_alphas=traj_alphas,
                point_alphas=point_alphas, 
            )
            
            obs_image = to_numpy(batch_viz_obs_images[i])
            goal_image = to_numpy(batch_viz_goal_images[i])

            # move channel to last dimension
            obs_image = np.moveaxis(obs_image, 0, -1)
            goal_image = np.moveaxis(goal_image, 0, -1)
            ax[1].imshow(obs_image)
            ax[2].imshow(goal_image)

            # set title
            ax[0].set_title(
                f"diffusion action predictions\n"
                f"epi={sample_ep_idx}  "
                f"local={local_time}  "
                f"({local_time*0.0333:.2f}s)  "
                f"{'TRAIN' if sample_action_mask else 'SKIP'}",
            )

            ax[1].set_title(f"observation")

            dist_label = int(np_distance_labels[i])
            neg_text = " [NEGATIVE]" if dist_label == total_len else ""

            ax[2].set_title(
                f"goal: label={dist_label} ({dist_label*0.0333:.2f}s) {neg_text} "
                f"gc_dist={gc_distances_avg[i]:.2f} ({gc_distances_avg[i]*0.0333:.2f}s)"
            )

            # make the plot large
            fig.set_size_inches(18.5, 10.5)

            if global_step is None:
                save_name = f"sample_{i}.png"
            else:
                save_name = f"step{global_step}_sample_{i}.png"

            save_path = os.path.join(visualize_path, save_name)
            plt.savefig(save_path)

            if use_wandb:
                wandb_list.append(wandb.Image(save_path))
            
            plt.close(fig)
            del fig
            gc.collect()

        # =================================================================================

        def add_final_distance_heading_arrow(ax, gt_traj, pred_traj, ang_err, offset=0.03):

            gt_end = gt_traj[-1, :2]
            pred_end = pred_traj[-1, :2]

            vec = pred_end - gt_end
            dist = np.linalg.norm(vec)

            if dist < 1e-8:
                return

            # 거리선이 trajectory와 겹치지 않도록 수직 방향으로 살짝 이동
            perp = np.array([-vec[1], vec[0]]) / dist

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            plot_scale = max(
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
            )

            offset_dist = offset * plot_scale

            p1 = gt_end + offset_dist * perp
            p2 = pred_end + offset_dist * perp

            # 양방향 화살표
            ax.annotate("",xy=p2,xytext=p1,arrowprops=dict(arrowstyle="<->",color="black",lw=1.3,alpha=0.75,),)

            # 끝점에서 거리선까지 보조 점선
            ax.plot([gt_end[0], p1[0]],[gt_end[1], p1[1]],linestyle=":",color="black",alpha=0.45,linewidth=1.0,)

            ax.plot([pred_end[0], p2[0]],[pred_end[1], p2[1]],linestyle=":",color="black",alpha=0.45,linewidth=1.0,)

            # 거리 텍스트
            mid = (p1 + p2) / 2

            ax.text(mid[0],mid[1],f"{dist:.3f} m\nΔθ={abs(ang_err):.1f}°",ha="center",va="bottom",fontsize=10,bbox=dict(facecolor="white",edgecolor="none",alpha=0.85,),)

        def set_axis_from_reference(ax, gt_traj, mean_traj, margin=0.1):

            ref = np.concatenate([
                gt_traj[:, :2],
                mean_traj[:, :2],
            ], axis=0)

            xmin, xmax = ref[:, 0].min(), ref[:, 0].max()
            ymin, ymax = ref[:, 1].min(), ref[:, 1].max()

            size = max(xmax - xmin, ymax - ymin)

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2

            half = size / 2 * (1 + margin)

            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)

        def add_velocity_arrows(ax, gt, mean):
            # --------------------------
            # 1. GT max-min
            # --------------------------
            idx_max = np.argmax(gt)
            idx_min = np.argmin(gt)

            x_range = idx_max

            gt_range = gt[idx_max] - gt[idx_min]

            ax.annotate(
                "",
                xy=(x_range, gt[idx_max]),
                xytext=(x_range, gt[idx_min]),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="blue",
                    lw=1.5,
                ),
            )

            ax.text(
                x_range,
                (gt[idx_max] + gt[idx_min]) / 2,
                f"range\n{gt_range:.4f}",
                color="blue",
                ha="left",
                va="center",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # --------------------------
            # 2. max |mean-GT|
            # --------------------------
            diff = np.abs(mean - gt)
            idx_diff = np.argmax(diff)

            y_gt = gt[idx_diff]
            y_mean = mean[idx_diff]

            ax.annotate(
                "",
                xy=(idx_diff, y_gt),
                xytext=(idx_diff, y_mean),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="red",
                    lw=1.5,
                ),
            )

            ax.text(
                idx_diff,
                (y_gt + y_mean) / 2,
                f"err\n{diff[idx_diff]:.4f}",
                color="red",
                ha="right",
                va="center",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # GT는 plot에서 끝까지 표시하되,
        # error/velocity 비교는 pred horizon 안에서만 수행
        # raw가 원래는 velocity, 아닌 경우 trajectory

        pred_action_len = action_label_raw.shape[0]

        # velocity
        eval_action_len = min(gt_dist, pred_action_len)
        # trajectory
        eval_traj_len = eval_action_len + 1

        # velocity eval
        uc_actions_raw_eval = uc_actions_raw[:, :eval_action_len]
        gc_actions_raw_eval = gc_actions_raw[:, :eval_action_len]
        action_label_raw_eval = action_label_raw[:eval_action_len]

        # trajectory eval
        uc_actions_eval = uc_actions[:, :eval_traj_len]
        gc_actions_eval = gc_actions[:, :eval_traj_len]
        action_label_eval = action_label[:eval_traj_len]

        uc_mean_linear  = uc_actions_raw_eval[:, :, 0].mean(0)
        gc_mean_linear  = gc_actions_raw_eval[:, :, 0].mean(0)

        uc_mean_angular = uc_actions_raw_eval[:, :, 1].mean(0)
        gc_mean_angular = gc_actions_raw_eval[:, :, 1].mean(0)

        uc_traj_mean = uc_actions_eval.mean(0)
        gc_traj_mean = gc_actions_eval.mean(0)

        # velocity plot은 GT long 전체를 그림
        action_label_raw_plot = action_label_long_raw

        # pred는 존재하는 만큼만 그림
        uc_actions_raw_plot = uc_actions_raw
        gc_actions_raw_plot = gc_actions_raw

        def compute_final_dist_heading_error_from_velocity(
            gt_traj,
            pred_traj,
            gt_action_raw,
            pred_action_raw,
            dt=0.0333,
        ):
            # position error는 trajectory 최종 위치 기준
            gt_end = gt_traj[-1, :2]
            pred_end = pred_traj[-1, :2]
            dist_err = np.linalg.norm(pred_end - gt_end)

            # heading error는 angular velocity 적분 기준
            gt_theta = np.sum(gt_action_raw[:, 1]) * dt
            pred_theta = np.sum(pred_action_raw[:, 1]) * dt

            dtheta = np.arctan2(
                np.sin(pred_theta - gt_theta),
                np.cos(pred_theta - gt_theta),
            )

            ang_err_deg = abs(np.degrees(dtheta))

            return dist_err, ang_err_deg

        uc_mean_raw = np.stack(
            [uc_mean_linear, uc_mean_angular],
            axis=1,
        )

        gc_mean_raw = np.stack(
            [gc_mean_linear, gc_mean_angular],
            axis=1,
        )

        uc_dist_err, uc_ang_err = compute_final_dist_heading_error_from_velocity(
            action_label_eval,
            uc_traj_mean,
            action_label_raw_eval,
            uc_mean_raw,
            dt=0.0333,
        )

        gc_dist_err, gc_ang_err = compute_final_dist_heading_error_from_velocity(
            action_label_eval,
            gc_traj_mean,
            action_label_raw_eval,
            gc_mean_raw,
            dt=0.0333,
        )

        traj_error_records.append({
            "episode": sample_ep_idx,
            "local_time": local_time,
            "time_sec": local_time * 0.0333,
            "uc_dist": uc_dist_err,
            "uc_ang": abs(uc_ang_err),
            "gc_dist": gc_dist_err,
            "gc_ang": abs(gc_ang_err),
        })

        if not visualize_detail:
            continue

        # ==================================================
        # 추가 상세 plot 저장: 6개 subplot
        # ==================================================
        detail_fig, detail_ax = plt.subplots(2, 3, figsize=(20, 11))
        detail_ax = detail_ax.flatten()

        detail_fig.suptitle(
            f"epi={sample_ep_idx} | "
            f"local={local_time} ({local_time*0.0333:.2f}s)| "
            f"{'TRAIN' if sample_action_mask else 'SKIP'}",
            fontsize=14
        )
        
        # 1) GT trajectory + all UC trajectories
        plot_trajs_and_points(
            detail_ax[0],
            [action_label_long] + list(uc_actions),
            point_list,
            ["magenta"] + ["red"] * len(uc_actions),
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=[1.0] + [0.15] * len(uc_actions),
            point_alphas=point_alphas,
        )
        detail_ax[0].plot(
            uc_traj_mean[:, 0],
            uc_traj_mean[:, 1],
            color="red",
            linewidth=3,
            label="UC mean",
        )

        set_axis_from_reference(
            detail_ax[0],
            action_label_long,
            uc_traj_mean,
        )
   
        add_final_distance_heading_arrow(
            detail_ax[0],
            action_label_eval,
            uc_traj_mean,
            uc_ang_err,
            offset=0.03,
        )

        detail_ax[0].set_title("GT + all UC traj")


        # 2) GT trajectory + all GC trajectories
        plot_trajs_and_points(
            detail_ax[1],
            [action_label_long] + list(gc_actions),
            point_list,
            ["magenta"] + ["green"] * len(gc_actions),
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=[1.0] + [0.15] * len(gc_actions),
            point_alphas=point_alphas,
        )

        detail_ax[1].plot(
            gc_traj_mean[:, 0],
            gc_traj_mean[:, 1],
            color="green",
            linewidth=3,
            label="GC mean",
        )

        set_axis_from_reference(
            detail_ax[1],
            action_label_long,
            gc_traj_mean,
        )

        add_final_distance_heading_arrow(
            detail_ax[1],
            action_label_eval,
            gc_traj_mean,
            gc_ang_err,
            offset=0.03,
        )
        
        detail_ax[1].set_title("GT + all GC traj")

        t_gt = np.arange(action_label_raw_plot.shape[0])
        t_pred = np.arange(uc_actions_raw_plot.shape[1])

        # 3) GT linear velocity + all UC linear velocities
        detail_ax[2].plot(
            t_gt,
            action_label_raw_plot[:, 0],
            label="GT linear",
            linewidth=2.5,
        )

        for k in range(len(uc_actions_raw_plot)):
            detail_ax[2].plot(
                t_pred,
                uc_actions_raw_plot[k, :, 0],
                color="red",
                alpha=0.15,
                linewidth=1.0,
            )

        detail_ax[2].plot(
            t_pred,
            uc_actions_raw_plot[:, :, 0].mean(0),
            color="red",
            linewidth=3,
            label="UC mean",
        )

        add_velocity_arrows(
            detail_ax[2],
            action_label_raw_eval[:, 0],
            uc_mean_linear,
        )

        detail_ax[2].set_title("GT + all UC linear vel")
        detail_ax[2].set_xlabel("timestep (x0.03s)")
        detail_ax[2].set_ylabel("linear vel")
        detail_ax[2].legend(loc="upper right")
        detail_ax[2].grid(True)

        detail_ax[2].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
        )

        # 4) GT linear velocity + all GC linear velocities
        detail_ax[3].plot(
            t_gt,
            action_label_raw_plot[:, 0],
            label="GT linear",
            linewidth=2.5,
        )

        for k in range(len(gc_actions_raw_plot)):
            detail_ax[3].plot(
                t_pred,
                gc_actions_raw_plot[k, :, 0],
                color="green",
                alpha=0.15,
                linewidth=1.0,
            )

        detail_ax[3].plot(
            t_pred,
            gc_actions_raw_plot[:, :, 0].mean(0),
            color="green",
            linewidth=3,
            label="GC mean",
        )

        add_velocity_arrows(
            detail_ax[3],
            action_label_raw_eval[:, 0],
            gc_mean_linear,
        )

        detail_ax[3].set_title("GT + all GC linear vel")
        detail_ax[3].set_xlabel("timestep (x0.03s)")
        detail_ax[3].set_ylabel("linear vel")
        detail_ax[3].legend(loc="upper right")
        detail_ax[3].grid(True)

        detail_ax[3].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
        )

        # 5) GT angular velocity + all UC angular velocities
        detail_ax[4].plot(
            t_gt,
            action_label_raw_plot[:, 1],
            label="GT angular",
            linewidth=2.5,
        )

        for k in range(len(uc_actions_raw_plot)):
            detail_ax[4].plot(
                t_pred,
                uc_actions_raw_plot[k, :, 1],
                color="red",
                alpha=0.15,
                linewidth=1.0,
            )

        detail_ax[4].plot(
            t_pred,
            uc_actions_raw_plot[:, :, 1].mean(0),
            color="red",
            linewidth=3,
            label="UC mean",
        )

        add_velocity_arrows(
            detail_ax[4],
            action_label_raw_eval[:, 1],
            uc_mean_angular,
        )

        detail_ax[4].set_title("GT + all UC angular vel")
        detail_ax[4].set_xlabel("timestep (x0.03s)")
        detail_ax[4].set_ylabel("angular vel")
        detail_ax[4].legend(loc="upper right")
        detail_ax[4].grid(True)

        detail_ax[4].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
        )

        # 6) GT angular velocity + all GC angular velocities
        detail_ax[5].plot(
            t_gt,
            action_label_raw_plot[:, 1],
            label="GT angular",
            linewidth=2.5,
        )

        for k in range(len(gc_actions_raw_plot)):
            detail_ax[5].plot(
                t_pred,
                gc_actions_raw_plot[k, :, 1],
                color="green",
                alpha=0.15,
                linewidth=1.0,
            )

        detail_ax[5].plot(
            t_pred,
            gc_actions_raw_plot[:, :, 1].mean(0),
            color="green",
            linewidth=3,
            label="GC mean",
        )

        add_velocity_arrows(
            detail_ax[5],
            action_label_raw_eval[:, 1],
            gc_mean_angular,
        )

        detail_ax[5].set_title("GT + all GC angular vel")
        detail_ax[5].set_xlabel("timestep (x0.03s)")
        detail_ax[5].set_ylabel("angular vel")
        detail_ax[5].legend(loc="upper right")
        detail_ax[5].grid(True)

        detail_ax[5].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
        )

        detail_fig.tight_layout(rect=[0, 0, 1, 0.96])

        detail_save_name = f"step{global_step}_sample_{i}_detail.png"
        detail_save_path = os.path.join(visualize_path, detail_save_name)
        detail_fig.savefig(detail_save_path)

        if use_wandb:
            wandb_list.append(wandb.Image(detail_save_path))
            
        plt.close(detail_fig)
        del detail_fig
        gc.collect()


    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)

    if visualize_detail:
        plt.close("all")

        del uc_actions_list
        del gc_actions_list
        del gc_distances_list
        del wandb_list
        del action_label_long_raw
        del action_label_long

        gc.collect()

    return traj_error_records

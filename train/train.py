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
from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.vint_dataset_episode import ViNT_H5_Action_Dataset
from vint_train.data.vint_dataset_episode import ViNT_H5_Action_Dataset_Test

from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)


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

            train_action_stats = None

            if data_config["test_other_episode"] is True:

                dataset = ViNT_H5_Action_Dataset(
                        h5_path=data_config["h5_path"],
                        split="train",
                        test_episode_num=0,
                        seed=config.get("seed", 42),
                        dataset_index=0,
                        image_key=data_config.get(
                            "image_key",
                            "image_bottom"
                        ),
                        action_key=data_config.get(
                            "action_key",
                            "encoder"
                        ),
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        normalize=config["normalize"],
                        action_stats=train_action_stats,
                        predict_velocity=config["predict_velocity"]
                    )
                
                train_action_stats = dataset.action_stats
                train_dataset.append(dataset)

                dataset = ViNT_H5_Action_Dataset_Test(
                        h5_path=data_config["test_h5_path"],
                        split="test",
                        test_episode_num=data_config["test_episode_num"],
                        seed=config.get("seed", 42),
                        dataset_index=0,
                        image_key=data_config.get(
                            "image_key",
                            "image_bottom"
                        ),
                        action_key=data_config.get(
                            "action_key",
                            "encoder"
                        ),
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        normalize=config["normalize"],
                        action_stats=train_action_stats,
                        predict_velocity=config["predict_velocity"],
                        use_global_goal_for_test=data_config["use_global_goal_for_test"]
                    )
                
                dataset_type = f"{dataset_name}_test"
                test_dataloaders[dataset_type] = dataset
            
            else:

                for split_type in ["train", "test"]:
                    dataset = ViNT_H5_Action_Dataset(
                        h5_path=data_config["h5_path"],
                        split=split_type,
                        test_episode_num=data_config.get("test_episode_num", 0),
                        seed=config.get("seed", 42),
                        dataset_index=0,
                        image_key=data_config.get(
                            "image_key",
                            "image_bottom"
                        ),
                        action_key=data_config.get(
                            "action_key",
                            "encoder"
                        ),
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        normalize=config["normalize"],
                        action_stats=train_action_stats,
                        predict_velocity=config["predict_velocity"]
                    )

                    if split_type == "train":
                        train_action_stats = dataset.action_stats
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{split_type}"
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
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
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

    # 이전 학습 이어하기 (Resume) 
    if "load_run" in config:
        # 로그 폴더 찾기
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
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
            max_distance=config["distance"]["max_dist_cat"]
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

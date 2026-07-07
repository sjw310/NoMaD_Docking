import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding


class SensorTokenizer(nn.Module):

    def __init__(self, input_dim, context_size, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                context_size * input_dim,
                embed_dim,
            ),
            nn.GELU(),
            nn.Linear(
                embed_dim,
                embed_dim,
            ),
        )

    def forward(self, x):
        return self.net(x)


class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        sensor_context_sizes: Optional[Dict[str, int]] = None,
        use_encoder: bool = False,
        use_imu: bool = False,
        use_lidar: bool = False,
        num_image_keys: int = 1
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        self.num_image_keys = num_image_keys
        self.obs_channels = 3 * self.num_image_keys

        # Initialize the observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=self.obs_channels) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        self.use_encoder = use_encoder
        self.use_imu = use_imu
        self.use_lidar = use_lidar

        if sensor_context_sizes is None:
            sensor_context_sizes = {
                "encoder": context_size + 1,
                "imu": context_size + 1,
                "lidar": context_size + 1,
            }

        self.sensor_context_sizes = sensor_context_sizes


        self.encoder_tokenizer = None
        self.imu_tokenizer = None
        self.lidar_tokenizer = None

        self.num_sensor_tokens = 0

        if self.use_encoder:
            self.encoder_tokenizer = SensorTokenizer(
                input_dim=2,
                context_size=self.sensor_context_sizes["encoder"],
                embed_dim=obs_encoding_size,
            )
            self.num_sensor_tokens += 1

        if self.use_imu:
            self.imu_tokenizer = SensorTokenizer(
                input_dim=6,
                context_size=self.sensor_context_sizes["imu"],
                embed_dim=obs_encoding_size,
            )
            self.num_sensor_tokens += 1

        if self.use_lidar:
            self.lidar_tokenizer = SensorTokenizer(
                input_dim=360,
                context_size=self.sensor_context_sizes["lidar"],
                embed_dim=obs_encoding_size,
            )
            self.num_sensor_tokens += 1

        # Initialize the goal encoder
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=2 * self.obs_channels) # obs+goal
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # 2 : current + goal 
        self.seq_len = self.context_size + 2 + self.num_sensor_tokens

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        self.goal_mask = torch.zeros((1, self.seq_len), dtype=torch.bool)

        self.goal_mask[:, -1] = True # Mask out the goal 
        
        self.no_mask = torch.zeros((1, self.seq_len), dtype=torch.bool) 
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.seq_len)/(self.context_size + 1))], dim=0)


    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None, encoder_hist=None, imu_hist=None,lidar_hist=None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = obs_img.device
        
        # TODO:
        goal_mask = None

        # Initialize the goal encoding
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)
        
        # Get the input goal mask 
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # Get the goal encoding
        expected_obs_channels = self.obs_channels * (self.context_size + 1)

        assert obs_img.shape[1] == expected_obs_channels, (
            f"Expected obs_img channel={expected_obs_channels}, "
            f"but got {obs_img.shape[1]}"
        )

        assert goal_img.shape[1] == self.obs_channels, (
            f"Expected goal_img channel={self.obs_channels}, "
            f"but got {goal_img.shape[1]}"
        )

        curr_obs_img = obs_img[:, -self.obs_channels:, :, :]

        obsgoal_img = torch.cat(
            [curr_obs_img, goal_img],
            dim=1,
        )# concatenate the obs image/context and goal image --> non image goal?
        
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img) # get encoding of this img
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding) # avg pooling 공간 차원 줄임
        
        if self.goal_encoder._global_params.include_top: # EfficientNet 계열 설정에서 classifier top을 포함하는 경우 처리
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding
        
        # Get the observation encoding
        obs_img = torch.split(obs_img, self.obs_channels, dim=1)
        assert len(obs_img) == self.context_size + 1 # channel 방향으로 붙어 있던 RGB 이미지들을 3채널씩 분리
        
        obs_img = torch.concat(obs_img, dim=0) # 분리한 이미지들을 batch 방향으로 다시 붙임

        obs_encoding = self.obs_encoder.extract_features(obs_img) # 각 observation image를 CNN encoder에 넣어서 feature map 추출
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding) # 공간 average pooling

        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        sensor_tokens = []

        if self.use_encoder and self.encoder_tokenizer is not None and encoder_hist is not None:
            enc_token = self.encoder_tokenizer(encoder_hist)
            enc_token = enc_token.unsqueeze(1)
            sensor_tokens.append(enc_token)

        if self.use_imu and self.imu_tokenizer is not None and imu_hist is not None:
            imu_token = self.imu_tokenizer(imu_hist)
            imu_token = imu_token.unsqueeze(1)
            sensor_tokens.append(imu_token)

        if self.use_lidar and self.lidar_tokenizer is not None and lidar_hist is not None:
            lidar_token = self.lidar_tokenizer(lidar_hist)
            lidar_token = lidar_token.unsqueeze(1)
            sensor_tokens.append(lidar_token)

        if len(sensor_tokens) > 0:
            sensor_tokens = torch.cat(sensor_tokens, dim=1)
            obs_encoding = torch.cat((obs_encoding, sensor_tokens, goal_encoding), dim=1)
        else:
            obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)

        
        # If a goal mask is provided, mask some of the goal tokens
        if goal_mask is not None:
            no_goal_mask = goal_mask.long() # mask를 정수형으로 변환
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        # Apply positional encoding 
        # token 순서 정보를 추가. transformer는 기본적으로 순서 개념이 없으니 첫 번째 obs인지 마지막 obs인지 goal token인지를 알려주는 역할
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)
    
        """
        print("obs_encoding:", obs_encoding.shape)
        print(
            "src_key_padding_mask:",
            src_key_padding_mask.shape if src_key_padding_mask is not None else None,
        )
        print(
            "use_encoder:", self.use_encoder,
            "encoder_hist is None:", encoder_hist is None,
        )
        print(
            "use_imu:", self.use_imu,
            "imu_hist is None:", imu_hist is None,
        )
        print(
            "use_lidar:", self.use_lidar,
            "lidar_hist is None:", lidar_hist is None,
        )
        print("seq_len:", self.seq_len)
        """

        # self attention encoder
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens



# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module








if __name__ == "__main__":
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    h5_path = "/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/saung_dock.h5"

    image_key = "image_bottom"
    action_key = "encoder"


    context_size = 5
    context_spacing = 6
    end_slack = 0
    dt = 0.0333

    sample_stride = 20

    save_dir = "./latent_semicircle_plots"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # "l2" or "cos" or "norm"
    metric_type = "l2"

    # "selected" or "all"
    episode_mode = "all"

    selected_episodes = [1,2,3,4]  # 1-based episode number

   
    # =========================
    # RK4 utils
    # =========================
    def reconstruct_pose_rk4(
        linear_vels,
        angular_vels,
        dt=0.0333,
        initial_pose=(0.0, 0.0, 0.0),
    ):
        n_steps = len(linear_vels)
        trajectory = np.zeros((n_steps + 1, 3), dtype=np.float32)
        trajectory[0] = initial_pose

        def f(q, v, w):
            return np.array([
                v * np.cos(q[2]),
                v * np.sin(q[2]),
                w,
            ], dtype=np.float32)

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

    def align_final_pose_to_origin_y(traj):
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

    # =========================
    # Model load
    # =========================
    model = NoMaD_ViNT(
        context_size=context_size,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=512,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    ).to(device)

    load_project_folder = (
        "/home/sjw00310/Desktop/visualnav-transformer/train/logs/nomad_train/"
        "seed_0_postech_260330_74epi_subgoal_100_interval_nomad_dist_max_norm_p60_a100_d100_2026_06_11_10_58_41"
    )

    latest_path = os.path.join(load_project_folder, "latest.pth")
    latest_checkpoint = torch.load(latest_path, map_location=device)

    def load_model(model, model_type, checkpoint: dict) -> None:
        if model_type == "nomad":
            state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        else:
            loaded_model = checkpoint["model"]
            try:
                state_dict = loaded_model.module.state_dict()
                model.load_state_dict(state_dict, strict=False)
            except AttributeError:
                state_dict = loaded_model.state_dict()
                model.load_state_dict(state_dict, strict=False)

    load_model(model, "nomad", latest_checkpoint)
    model.eval()

    # =========================
    # H5 processing
    # =========================
    with h5py.File(h5_path, "r") as h5:
        print("===== H5 keys =====")
        for key in h5.keys():
            obj = h5[key]
            print(f"{key}: type={type(obj)}")

            if isinstance(obj, h5py.Dataset):
                print(f"  shape: {obj.shape}")
                print(f"  dtype: {obj.dtype}")

        actions = h5[action_key][:]
        episode_ends = h5["episode_ends"][:]
        episode_starts = np.concatenate([[0], episode_ends[:-1]])

        print("actions:", actions.shape)
        print("num episodes:", len(episode_ends))
        print("selected episodes:", selected_episodes)

        if episode_mode == "all":
            episodes_to_process = list(
                range(
                    1,
                    len(episode_ends) + 1
                )
            )
        else:
            episodes_to_process = selected_episodes


        all_aligned_trajs = []
        selected_aligned_trajs = []
        selected_ratio_xy = []
        selected_metric_values = []


     
        # 전체 trajectory는 배경용으로 전부 그림
        for ep_idx, (s, e) in enumerate(zip(episode_starts, episode_ends)):
            s = int(s)
            e = int(e)

            ep_vel = actions[s:e]

            traj = reconstruct_pose_rk4(
                ep_vel[:, 0],
                ep_vel[:, 1],
                dt=dt,
            )

            aligned_traj = align_final_pose_to_origin_y(traj)
            all_aligned_trajs.append(aligned_traj)

        # =========================
        # selected episodes만 latent 계산
        # =========================
        for episode in episodes_to_process:
            ep_idx = episode - 1

            if ep_idx < 0 or ep_idx >= len(episode_ends):
                print(f"Skip invalid episode: {episode}")
                continue

            ep_start = int(episode_starts[ep_idx])
            ep_end = int(episode_ends[ep_idx])

            print("=" * 80)
            print(f"Episode {episode}")
            print("ep_start:", ep_start, "ep_end:", ep_end)
            print("length =", ep_end - ep_start)

            begin_time = ep_start + context_size * context_spacing
            end_time = ep_end - end_slack - 1

            if end_time <= begin_time:
                print(f"Skip too short episode: {episode}")
                continue

            curr_times = list(
                range(
                    begin_time,
                    end_time + 1,
                    sample_stride,
                )
            )
            if curr_times[-1] != end_time:
                curr_times.append(end_time)

            print("begin_time:", begin_time)
            print("end_time:", end_time)
            print("num curr_times:", len(curr_times))

            goal_time = ep_end - 1

            latents = []
            latent_times = []

            for curr_time in curr_times:
                context_times = list(
                    range(
                        curr_time - context_size * context_spacing,
                        curr_time + 1,
                        context_spacing,
                    )
                )

                assert len(context_times) == context_size + 1
                assert context_times[0] >= ep_start
                assert context_times[-1] == curr_time

                obs_imgs = []

                for t in context_times:
                    img = torch.tensor(
                        h5[image_key][t],
                        dtype=torch.float32,
                    ) / 255.0

                    obs_imgs.append(img)

                obs_img = torch.cat(obs_imgs, dim=0).unsqueeze(0)

                goal_img = torch.tensor(
                    h5[image_key][goal_time],
                    dtype=torch.float32,
                ) / 255.0
                goal_img = goal_img.unsqueeze(0)

                obs_img = obs_img.to(device)
                goal_img = goal_img.to(device)

                with torch.no_grad():
                    z = model(obs_img, goal_img)

                latents.append(z.detach().cpu())
                latent_times.append(curr_time)

            latents = torch.cat(latents, dim=0)  # [10, 512]

            z_ref = latents[-1:].clone()

            cos_to_final = F.cosine_similarity(
                latents,
                z_ref.expand_as(latents),
                dim=-1,
            )

            dist_to_final = torch.norm(
                latents - z_ref,
                dim=-1,
            )

            latent_norm = torch.norm(
                latents,
                dim=-1,
            )

            print("latent_times:", latent_times)
            print("cos_to_final:", cos_to_final)
            print("dist_to_final:", dist_to_final)

            aligned_traj = all_aligned_trajs[ep_idx]

            local_indices = [
                curr_time - ep_start
                for curr_time in curr_times
            ]

            ratio_xy = np.array([
                aligned_traj[idx, :2]
                for idx in local_indices
            ])

            selected_aligned_trajs.append((episode, aligned_traj))
            selected_ratio_xy.append(ratio_xy)
            
            if metric_type == "l2":
                selected_metric_values.append(
                    dist_to_final.numpy()
                )

            elif metric_type == "cos":
                selected_metric_values.append(
                    cos_to_final.numpy()
                )

            elif metric_type == "norm":
                selected_metric_values.append(
                    latent_norm.numpy()
                )

            else:
                raise ValueError(metric_type)

        # =========================
        # selected 결과 정리
        # =========================
        selected_sample_xy_np = np.concatenate(selected_ratio_xy, axis=0)
        selected_metric_values_np = np.concatenate(
            selected_metric_values,
            axis=0
        )
        
        # =========================
        # Cumulative radius analysis
        # =========================
        r_samples = np.linalg.norm(
            selected_sample_xy_np,
            axis=1,
        )

        r_max = np.percentile(
            r_samples,
            98,
        )

        num_radius_steps = 80

        radius_grid = np.linspace(
            0.05,
            r_max,
            num_radius_steps,
        )

        cum_mean = np.full(
            num_radius_steps,
            np.nan,
        )

        cum_std = np.full(
            num_radius_steps,
            np.nan,
        )

        cum_count = np.zeros(
            num_radius_steps,
            dtype=np.int32,
        )

        for i, r_th in enumerate(radius_grid):

            mask = r_samples <= r_th

            cum_count[i] = mask.sum()

            if mask.sum() >= 5:
                vals = selected_metric_values_np[mask]

                cum_mean[i] = vals.mean()
                cum_std[i] = vals.std()

        print("radius_grid:", radius_grid)
        print("cum_mean:", cum_mean)
        print("cum_std:", cum_std)
        print("cum_count:", cum_count)

        # =========================
        # Ring-wise radius analysis
        # 각 반경 구간별 mean/std 계산
        # =========================
        r_samples = np.linalg.norm(
            selected_sample_xy_np,
            axis=1,
        )

        r_max = np.percentile(r_samples, 98)
        num_bins = 50

        bin_edges = np.linspace(
            0.0,
            r_max,
            num_bins + 1,
        )

        bin_centers = 0.5 * (
            bin_edges[:-1] + bin_edges[1:]
        )

        ring_mean = np.full(num_bins, np.nan)
        ring_std = np.full(num_bins, np.nan)
        ring_count = np.zeros(num_bins, dtype=np.int32)

        for i in range(num_bins):
            mask = (
                (r_samples >= bin_edges[i]) &
                (r_samples < bin_edges[i + 1])
            )

            ring_count[i] = mask.sum()

            if mask.sum() >= 5:
                vals = selected_metric_values_np[mask]
                ring_mean[i] = vals.mean()
                ring_std[i] = vals.std()

        print("bin_centers:", bin_centers)
        print("ring_mean:", ring_mean)
        print("ring_std:", ring_std)
        print("ring_count:", ring_count)

        valid = ~np.isnan(ring_std)

        best_idx = np.where(valid)[0][
            np.argmin(ring_std[valid])
        ]

        print("min ring std radius:", bin_centers[best_idx])
        print("ring mean:", ring_mean[best_idx])
        print("ring std:", ring_std[best_idx])
        print("count:", ring_count[best_idx])

        # =========================
        # Semicircle heatmap
        # =========================
        all_xy = np.concatenate([tr[:, :2] for tr in all_aligned_trajs], axis=0)

        x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
        y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()

        R = max(
            abs(x_min),
            abs(x_max),
            abs(y_min),
            abs(y_max),
        )

        theta = np.linspace(np.pi, 2 * np.pi, 361)
        radius = np.linspace(0, R, 200)

        Theta, Radius = np.meshgrid(theta, radius)

        X = Radius * np.cos(Theta)
        Y = Radius * np.sin(Theta)

        # =========================
        # Gaussian splatting
        # selected_sample_xy_np: [N, 2]
        # selected_metric_values_np: [N]
        # =========================
        points = selected_sample_xy_np
        values = selected_metric_values_np

        grid_points = np.stack(
            [X.reshape(-1), Y.reshape(-1)],
            axis=1
        )

        sigma = 0.12  # meter, 0.08~0.25 사이 조절
        max_radius = 3.0 * sigma

        Z_sum = np.zeros(len(grid_points), dtype=np.float32)
        W_sum = np.zeros(len(grid_points), dtype=np.float32)

        from scipy.spatial import cKDTree

        tree = cKDTree(points)

        neighbors = tree.query_ball_point(
            grid_points,
            r=max_radius,
        )

        for i, idxs in enumerate(neighbors):
            if len(idxs) == 0:
                continue

            pts = points[idxs]
            vals = values[idxs]

            d2 = np.sum(
                (pts - grid_points[i]) ** 2,
                axis=1,
            )

            w = np.exp(
                -d2 / (2 * sigma ** 2)
            )

            Z_sum[i] = np.sum(w * vals)
            W_sum[i] = np.sum(w)

        Z_flat = np.full(
            len(grid_points),
            np.nan,
            dtype=np.float32,
        )

        valid = W_sum > 1e-8

        Z_flat[valid] = Z_sum[valid] / W_sum[valid]

        Z = Z_flat.reshape(X.shape)

        # Alpha: 근처 sample이 많은 곳/가까운 곳만 진하게
        Alpha = np.zeros_like(Z, dtype=np.float32)
        Alpha.reshape(-1)[valid] = np.clip(
            W_sum[valid] / W_sum[valid].max(),
            0.0,
            1.0,
        )

        # alpha를 조금 부드럽게
        Alpha = Alpha ** 0.5

        # 반원 바깥 투명
        semi_mask = (X ** 2 + Y ** 2 <= R ** 2) & (Y <= 0)
        Alpha = Alpha * semi_mask.astype(np.float32)

        # NaN은 색상 계산용으로만 임시 값 채움
        Z_plot = np.where(
            np.isnan(Z),
            np.nanmean(values),
            Z,
        )

        # =========================
        # Plot
        # =========================
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 8),
            gridspec_kw={"width_ratios": [1.1, 1.0]},
        )

        ax_traj = axes[0]
        ax_semi = axes[1]

        # -------------------------
        # Left: all trajectories
        # -------------------------
        for tr in all_aligned_trajs:
            ax_traj.plot(
                tr[:, 0],
                tr[:, 1],
                color="lightgray",
                alpha=0.5,
                linewidth=0.8,
            )

        # selected episodes 강조
        for episode, aligned_traj in selected_aligned_trajs:


            if episode_mode == "selected":
                label = f"ep {episode}"
            else:
                label = None

            ax_traj.plot(
                aligned_traj[:, 0],
                aligned_traj[:, 1],
                linewidth=1.0,
                alpha=0.9,
                label=label,
            )

        sc = ax_traj.scatter(
            selected_sample_xy_np[:, 0],
            selected_sample_xy_np[:, 1],
            c=selected_metric_values_np,
            cmap="jet",
            s=20,
            edgecolors="black",
            linewidths=0.4,
            zorder=10,
            vmin=selected_metric_values_np.min(),
            vmax=selected_metric_values_np.max(),
        )

        ax_traj.scatter(
            0,
            0,
            color="black",
            marker="x",
            s=120,
            linewidths=3,
            label="aligned final",
        )


        ax_traj.set_title("Trajectories with Selected Episode Latent Distance")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.axis("equal")
        ax_traj.grid(True)
        if episode_mode == "selected":
            ax_traj.legend()

        # -------------------------
        # Right: lower semicircle
        # -------------------------
        import matplotlib.colors as colors

        vmin = selected_metric_values_np.min()
        vmax = selected_metric_values_np.max()

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("jet")

        rgba = cmap(norm(Z_plot))
        rgba[..., 3] = Alpha

        # RGBA 직접 표시: Gaussian weighted color + alpha
        pcm = ax_semi.pcolormesh(
            X,
            Y,
            rgba,
            shading="auto"
        )

        # colorbar용 별도 mappable
        sm = plt.cm.ScalarMappable(
            norm=norm,
            cmap=cmap
        )
        sm.set_array([])

        ax_semi.scatter(
            selected_sample_xy_np[:, 0],
            selected_sample_xy_np[:, 1],
            c=selected_metric_values_np,
            cmap="jet",
            s=8,
            edgecolors="none",
            alpha=0.8,
            vmin=selected_metric_values_np.min(),
            vmax=selected_metric_values_np.max(),
        )

        ax_semi.plot(
            R * np.cos(theta),
            R * np.sin(theta),
            color="black",
            linewidth=1.2,
        )

        ax_semi.plot(
            [-R, R],
            [0, 0],
            color="black",
            linewidth=1.2,
        )

        ax_semi.scatter(
            0,
            0,
            color="black",
            s=70,
            zorder=10,
            label="aligned final",
        )

    

        ax_semi.set_title("Spatially Interpolated Latent Metric")
        ax_semi.set_xlabel("x")
        ax_semi.set_ylabel("y")
        ax_semi.axis("equal")
        ax_semi.grid(True, alpha=0.25)
        ax_semi.legend(loc="lower right")

        cbar = fig.colorbar(
            sm,
            ax=[ax_traj, ax_semi],
            shrink=0.85,
            pad=0.02,
        )

        if metric_type == "l2":
            metric_label = "L2 distance to final latent"

        elif metric_type == "cos":
            metric_label = "Cosine similarity to final latent"

        elif metric_type == "norm":
            metric_label = "Latent norm"

        cbar.set_label(metric_label)

        if episode_mode == "all":
            title_text = "All Episodes"
        else:
            title_text = f"Selected Episodes {selected_episodes}"

        fig.suptitle(
            f"{metric_label}\n{title_text}",
            fontsize=16,
        )

        plt.tight_layout()

        save_name = (
            "all_episodes"
            if episode_mode == "all"
            else "selected_episodes"
        )

        save_path = os.path.join(
            save_dir,
            f"{save_name}_{metric_type}.png"
        )

        plt.savefig(save_path, dpi=200)
        plt.show()

        print("Saved:", save_path)


        fig_curve, ax_curve = plt.subplots(
            figsize=(8,5)
        )

        valid = ~np.isnan(cum_mean)

        ax_curve.plot(
            radius_grid[valid],
            cum_mean[valid],
            linewidth=2,
            label="mean"
        )

        ax_curve.fill_between(
            radius_grid[valid],
            cum_mean[valid] - cum_std[valid],
            cum_mean[valid] + cum_std[valid],
            alpha=0.25,
            label="±1 std"
        )

        ax_curve.set_xlabel(
            "Radius from goal [m]"
        )

        ax_curve.set_ylabel(
            metric_label
        )

        ax_curve.set_title(
            "Cumulative Radius Analysis"
        )

        ax_curve.grid(True)
        ax_curve.legend()

        plt.tight_layout()
        plt.show()



        fig_ring, ax_ring = plt.subplots(figsize=(8, 5))

        valid = ~np.isnan(ring_mean)

        ax_ring.plot(
            bin_centers[valid],
            ring_mean[valid],
            linewidth=2,
            label="ring mean",
        )

        ax_ring.fill_between(
            bin_centers[valid],
            ring_mean[valid] - ring_std[valid],
            ring_mean[valid] + ring_std[valid],
            alpha=0.25,
            label="±1 std",
        )

        ax_ring.axvline(
            bin_centers[best_idx],
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"min std r={bin_centers[best_idx]:.2f}m",
        )

        ax_ring.set_xlabel("Radius from goal [m]")
        ax_ring.set_ylabel(metric_label)
        ax_ring.set_title("Ring-wise Radius Metric Analysis")
        ax_ring.grid(True)
        ax_ring.legend()

        plt.tight_layout()
        plt.show()
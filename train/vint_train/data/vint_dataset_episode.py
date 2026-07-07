import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from typing import Tuple


class ViNT_H5_Action_Dataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        dataset_index: int = 0,
        image_keys: list[str] = None,
        action_key: str = "encoder",
        image_size: Tuple[int, int] = (320, 240),
        split: str = "train",
        test_episode_num: int = 0,
        seed: int = 42,
        context_spacing: int = 1,
        waypoint_spacing: int = 1,
        min_dist_cat: int = 1,
        max_dist_cat: int = 20,
        min_action_distance: int = 0,
        max_action_distance: int = 10,
        negative_mining: bool = False,
        len_traj_pred: int = 5,
        context_size: int = 5,
        context_type: str = "temporal",
        end_slack: int = 0,
        normalize: bool = False,
        action_stats=None,
        pose_stats=None,
        predict_velocity: bool = True,
        use_global_goal_for_test: bool = False,
        percent_99: bool = True,
        test_goal_interval: int = 100,
        encoder_key: str = None,
        imu_key: str = None,
        lidar_key: str = None,
        return_sensor_history: bool = False,
        crop_condition_fn=None,
        crop_dt: float = 0.0333,

        encoder_imu_context_size: int = None,
        encoder_imu_context_spacing: int = 1,
        
        lidar_context_size: int = None,
        lidar_context_spacing: int = None,
        chunk_size: int = 30,
        exclude_train_episode_nums: list[int] | None = None,
    ):
        self.h5_path = h5_path
        self.dataset_index = dataset_index
        self.image_keys = image_keys
        self.action_key = action_key
        self.image_size = image_size

        self.split = split
        self.test_episode_num = test_episode_num
        self.seed = seed

        self.context_spacing = context_spacing
        self.waypoint_spacing = waypoint_spacing
        self.min_dist_cat = min_dist_cat
        self.max_dist_cat = max_dist_cat
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance
        self.negative_mining = negative_mining
        self.len_traj_pred = len_traj_pred
        self.context_type = context_type
        self.end_slack = end_slack

        self.normalize = normalize
        self.action_stats = action_stats
        self.pose_stats = pose_stats
        self.predict_velocity = predict_velocity
        self.use_global_goal_for_test = use_global_goal_for_test
        self.percent_99 = percent_99
        self.test_goal_interval = test_goal_interval

        self.encoder_key = encoder_key
        self.imu_key = imu_key
        self.lidar_key = lidar_key
        self.return_sensor_history = return_sensor_history

        self.encoder_data = None
        self.imu_data = None
        self.lidar_data = None

        assert self.context_type == "temporal"
        assert self.split in ["train", "test"]

        self.num_action_params = 2
        self.chunk_size = chunk_size
        self.exclude_train_episode_nums = (
            []
            if exclude_train_episode_nums is None
            else exclude_train_episode_nums
        )

        self.h5 = None
        self.images = None

        self.crop_condition_fn = crop_condition_fn
        self.crop_dt = crop_dt

        self.context_size = context_size
        self.context_spacing = context_spacing

        # encoder / imu
        self.encoder_imu_context_size = (
            context_size + 1
            if encoder_imu_context_size is None
            else encoder_imu_context_size
        )
        self.encoder_imu_context_spacing = encoder_imu_context_spacing

        # lidar
        self.lidar_context_size = (
            context_size
            if lidar_context_size is None
            else lidar_context_size
        )
        self.lidar_context_spacing = (
            context_spacing
            if lidar_context_spacing is None
            else lidar_context_spacing
        )

        with h5py.File(self.h5_path, "r") as h5:
            self.actions = h5[self.action_key][:]
            self.episode_ends = h5["episode_ends"][:]

            if self.return_sensor_history:
                if self.encoder_key is not None and self.encoder_key in h5:
                    self.encoder_data = h5[self.encoder_key][:]

                if self.imu_key is not None and self.imu_key in h5:
                    self.imu_data = h5[self.imu_key][:]

                if self.lidar_key is not None and self.lidar_key in h5:
                    self.lidar_data = h5[self.lidar_key][:]

        assert self.actions.shape[1] == 2, self.actions.shape

        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.selected_episodes = self._make_episode_split()

        if self.normalize:
            if action_stats is None:
                raise ValueError(
                    "normalize=True requires action_stats. "
                    "Compute global action_stats in train.py and pass it to all datasets."
                )

            self.action_stats = {
                "min": np.asarray(action_stats["min"], dtype=np.float32),
                "max": np.asarray(action_stats["max"], dtype=np.float32),
                "scale": np.asarray(action_stats["scale"], dtype=np.float32),
            }


        self.aligned_trajectories = self._precompute_aligned_trajectories()


        self.index_to_data, self.goals_index = self._build_index()

        print(f"[{self.split}] h5_path : {self.h5_path}")
        print(f"[{self.split}] image_keys:", self.image_keys)
        print(f"[{self.split}] episodes: {len(self.selected_episodes)}")
        print(f"[{self.split}] samples : {len(self.index_to_data)}")

    def _make_episode_split(self):
        num_episodes = len(self.episode_ends)
        episode_indices = np.arange(num_episodes)

        if self.split == "train":

            exclude = np.array(
                [ep - 1 for ep in self.exclude_train_episode_nums],
                dtype=int,
            )

            # np.setdiff1d : 두 배열의 차집합을 반환
            train_eps = np.setdiff1d(
                episode_indices,
                exclude,
                assume_unique=False,
            )

            return np.sort(train_eps)

        # test는 test_episode_num 번째 episode만 사용
        test_ep = int(self.test_episode_num) - 1
        assert 0 <= test_ep < num_episodes, (
            f"Invalid test_episode_num={self.test_episode_num}, "
            f"num_episodes={num_episodes}"
        )

        return np.array([test_ep])

    def reconstruct_pose_rk4(
        self,
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

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi


    def align_final_pose_to_origin_y(self, traj):
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

        aligned_theta = self.wrap_angle(traj[:, 2] + rot_angle)

        aligned_traj = np.zeros_like(traj)
        aligned_traj[:, :2] = aligned_xy
        aligned_traj[:, 2] = aligned_theta

        return aligned_traj

    def _precompute_aligned_trajectories(self):
        aligned_trajectories = {}

        for ep_idx, (ep_start, ep_end) in enumerate(
            zip(self.episode_starts, self.episode_ends)
        ):
            ep_start = int(ep_start)
            ep_end = int(ep_end)

            ep_vel = self.actions[ep_start:ep_end].astype(np.float32)

            traj = self.reconstruct_pose_rk4(
                linear_vels=ep_vel[:, 0],
                angular_vels=ep_vel[:, 1],
                dt=self.crop_dt,
                initial_pose=(0.0, 0.0, 0.0),
            )

            aligned_traj = self.align_final_pose_to_origin_y(traj)
            aligned_trajectories[ep_idx] = aligned_traj

        return aligned_trajectories

    def _get_pose_align_target(self, ep_idx, curr_time):
        ep_start = int(self.episode_starts[ep_idx])
        local_t = int(curr_time - ep_start)

        aligned_traj = self.aligned_trajectories[ep_idx]

        x = aligned_traj[local_t, 0]
        y = aligned_traj[local_t, 1]
        theta = aligned_traj[local_t, 2]

        return np.array([x, 
                        y, 
                        theta], dtype=np.float32)

    def _get_valid_time_start_by_condition(self, ep_start, ep_end):
        
        if self.crop_condition_fn is None:
            return ep_start

        ep_vel = self.actions[ep_start:ep_end].astype(np.float32)

        traj = self.reconstruct_pose_rk4(
            linear_vels=ep_vel[:, 0],
            angular_vels=ep_vel[:, 1],
            dt=self.crop_dt,
            initial_pose=(0.0, 0.0, 0.0),
        )

        traj = self.align_final_pose_to_origin_y(traj)

        valid = self.crop_condition_fn(traj)

        valid = np.asarray(valid, dtype=bool)

        if len(valid) != len(traj):
            raise ValueError(
                f"crop_condition_fn must return bool array with len(traj). "
                f"got {len(valid)}, expected {len(traj)}"
            )

        outside = ~valid
        outside_idx = np.where(outside)[0]

        if len(outside_idx) == 0:
            return ep_start

        if len(outside_idx) == len(traj):
            return None

        crop_local_start = int(outside_idx[-1]) + 1

        if crop_local_start >= ep_end - ep_start:
            return None

        return ep_start + crop_local_start


    def _build_index(self):
        samples_index = []
        goals_index = []

        for ep_idx in self.selected_episodes:
            ep_start = int(self.episode_starts[ep_idx])
            ep_end = int(self.episode_ends[ep_idx])

            for goal_time in range(ep_start, ep_end):
                goals_index.append((ep_idx, goal_time))

            valid_start = self._get_valid_time_start_by_condition(ep_start, ep_end)

            if valid_start is None:
                continue

            required_history = self.context_size * self.context_spacing

            if self.return_sensor_history:

                if self.encoder_data is not None or self.imu_data is not None:
                    required_history = max(
                        required_history,
                        (self.encoder_imu_context_size - 1)* self.encoder_imu_context_spacing,
                    )

                if self.lidar_data is not None:
                    required_history = max(
                        required_history,
                        self.lidar_context_size* self.lidar_context_spacing,
                    )

            begin_time = max(
                ep_start + required_history,
                valid_start,
            )

            chunk_time_size = self.chunk_size * self.waypoint_spacing

            action_horizon = self.len_traj_pred * self.waypoint_spacing

            # padding을 쓸 거면 action_horizon을 빼지 않아도 됨
            end_time = ep_end - self.end_slack

            if end_time <= begin_time:
                continue

            for curr_time in range(begin_time, end_time):
                samples_index.append((ep_idx, curr_time))

        return samples_index, goals_index


    def _sample_goal(self, ep_idx, curr_time, max_goal_dist):
        max_goal_cat = max_goal_dist

        if self.negative_mining and np.random.rand() < 0.05:
            neg_ep_idx, neg_goal_time = self._sample_negative()
            return neg_ep_idx, neg_goal_time, True

        goal_offset = np.random.randint(1, max_goal_cat + 1)
        chunk_size = self.chunk_size * self.waypoint_spacing

        goal_time = curr_time + int(goal_offset * chunk_size)

        return ep_idx, goal_time, False

    def _sample_negative(self):
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _ensure_h5_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
            self.images = [
                self.h5[key]
                for key in self.image_keys
            ]

    def _load_image(self, time):
        self._ensure_h5_open()

        imgs = []

        for image_ds in self.images:

            img = image_ds[time]
            img = torch.as_tensor(img, dtype=torch.float32) / 255.0

            if self.image_size is not None:
                w, h = self.image_size
                img = TF.resize(img, [h, w])

            imgs.append(img)

        # channel 방향으로 concat
        img = torch.cat(imgs, dim=0)

        return img

    def _compute_actions(self, curr_time, ep_end):
        needed_len = self.len_traj_pred * self.waypoint_spacing

        start_index = curr_time
        end_index = curr_time + needed_len
        valid_end = min(end_index, ep_end)

        raw_actions = self.actions[start_index:valid_end].astype(np.float32)

        pad_len = needed_len - len(raw_actions)

        if pad_len > 0:
            pad = np.zeros((pad_len, 2), dtype=np.float32)
            raw_actions = np.concatenate([raw_actions, pad], axis=0)

        if self.predict_velocity:
            actions = raw_actions[::self.waypoint_spacing]

            traj = self.reconstruct_pose_rk4(
                linear_vels=actions[:, 0],
                angular_vels=actions[:, 1],
                dt=0.0333,
                initial_pose=(0.0, 0.0, 0.0),
            )

            goal_pos = traj[-1, :2]

        else:
            traj = self.reconstruct_pose_rk4(
                linear_vels=raw_actions[:, 0],
                angular_vels=raw_actions[:, 1],
                dt=0.0333,
                initial_pose=(0.0, 0.0, 0.0),
            )

            waypoint_indices = np.arange(
                0,
                self.len_traj_pred * self.waypoint_spacing + 1,
                self.waypoint_spacing,
            )

            waypoint_xy = traj[waypoint_indices, :2]
            actions = np.diff(waypoint_xy, axis=0)
            goal_pos = waypoint_xy[-1]

        actions = actions.astype(np.float32)
        goal_pos = goal_pos.astype(np.float32)

        if self.normalize:
            if self.percent_99:
                actions = np.clip(
                    actions,
                    self.action_stats["min"],
                    self.action_stats["max"],
                )

            actions = (
                2.0
                * (actions - self.action_stats["min"])
                / self.action_stats["scale"]
                - 1.0
            )

        assert actions.shape == (self.len_traj_pred, 2), actions.shape

        return actions, goal_pos

    def __len__(self):
        return len(self.index_to_data)

    def __getitem__(self, i):
        ep_idx, curr_time = self.index_to_data[i]

        ep_start = int(self.episode_starts[ep_idx])
        ep_end = int(self.episode_ends[ep_idx])

        if self.use_global_goal_for_test and self.split == "test":
            goal_time = ep_end - 1
            goal_is_negative = False

        elif self.split == "test":
            local_t = curr_time - ep_start
            goal_local_t = (
                (local_t // self.test_goal_interval) + 1
            ) * self.test_goal_interval

            goal_time = min(ep_start + goal_local_t, ep_end - 1)
            goal_is_negative = False

        else:

            goal_time = ep_end - 1
            goal_is_negative = False
            
            """
            _, goal_time, goal_is_negative = self._sample_goal(
                ep_idx,
                curr_time,
                max_goal_dist,
            )
            """

        # image
        context_times = list(
            range(
                curr_time - self.context_size * self.context_spacing,
                curr_time + 1,
                self.context_spacing,
            )
        )

        # encoder / imu
        context_times_encoder_imu = list(
            range(
                curr_time
                - (self.encoder_imu_context_size - 1)* self.encoder_imu_context_spacing,
                curr_time + 1,
                self.encoder_imu_context_spacing,
            )
        )

        # lidar
        context_times_lidar = list(
            range(
                curr_time
                - self.lidar_context_size* self.lidar_context_spacing,
                curr_time + 1,
                self.lidar_context_spacing,
            )
        )
        sensor_dict = {}

        if self.return_sensor_history:
            
            if self.encoder_data is not None:
                sensor_dict["encoder_hist"] = torch.as_tensor(
                    self.encoder_data[context_times_encoder_imu],
                    dtype=torch.float32,
                )

            if self.imu_data is not None:
                sensor_dict["imu_hist"] = torch.as_tensor(
                    self.imu_data[context_times_encoder_imu],
                    dtype=torch.float32,
                )

            if self.lidar_data is not None:
                sensor_dict["lidar_hist"] = torch.as_tensor(
                    self.lidar_data[context_times_lidar],
                    dtype=torch.float32,
                )

        obs_image = torch.cat(
            [self._load_image(t) for t in context_times],
            dim=0,
        )

        goal_image = self._load_image(goal_time)

        actions, goal_pos = self._compute_actions(curr_time, ep_end)

        pose_align_target = self._get_pose_align_target(ep_idx, curr_time)
        pose_align_target = (2.0*((pose_align_target - self.pose_stats["min"])/self.pose_stats["scale"])-1.0)

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)

        if self.use_global_goal_for_test and self.split == "test":
            remaining_dist = (goal_time - curr_time) // self.waypoint_spacing
            distance = min(self.max_dist_cat, remaining_dist)

        elif goal_is_negative:
            distance = self.max_dist_cat

        else:
            if self.split == "test":
                distance = (goal_time - curr_time) // self.waypoint_spacing
                distance = min(distance, self.max_dist_cat)
            else:
                chunk_size = self.chunk_size * self.waypoint_spacing
                distance = (goal_time - curr_time) // chunk_size

        action_mask = (
            (distance < self.max_action_distance)
            and (distance > self.min_action_distance)
            and (not goal_is_negative)
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.as_tensor(ep_idx, dtype=torch.int64),
            torch.as_tensor(curr_time, dtype=torch.int64),
            torch.as_tensor(pose_align_target, dtype=torch.float32),
            sensor_dict,

        )

    def close(self):
        if hasattr(self, "h5") and self.h5 is not None:
            self.h5.close()
            self.h5 = None
            self.images = None

def debug_first90_velocity():
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    h5_path = "/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/train_episode_postech_260330_dock.h5"

    with h5py.File(h5_path, "r") as h5:
        actions = h5["encoder"][:]
        episode_ends = h5["episode_ends"][:]

    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for ep_idx in range(len(episode_ends)):
        start = int(episode_starts[ep_idx])
        end = int(episode_ends[ep_idx])

        vel = actions[start:min(start + 90, end)]

        t = np.arange(len(vel))

        ax[0].plot(t, vel[:, 0], alpha=0.7)
        ax[1].plot(t, vel[:, 1], alpha=0.7)

    ax[0].set_title("Linear velocity (first 90 steps)")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("v")
    ax[0].grid(True)

    ax[1].set_title("Angular velocity (first 90 steps)")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("w")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def debug_train_velocity_distribution():
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt


    h5_paths = [
        #"/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/train_episode_postech_260328_dock.h5",
        #"/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/train_episode_postech_260330_dock.h5",
        #"/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/saung_dock.h5"
        "/home/sjw00310/Desktop/diffusion_policy_robot_docking/dataset/h5_dataset/260619.h5"
    ]

    action_key = "encoder"

    # 제외할 test episode가 있으면 여기에 직접 적기
    test_episode_list = []
    # 제외할 test episode (1-based 번호)
    # test_episode_list = [1, 5, 12]
    
    all_train_actions = []

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5:
            actions = h5[action_key][:]
            episode_ends = h5["episode_ends"][:]

        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        num_episodes = len(episode_ends)
        episode_indices = np.arange(num_episodes)

        test_episodes = np.array(
            [ep - 1 for ep in test_episode_list],
            dtype=int,
        )

        train_episodes = np.setdiff1d(
            episode_indices,
            test_episodes,
            assume_unique=False,
        )

        selected_indices = []
        for ep_idx in train_episodes:
            ep_start = int(episode_starts[ep_idx])
            ep_end = int(episode_ends[ep_idx])
            selected_indices.append(np.arange(ep_start, ep_end))

        selected_indices = np.concatenate(selected_indices)
        train_actions_one_h5 = actions[selected_indices].astype(np.float32)

        print(f"[Loaded] {h5_path}")
        print("  actions:", actions.shape)
        print("  train_actions:", train_actions_one_h5.shape)

        all_train_actions.append(train_actions_one_h5)

    train_actions = np.concatenate(all_train_actions, axis=0)

    v = train_actions[:, 0]
    w = train_actions[:, 1]

    action_min = np.min(train_actions, axis=0)
    action_max = np.max(train_actions, axis=0)
    action_scale = action_max - action_min

    # -----------------------------
    # Percentile (1% ~ 99%)
    # -----------------------------
    v_p1, v_p99 = np.percentile(v, [1, 99])
    w_p1, w_p99 = np.percentile(w, [1, 99])

    print("[99% Percentile]")
    print(f"v : p1={v_p1:.4f}, p99={v_p99:.4f}")
    print(f"w : p1={w_p1:.4f}, p99={w_p99:.4f}")

    print("=" * 60)
    print("[Train velocity stats]")
    #print("h5_path:", h5_path)
    print("actions shape:", actions.shape)
    print("train_actions shape:", train_actions.shape)
    print("num episodes:", num_episodes)
    print("test episodes (1-based):", test_episode_list)
    print("train episodes:", len(train_episodes))
    print()
    print("[Min-Max normalization values]")
    print("min   [v, w]:", action_min)
    print("max   [v, w]:", action_max)
    print("scale [v, w]:", action_scale)
    print()
    print("[Mean / Std]")
    print("mean  [v, w]:", np.mean(train_actions, axis=0))
    print("std   [v, w]:", np.std(train_actions, axis=0))
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # =========================
    # Linear velocity
    # =========================
    axes[0].hist(v, bins=100)

    axes[0].axvline(
        v_p1,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"1% = {v_p1:.3f}",
    )

    axes[0].axvline(
        v_p99,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"99% = {v_p99:.3f}",
    )

    axes[0].set_title(
        f"Linear velocity v\n"
        f"min={action_min[0]:.4f}, max={action_max[0]:.4f}"
    )
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("count")
    axes[0].grid(True)
    axes[0].legend()


    # =========================
    # Angular velocity
    # =========================
    axes[1].hist(w, bins=100)

    axes[1].axvline(
        w_p1,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"1% = {w_p1:.3f}",
    )

    axes[1].axvline(
        w_p99,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"99% = {w_p99:.3f}",
    )

    axes[1].set_title(
        f"Angular velocity w\n"
        f"min={action_min[1]:.4f}, max={action_max[1]:.4f}"
    )
    axes[1].set_xlabel("w")
    axes[1].set_ylabel("count")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Train data velocity distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    debug_train_velocity_distribution()
    #debug_first90_velocity()


"""
postech 260330
[Min-Max normalization values]
min   [v, w]: [-0.2649721 -0.7315362]
max   [v, w]: [0.2977743  0.30039853]
scale [v, w]: [0.5627464 1.0319347]
"""



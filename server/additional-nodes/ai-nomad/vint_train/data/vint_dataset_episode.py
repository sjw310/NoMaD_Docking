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
        image_key: str = "image_bottom",
        action_key: str = "encoder",
        image_size: Tuple[int, int] = (320, 240),  # (W, H)
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
        action_stats = None,
        predict_velocity = True
    ):
        self.h5_path = h5_path
        self.dataset_index = dataset_index
        self.image_key = image_key
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
        self.context_size = context_size
        self.context_type = context_type
        self.end_slack = end_slack

        self.normalize = normalize
        self.action_stats = action_stats

        assert self.context_type == "temporal"
        assert self.split in ["train", "test"]

        self.num_action_params = 2
        self.predict_velocity = predict_velocity

        self.h5 = None
        self.images = None

        with h5py.File(self.h5_path, "r") as h5:
            self.actions = h5[self.action_key][:]
            self.episode_ends = h5["episode_ends"][:]

        assert self.actions.shape[1] == 2, self.actions.shape

        # start index
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # train or test
        self.selected_episodes = self._make_episode_split()

        if self.normalize:

            if self.split == "train":

                selected_indices = []

                # ep_idx : episode idx
                for ep_idx in self.selected_episodes:
                    
                    # global indexing
                    ep_start = int(self.episode_starts[ep_idx])
                    ep_end = int(self.episode_ends[ep_idx])
                    
                    selected_indices.append(
                        np.arange(ep_start, ep_end)
                    )

                selected_indices = np.concatenate(selected_indices)

                train_actions = self.actions[selected_indices]

                action_min = np.min(train_actions, axis=0)
                action_max = np.max(train_actions, axis=0)

                action_scale = action_max - action_min

                self.action_stats = {
                    "min": action_min.astype(np.float32),
                    "max": action_max.astype(np.float32),
                    "scale": action_scale.astype(np.float32),
                }

                print("[Action normalization]")
                print("min:",action_min.astype(np.float32),
                    "max:",action_max.astype(np.float32),
                    "scale:",action_scale.astype(np.float32)
                    )

            else:
                # test는 train scale 사용
                self.action_stats = action_stats

            self.index_to_data, self.goals_index = self._build_index()

            print(f"[{self.split}] episodes: {len(self.selected_episodes)}")
            print(f"[{self.split}] samples : {len(self.index_to_data)}")


    def reconstruct_pose_rk4(self,linear_vels, angular_vels, dt=0.0333, initial_pose=(0.0, 0.0, 0.0)):
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



    def _make_episode_split(self):
        
        num_episodes = len(self.episode_ends)

        rng = np.random.default_rng(self.seed)
        episode_indices = np.arange(num_episodes)
        rng.shuffle(episode_indices)

    
        num_test = self.test_episode_num

        test_episodes = np.sort(episode_indices[:num_test])
        train_episodes = np.sort(episode_indices[num_test:])


        return train_episodes if self.split == "train" else test_episodes


    def _build_index(self):

        samples_index = []
        goals_index = []

        # per episode
        for ep_idx in self.selected_episodes:
            ep_start = int(self.episode_starts[ep_idx])
            ep_end = int(self.episode_ends[ep_idx])

            for goal_time in range(ep_start, ep_end):
                goals_index.append((ep_idx, goal_time))

            begin_time = ep_start + self.context_size * self.context_spacing

            end_time = ep_end - self.end_slack -1

            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, ep_end-curr_time-1)

                if max_goal_distance <= 0:
                    continue

                samples_index.append((ep_idx, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, ep_idx, curr_time, max_goal_dist):

        max_goal_cat = max_goal_dist // self.waypoint_spacing

        # max_goal_cat 범위 내 랜덤 부여
        if self.negative_mining:
            goal_offset = np.random.randint(0, max_goal_cat + 1)

            if goal_offset == 0:
                neg_ep_idx, neg_goal_time = self._sample_negative()
                return neg_ep_idx, neg_goal_time, True
        else:
            goal_offset = np.random.randint(1, max_goal_cat + 1)

        goal_time = curr_time + int(goal_offset * self.waypoint_spacing)

        return ep_idx, goal_time, False

    def _sample_negative(self):
        # self.goals_index : [(ep_idx, goal_time),...]
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _ensure_h5_open(self):

        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
            self.images = self.h5[self.image_key]

    def _load_image(self, time):
        
        #img = self.images[time]  # (3, H, W), uint8

        self._ensure_h5_open()
        img = self.images[time]

        img = torch.as_tensor(img, dtype=torch.float32) / 255.0

        if self.image_size is not None:
            w, h = self.image_size
            img = TF.resize(img, [h, w])

        return img

    def _compute_actions(self, curr_time, ep_end):

        needed_len = self.len_traj_pred * self.waypoint_spacing

        start_index = curr_time
        end_index = curr_time + needed_len

        valid_end = min(end_index, ep_end)

        raw_actions = self.actions[start_index:valid_end]
        raw_actions = raw_actions.astype(np.float32)

        # episode 끝을 넘어가는 부족분은 0으로 padding
        pad_len = needed_len - len(raw_actions)

        if pad_len > 0:
            pad = np.zeros((pad_len, 2), dtype=np.float32)
            raw_actions = np.concatenate([raw_actions, pad], axis=0)


        if self.predict_velocity:

            actions = raw_actions[::self.waypoint_spacing]

            # goal position 계산 (normalize 전)
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

            actions = np.diff(
                waypoint_xy,
                axis=0,
            )

            # 마지막 waypoint 자체가 goal
            goal_pos = waypoint_xy[-1]

        actions = actions.astype(np.float32)
        goal_pos = goal_pos.astype(np.float32)

        if self.normalize:
            actions = (
                2.0
                * (actions - self.action_stats["min"])
                / self.action_stats["scale"]
                - 1.0
            )

        assert actions.shape == (self.len_traj_pred, 2)

        return actions, goal_pos


    def __len__(self):
        return len(self.index_to_data)

    def __getitem__(self, i):
        ep_idx, curr_time, max_goal_dist = self.index_to_data[i]

        ep_end = int(self.episode_ends[ep_idx])

        goal_ep_idx, goal_time, goal_is_negative = self._sample_goal(ep_idx,curr_time,max_goal_dist)

        context_times = list(
            range(
                curr_time - self.context_size * self.context_spacing,
                curr_time + 1,
                self.context_spacing,
            )
        )

        obs_image = torch.cat(
            [self._load_image(t) for t in context_times],
            dim=0,
        )

        # goal time -> goal image
        goal_image = self._load_image(goal_time)

        actions, goal_pos = self._compute_actions(curr_time,ep_end)
        actions_torch = torch.as_tensor(
            actions,
            dtype=torch.float32,
        )

        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing

        # (Batch size 고려)
        # True → action loss 계산
        # False → action loss 무시
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
            # added
            torch.as_tensor(ep_idx, dtype=torch.int64),
            torch.as_tensor(curr_time, dtype=torch.int64),
        )

    def close(self):
        if hasattr(self, "h5") and self.h5 is not None:
            self.h5.close()



"""======================== For Test Data (non-seen data) ========================"""

class ViNT_H5_Action_Dataset_Test(Dataset):
    def __init__(
        self,
        h5_path: str,
        dataset_index: int = 0,
        image_key: str = "image_bottom",
        action_key: str = "encoder",
        image_size: Tuple[int, int] = (320, 240),  # (W, H)
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
        action_stats = None,
        predict_velocity = True,
        use_global_goal_for_test=False
    ):
        self.h5_path = h5_path
        self.dataset_index = dataset_index
        self.image_key = image_key
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
        self.context_size = context_size
        self.context_type = context_type
        self.end_slack = end_slack

        self.normalize = normalize
        self.action_stats = action_stats

        assert self.context_type == "temporal"
        assert self.split in ["train", "test"]

        self.num_action_params = 2
        self.predict_velocity = predict_velocity
        self.use_global_goal_for_test = use_global_goal_for_test

        """
        # === postech-episode === #
        [Dataset] encoder | shape=(186034, 2) | dtype=float32
        [Dataset] episode_ends | shape=(148,) | dtype=int64
        [Dataset] image_bottom | shape=(186034, 3, 240, 320) | dtype=uint8
        [Dataset] image_top | shape=(186034, 3, 240, 320) | dtype=uint8
        """

        self.h5 = None
        self.images = None

        with h5py.File(self.h5_path, "r") as h5:
            self.actions = h5[self.action_key][:]
            self.episode_ends = h5["episode_ends"][:]

        assert self.actions.shape[1] == 2, self.actions.shape

        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # train or test
        self.selected_episodes = self._make_episode_split()

        if self.normalize:

            if self.split == "train":

                selected_indices = []

                
                for ep_idx in self.selected_episodes:
                    ep_start = int(self.episode_starts[ep_idx])
                    ep_end = int(self.episode_ends[ep_idx])

                    selected_indices.append(
                        np.arange(ep_start, ep_end)
                    )

                selected_indices = np.concatenate(selected_indices)

                train_actions = self.actions[selected_indices]

                action_min = np.min(train_actions, axis=0)
                action_max = np.max(train_actions, axis=0)

                action_scale = action_max - action_min

                self.action_stats = {
                    "min": action_min.astype(np.float32),
                    "max": action_max.astype(np.float32),
                    "scale": action_scale.astype(np.float32),
                }

                print("[Action normalization]")
                print("min:",action_min.astype(np.float32),
                    "max:",action_max.astype(np.float32),
                    "scale:",action_scale.astype(np.float32)
                    )

            else:
                # test는 train scale 사용
                self.action_stats = action_stats


            self.index_to_data, self.goals_index = self._build_index()


            print(f"[{self.split}] episodes: {len(self.selected_episodes)}")
            print(f"[{self.split}] samples : {len(self.index_to_data)}")


    def reconstruct_pose_rk4(self,linear_vels, angular_vels, dt=0.0333, initial_pose=(0.0, 0.0, 0.0)):
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


    def _make_episode_split(self):
        
        num_episodes = len(self.episode_ends)

        episode_indices = np.arange(num_episodes)
        num_test = self.test_episode_num

        test_episodes = np.array([episode_indices[int(num_test - 1)]])
        train_episodes = np.sort(episode_indices)
        
        return train_episodes if self.split == "train" else test_episodes

    def _build_index(self):

        samples_index = []
        goals_index = []

        # per episode
        for ep_idx in self.selected_episodes:
            ep_start = int(self.episode_starts[ep_idx])
            ep_end = int(self.episode_ends[ep_idx])

            for goal_time in range(ep_start, ep_end):
                goals_index.append((ep_idx, goal_time))

            begin_time = ep_start + self.context_size * self.context_spacing
            end_time = ep_end - self.end_slack -1


            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, ep_end-curr_time-1)

                if max_goal_distance <= 0:
                    continue

                samples_index.append((ep_idx, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, ep_idx, curr_time, max_goal_dist):

        max_goal_cat = max_goal_dist // self.waypoint_spacing

        # max_goal_cat 범위 내 랜덤 부여
        goal_offset = np.random.randint(0, max_goal_cat + 1)

        if self.negative_mining and goal_offset == 0:
            neg_ep_idx, neg_goal_time = self._sample_negative()
            return neg_ep_idx, neg_goal_time, True

        goal_offset = max(goal_offset, 1)
        goal_time = curr_time + int(goal_offset * self.waypoint_spacing)

        return ep_idx, goal_time, False

    def _sample_negative(self):
        # self.goals_index : [(ep_idx, goal_time),...]
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _ensure_h5_open(self):

        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
            self.images = self.h5[self.image_key]

    def _load_image(self, time):
        
        #img = self.images[time]  # (3, H, W), uint8

        self._ensure_h5_open()
        img = self.images[time]

        img = torch.as_tensor(img, dtype=torch.float32) / 255.0

        if self.image_size is not None:
            w, h = self.image_size
            img = TF.resize(img, [h, w])

        return img

    def _compute_actions(self, curr_time, ep_end):

        needed_len = self.len_traj_pred * self.waypoint_spacing

        start_index = curr_time
        end_index = curr_time + needed_len

        valid_end = min(end_index, ep_end)

        raw_actions = self.actions[start_index:valid_end]
        raw_actions = raw_actions.astype(np.float32)

        # episode 끝을 넘어가는 부족분은 0으로 padding
        pad_len = needed_len - len(raw_actions)

        if pad_len > 0:
            pad = np.zeros((pad_len, 2), dtype=np.float32)
            raw_actions = np.concatenate([raw_actions, pad], axis=0)

        if self.predict_velocity:

            actions = raw_actions[::self.waypoint_spacing]

            # goal position 계산 (정규화 전)
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

            actions = np.diff(
                waypoint_xy,
                axis=0,
            )

            # 마지막 waypoint 자체가 goal
            goal_pos = waypoint_xy[-1]

        actions = actions.astype(np.float32)
        goal_pos = goal_pos.astype(np.float32)

        if self.normalize:
            actions = (
                2.0
                * (actions - self.action_stats["min"])
                / self.action_stats["scale"]
                - 1.0
            )

        assert actions.shape == (self.len_traj_pred, 2)

        return actions, goal_pos


    def __len__(self):
        return len(self.index_to_data)

    def __getitem__(self, i):
        ep_idx, curr_time, max_goal_dist = self.index_to_data[i]

        ep_end = int(self.episode_ends[ep_idx])

        if self.use_global_goal_for_test:
            goal_time = ep_end - 1
            goal_ep_idx = ep_idx
            goal_is_negative = False

        elif self.split == "test":

            # topology-map style fixed interval goal
            interval = 100  # 원하는 간격 step

            ep_start = int(self.episode_starts[ep_idx])
            local_t = curr_time - ep_start

            goal_local_t = ((local_t // interval) + 1) * interval
            goal_time = min(ep_start + goal_local_t, ep_end - 1)

            goal_ep_idx = ep_idx
            goal_is_negative = False

        else:
            goal_ep_idx, goal_time, goal_is_negative = self._sample_goal(
                ep_idx, curr_time, max_goal_dist
            )

        context_times = list(
            range(
                curr_time - self.context_size * self.context_spacing,
                curr_time + 1,
                self.context_spacing,
            )
        )

        obs_image = torch.cat(
            [self._load_image(t) for t in context_times],
            dim=0,
        )

        # goal time -> goal image
        goal_image = self._load_image(goal_time)
        actions, goal_pos = self._compute_actions(curr_time, ep_end)
        
        actions_torch = torch.as_tensor(
            actions,
            dtype=torch.float32,
        )

        if self.use_global_goal_for_test:
            remaining_dist = (goal_time - curr_time) // self.waypoint_spacing

            # 멀리 있으면 max_dist_cat으로 고정,
            # 끝에 가까워지면 99, 98, ... 이런 식으로 감소
            distance = min(self.max_dist_cat, remaining_dist)

        elif goal_is_negative:
            distance = self.max_dist_cat

        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing

        # (Batch size 고려)
        # True → action loss 계산
        # False → action loss 무시
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
            # added
            torch.as_tensor(ep_idx, dtype=torch.int64),
            torch.as_tensor(curr_time, dtype=torch.int64),
        )

    def close(self):
        if hasattr(self, "h5") and self.h5 is not None:
            self.h5.close()
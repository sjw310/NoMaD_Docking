import os
import argparse
import time
import pdb

import torch
import torch.nn as nn


class NoMaD(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(NoMaD, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs.get("input_goal_mask", None),
                encoder_hist=kwargs.get("encoder_hist", None),
                imu_hist=kwargs.get("imu_hist", None),
                lidar_hist=kwargs.get("lidar_hist", None),
            )
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class NoMaD_pose(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       pose_pred_net):
        super(NoMaD_pose, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.pose_pred_net = pose_pred_net
    
    def forward(self, func_name, **kwargs):
        if func_name == "vision_encoder" :
            output = self.vision_encoder(
                kwargs["obs_img"],
                kwargs["goal_img"],
                input_goal_mask=kwargs.get("input_goal_mask", None),
                encoder_hist=kwargs.get("encoder_hist", None),
                imu_hist=kwargs.get("imu_hist", None),
                lidar_hist=kwargs.get("lidar_hist", None),
            )

        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "pose_pred_net":
            output = self.pose_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output



class PoseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(PoseNetwork, self).__init__()

        self.embedding_dim = embedding_dim

        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim // 16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 16, 3),   # x, y, theta
        )

    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output

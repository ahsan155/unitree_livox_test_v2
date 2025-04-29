#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Some codes are modified from the OpenPCDet.
"""

import os
import glob
import datetime
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from livoxdetection.models.ld_base_v1 import LD_base
import copy
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point
#import sensor_msgs.point_cloud2 as pcl2
import sensor_msgs_py.point_cloud2 as pcl2


from vis_ros_update_2 import ROS_MODULE

last_box_num = 0
last_gtbox_num = 0

def mask_points_out_of_range(pc, pc_range):
    pc_range = np.array(pc_range)
    pc_range[3:6] -= 0.01  # np -> cuda .999999 = 1.0
    mask_x = (pc[:, 0] > pc_range[0]) & (pc[:, 0] < pc_range[3])
    mask_y = (pc[:, 1] > pc_range[1]) & (pc[:, 1] < pc_range[4])
    mask_z = (pc[:, 2] > pc_range[2]) & (pc[:, 2] < pc_range[5])
    mask = mask_x & mask_y & mask_z
    pc = pc[mask]
    return pc

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d.numpy() if is_numpy else corners3d

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pt', type=str, default=None, help='checkpoint to start from')
    args = parser.parse_args()
    return args

class RosDemo(Node):
    def __init__(self, model, args=None):
        super().__init__('livox_detection_node')
        self.args = args
        self.model = model
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.offset_angle = 0
        self.offset_ground = 1.3 # 1.8, 1.5 works well!
        self.point_cloud_range = [0, -44.8, -2, 224, 44.8, 4]
        self.subscription = self.create_subscription(
            PointCloud2,
            '/kitti/point_cloud',
            self.online_inference,
            10
        )

    def receive_from_ros(self, msg):
        points_list = pcl2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        #points_list = np.array(points).reshape(-1,4)
        #points_list = np.array(points_list, dtype=np.float32)


        # preprocess 
        points_list[:, 2] += points_list[:, 0] * np.tan(self.offset_angle / 180. * np.pi) + self.offset_ground
        rviz_points = copy.deepcopy(points_list)
        points_list = mask_points_out_of_range(points_list, self.point_cloud_range)

        input_dict = {
            'points': points_list,
            'points_rviz': rviz_points
        }
        data_dict = input_dict
        return data_dict
    
    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            if key in ['points']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
        ret['batch_size'] = batch_size
        return ret

    def online_inference(self, msg):
        data_dict = self.receive_from_ros(msg)
        data_infer = RosDemo.collate_batch([data_dict])
        RosDemo.load_data_to_gpu(data_infer)
        
        self.model.eval()
        with torch.no_grad(): 
            torch.cuda.synchronize()
            self.starter.record()
            pred_dicts = self.model.forward(data_infer)
            self.ender.record()
            torch.cuda.synchronize()
            curr_latency = self.starter.elapsed_time(self.ender)
        
        data_infer, pred_dicts = ROS_MODULE.gpu2cpu(data_infer, pred_dicts)
        global last_box_num
        last_box_num, _ = ros_vis.ros_print(data_dict['points_rviz'][:, 0:4], pred_dicts=pred_dicts, last_box_num=last_box_num)

def main(args=None):

    args = parse_config()
    model = LD_base()
    checkpoint = torch.load(args.pt, map_location=torch.device('cuda'), weights_only=False)  
    model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['model_state_dict'].items()})
    model.cuda()
    demo_ros = RosDemo(model, args)
    rclpy.spin(demo_ros)
    demo_ros.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    rclpy.init(args=None)
    ros_vis = ROS_MODULE()
    main()
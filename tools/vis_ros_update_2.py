import glob
import time
import copy
import argparse
from pathlib import Path

# numpy and torch
import torch
import numpy as np
from torch.utils.data import DataLoader

# rospy
import std_msgs.msg
from geometry_msgs.msg import Point
#import sensor_msgs.point_cloud2 as pcl2
import sensor_msgs_py.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray 
import rclpy
from rclpy.node import Node  # Import Node class for ROS2
from rclpy.duration import Duration
import matplotlib.pyplot as plt

from trajectory_pred.trajectory_prediction import predict_trajectory
from util import SimpleTracker, PredictionWorker


import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
import gin
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params
from human_scene_transformer.pedestrians import dataset_params
from human_scene_transformer.pedestrians import input_fn
import numpy as np
import tensorflow as tf
import tqdm

# ros marker
gtbox_array = MarkerArray()
marker_array = MarkerArray()
marker_array_text = MarkerArray()
prediction_marker_array = MarkerArray()


"""
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
"""

lines = [[0, 1], [1, 2], [2, 3], [3, 0], 
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]]

color_maps = {'car': [0, 1, 1], 'Car' : [0, 1, 1], 'truck': [0, 1, 1], 'Vehicle': [0, 1, 1], 'construction_vehicle': [0, 1, 1], 'bus': [0, 1, 1], 'trailer': [0, 1, 1],
        'motorcycle': [0, 1, 0], 'bicycle': [0, 1, 0], 'Cyclist': [0, 1, 0],
        'Pedestrian': [1, 1, 0], 'pedestrian': [1, 1, 0], 
        'barrier' : [1, 1, 1], 'traffic_cone': [1, 1, 1]}

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
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
    """
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d




class ROS_MODULE(Node):

    def __init__(self):
        super().__init__('vis_ros')
        self.class_names = ['Vehicle', 'Pedestrian', 'Cyclist'] 
        self.node = Node  # Store the node instance for later use (e.g., clock access)
        self.tracker = SimpleTracker(max_disappeared=5, max_distance=0.4)  # Initialize tracker 0.2

        # Create publishers using the provided node
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/pointcloud', 10)
        self.gtbox_array_pub = self.create_publisher(MarkerArray, '/detect_gtbox', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/detect_box3d', 10)
        self.marker_text_pub = self.create_publisher(MarkerArray, '/text_det', 10)

        # New publisher for predicted trajectories
        self.prediction_pub = self.create_publisher(MarkerArray, '/predicted_traj', 10)

        self.frame_id_num = 0  # Track frame number
        self.start_frame_id = 780
        self.track_file = open('./tracking_data.txt', 'w')



        self.num_prediction = 0
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


        logging.set_verbosity(logging.ERROR)
        gin.parse_config_files_and_bindings(
            [os.path.join('/media/ahsan/T7/jrdb/checkpoints/pedestrians_eth', 'params', 'operative_config.gin')],
            None,
            skip_unknown=True)
        print('Actual gin config used:')
        print(gin.config_str())

        d_params = dataset_params.PedestriansDatasetParams(
            num_agents=None,
            eval_config='test'
            )

        self.dataset = input_fn.load_dataset(
            d_params,
            split='test',
            augment=False,
            shuffle=False,
            repeat=False,
            deterministic=True,
        )
        
        model_p = model_params.ModelParams()
        self.model = hst_model.HumanTrajectorySceneTransformer(model_p)
        _, _ = self.model(next(iter(self.dataset.batch(1))), training=False)

        checkpoint_path = '/media/ahsan/T7/jrdb/checkpoints/pedestrians_eth/ckpts/ckpt-20'
        checkpoint_mngr = tf.train.Checkpoint(model=self.model)
        checkpoint_mngr.restore(checkpoint_path).assert_existing_objects_matched()
        #status = checkpoint_mngr.restore(checkpoint_path)
        #status.expect_partial()       
        logging.info('Restored checkpoint: %s', checkpoint_path)

        self.prediction_worker = PredictionWorker(self.model)  # Initialize worker

        
        
    def __del__(self):
        # Close the file when the node is destroyed
        self.prediction_worker.stop()  # Cleanup on shutdown
        self.track_file.close()

    @staticmethod
    def gpu2cpu(data_dict, pred_dicts):
        data_dict['points'] = data_dict['points'].cpu().numpy()
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'].cpu().numpy()
        pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu().numpy()
        pred_dicts[0]['pred_scores'] = pred_dicts[0]['pred_scores'].cpu().numpy()
        pred_dicts[0]['pred_labels'] = pred_dicts[0]['pred_labels'].cpu().numpy()
        torch.cuda.empty_cache()
        return data_dict, pred_dicts
 
    def ros_print(self, pts, pred_dicts=None, last_box_num=None, gt_boxes=None, last_gtbox_num=None):
        def xyzr_to_pc2(pts, header):
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            points = [list(pt) for pt in pts]  # Convert NumPy array to list of lists
            msg = pc2.create_cloud(header, fields, points)
            return msg

        # ROS Header
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()  # Use node's clock
        header.frame_id = 'livox_frame'
        pointcloud_msg = xyzr_to_pc2(pts, header)
        self.pointcloud_pub.publish(pointcloud_msg)

                
        if pred_dicts is not None:

            # Update tracker with pred_boxes
            current_time = header.stamp.sec + header.stamp.nanosec * 1e-9
            pred_boxes = pred_dicts[0]['pred_boxes']
            pred_scores = pred_dicts[0]['pred_scores']
            pred_labels = pred_dicts[0]['pred_labels']

            tracked_objects = self.tracker.update(pred_boxes, pred_scores, pred_labels, current_time)
            boxes = boxes_to_corners_3d(np.array([box for (_, box, _, _, _) in tracked_objects]))
                        
        
            marker_array.markers.clear()
            marker_array_text.markers.clear()
            prediction_marker_array.markers.clear()

            # Submit prediction tasks asynchronously
            track_ids = []
            trajectories = []
            
            for idx, (track_id, ob, score, label, trajectory) in enumerate(tracked_objects):
                # boxes
                marker = Marker()
                marker.header.frame_id = header.frame_id
                marker.header.stamp = header.stamp
                marker.id = track_id * 2
                marker.action = Marker.ADD
                marker.type = Marker.LINE_LIST
                marker.lifetime = Duration(seconds=0.1).to_msg()

                color = color_maps["pedestrian"] 
                marker.color.r = float(color[0])
                marker.color.g = float(color[1])
                marker.color.b = float(color[2])
                marker.color.a = 0.8
                marker.scale.x = 0.05
                
                marker.points = []
                for line in lines:
                    ptu = boxes[idx][line[0]]
                    ptv = boxes[idx][line[1]]
                    marker.points.append(Point(x=float(ptu[0]), y=float(ptu[1]), z=float(ptu[2])))
                    marker.points.append(Point(x=float(ptv[0]), y=float(ptv[1]), z=float(ptv[2])))

                if (boxes[idx][0][0] + boxes[idx][2][0]) / 2 > 2.3:
                    continue
                marker_array.markers.append(marker)
                


                if len(trajectory) >= 8:
                    track_ids.append(track_id)
                    trajectories.append(trajectory)
                    print("adding trajectory...", self.frame_id_num)
                else:
                    print("not adding trajectory...", self.frame_id_num)


                '''
                historical_pos = None
                trajectory = np.array(trajectory) # historical positions
                positions = np.zeros((1, 1, 20, 2), dtype=np.float32) # empty array to be filled
                pred = None

                
                if trajectory.shape[0] >= 8:
                    historical_pos = trajectory[-8:, 1:3] # taking last eight historcal position
                    positions[:, :, :8, :] = historical_pos # assigning last eight position from historical data
                    positions[:, :, 8:, :] = 0.0

                    
                    #last_vel = historical_pos[-1] - historical_pos[-2]
                    #for t in range(8,20):
                    #    positions[0,0,t] = positions[0,0,t-1] + last_vel
                    #positions = tf.convert_to_tensor(positions)
                    
                    
                    for input_batch in tqdm.tqdm(self.dataset.batch(1)):
                        break
                    
                    input_batch['agents/position'] = positions
                    mask = np.zeros((1,1,20), dtype=bool)
                    mask[:,:, :8] = True
                    input_batch['agents/mask']     = tf.convert_to_tensor(mask)
                    input_batch['agents/position'] = tf.convert_to_tensor(positions)
                    
                    start = time.perf_counter()
                    full_pred, _ = self.model(input_batch, training=False)
                    end = time.perf_counter()
                    elapsed_ms = (end - start) * 1000
                    print(f"Inference time: {elapsed_ms:.2f} ms")

                    ml_indices = tf.squeeze(
                        tf.math.argmax(full_pred['mixture_logits'], axis=-1)
                    )
                    pred = full_pred['agents/position'][..., ml_indices, :][:, :, 8:, :]
                    print("publishing trajectory...", self.frame_id_num)                    
                else:
                    print("not publishing trajectory...")
                '''
               
                
                # confidence
                markert = Marker()
                markert.header.frame_id = header.frame_id
                markert.header.stamp = header.stamp
                markert.id = track_id * 2 + 1
                markert.action = Marker.ADD
                markert.type = Marker.TEXT_VIEW_FACING
                markert.lifetime = Duration(seconds=0.1).to_msg()

                color = color_maps["pedestrian"] 

                markert.color.r = float(color[0])
                markert.color.g = float(color[1])
                markert.color.b = float(color[2])
                markert.color.a = float(1)
                markert.scale.z = float(0.15)
               
                markert.pose.orientation.w = 1.0
                
                markert.pose.position.x = (boxes[idx][0][0] + boxes[idx][2][0]) / 2 + 0.4
                markert.pose.position.y = (boxes[idx][0][1] + boxes[idx][2][1]) / 2 + 0.4
                markert.pose.position.z = (boxes[idx][0][2] + boxes[idx][4][2]) / 2 + 0.5
                markert.text = f"{self.class_names[label - 1]}:{track_id}:{score:.2f}"  # Use tracked score and label
                marker_array_text.markers.append(markert)

               
                
            # clear ros cache   
            if last_box_num > len(tracked_objects):
                for i in range(len(tracked_objects), last_box_num): # boxes.shape[0] replaced w len(tracked_objects)
                    marker = Marker()
                    marker.header.frame_id = header.frame_id
                    marker.header.stamp = header.stamp
                    marker.id = i * 2
                    marker.action = Marker.ADD
                    marker.type = Marker.LINE_LIST
                    marker.lifetime = Duration(seconds=0.01).to_msg()
                    marker.color.a = float(0)
                    marker_array.markers.append(marker)

                    markert = Marker()
                    markert.header.frame_id = header.frame_id
                    markert.header.stamp = header.stamp
                    markert.id = i * 2 + 1
                    markert.action = Marker.ADD
                    markert.type = Marker.TEXT_VIEW_FACING
                    markert.lifetime = Duration(seconds=0.01).to_msg()
                    markert.color.a = 0.0
                    marker_array_text.markers.append(markert)


            # publish
            print('length of markers', len(marker_array.markers), self.frame_id_num)
            self.marker_pub.publish(marker_array)
            self.marker_text_pub.publish(marker_array_text)

            
            if track_ids:
                print('xx', self.frame_id_num, len(trajectories), track_ids)
                self.prediction_worker.add_task(track_ids, trajectories)
            
            # Retrieve predictions (non-blocking)
            for idx, (track_id, ob, score, label, trajectory) in enumerate(tracked_objects):
                pred = self.prediction_worker.get_result(track_id)
                
                if pred is not None:
                    print(pred.shape)
                    pred_marker = Marker()
                    pred_marker.header.frame_id = header.frame_id
                    pred_marker.header.stamp = header.stamp
                    # Use a unique id offset (e.g., track_id + 1000)
                    pred_marker.id = track_id + 1000
                    pred_marker.action = Marker.ADD
                    pred_marker.type = Marker.LINE_STRIP
                    pred_marker.lifetime = Duration(seconds=0.1).to_msg()
                    pred_marker.scale.x = 0.03  # thickness of the line
                    # Set color (e.g., red for prediction)
                    pred_marker.color.r = 1.0
                    pred_marker.color.g = 0.0
                    pred_marker.color.b = 0.0
                    pred_marker.color.a = 0.8
                    pred_marker.points = []
                    
                    for p in pred:
                        pt = Point(x=float(p[0]), y=float(p[1]), z=0.0)
                        pred_marker.points.append(pt)
                    prediction_marker_array.markers.append(pred_marker)

            self.prediction_pub.publish(prediction_marker_array)
            

            self.frame_id_num += 1

        
        box_size = 0 if pred_dicts is None else boxes.shape[0]
        gtbox_size = 0 if gt_boxes is None else gt_boxes.shape[0]

        return box_size, gtbox_size
        # input()

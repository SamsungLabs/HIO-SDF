#!/usr/bin/env python3

"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Suveer Garg (suveer.garg@samsung.com)
Vasileios Vasilopoulos (vasileios.v@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

import os
import json
import numpy as np
import rospy
np.float = np.float64 # Hack for older ros_numpy version
import ros_numpy
import tf
import cv2
import open3d as o3d
import time
from sensor_msgs.msg import PointCloud2

class DatasetAdapterNodeReplicaCAD:
    '''
    Dataset Adapter Node for ReplicaCAD dataset
    '''
    def __init__(self, path_to_data = '/home/vasileiosv/data/ReplicaCAD/apt_3_nav'):
        rospy.init_node("DatasetAdapterNodeReplicaCAD")

        # Initialize frame ID
        self.frame_id = 0

        # Frequency and period for ROS Timer-based frame publishing
        self.freq = 10
        self.period = 1.0/self.freq

        # FPS to use for sensor data timestamps
        self.fps = 30

        # Names of used frames and topics
        self.map_frame = 'map'
        self.dataset_map_frame = 'replica_map'
        self.camera_frame = 'camera'
        self.lidar_frame = 'velodyne'
        self.pcd_topic = "/velodyne_points"
        
        # Pointcloud publisher
        self.pcd_publisher = rospy.Publisher(self.pcd_topic, PointCloud2, queue_size=1)

        # Frame publisher hooked on a timer
        self.frame_timer = rospy.Timer(rospy.Duration(self.period), self.publish_data)

        # Paths to data, depth and trajectory
        self.path_to_data = path_to_data
        self.path_to_depth = os.path.join(self.path_to_data, "results")
        self.traj = np.loadtxt(os.path.join(self.path_to_data, "traj.txt")).reshape(-1, 4 ,4)

        # Load camera intrinsics
        with open(os.path.join(self.path_to_data, 'replicaCAD_info.json')) as json_file:
            dataset_parameters = json.load(json_file)
        self.depth_scale = dataset_parameters['depth_scale']

        # Create Open3D pinhole camera based on intrinsics
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            dataset_parameters['camera']['w'],
            dataset_parameters['camera']['h'],
            dataset_parameters['camera']['fx'],
            dataset_parameters['camera']['fy'],
            dataset_parameters['camera']['cx'],
            dataset_parameters['camera']['cy'],
        )

        self.start_spin()
    
    def start_spin(self):
        try:
            rospy.loginfo("Spinning Node")
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("Could not spin dataset adapter node.")
            pass

    def publish_data(self, event = None):
        if self.frame_id >= self.traj.shape[0]:
            rospy.logwarn(f"Done with all frames")
            return
        
        rospy.loginfo(f" Publishing Frame : {self.frame_id}")

        # Unpack transform
        transform = self.traj[self.frame_id, :, :]

        # Load depth and create pcd
        image_path = os.path.join(self.path_to_depth, 'depth' + str(self.frame_id).zfill(6) + ".png")
        depth_image = o3d.io.read_image(image_path)
        o3d_depth = o3d.geometry.Image(depth_image)
        pcd = np.asarray(o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, self.o3d_intrinsics, depth_scale = self.depth_scale).points)   
        
        # Calcuate timestamp based on frame id and FPS 
        seq_time = self.frame_id/self.fps         
        t = rospy.Time.from_sec(seq_time)

        # Transform Broadcaster
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (0, 0, 0),
            tf.transformations.quaternion_from_euler(np.pi/2, -np.pi/2, 0),
            t,
            self.lidar_frame,
            self.camera_frame,
        )

        br.sendTransform(
            (transform[0,3], transform[1,3], transform[2,3]),
            tf.transformations.quaternion_from_matrix(transform),
            t,
            self.camera_frame,
            self.dataset_map_frame,
        )
        
        br.sendTransform(
            (0,0,0),
            tf.transformations.quaternion_from_euler(np.pi/2, 0, 0),
            t,
            self.dataset_map_frame,
            self.map_frame,
        )

        # Construct pointcloud
        pc_array = np.zeros(len(pcd), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])
        pc_array['x'] = pcd[:, 0]
        pc_array['y'] = pcd[:, 1]
        pc_array['z'] = pcd[:, 2]

        # Publish pointcloud
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp = t, frame_id = self.camera_frame)
        self.pcd_publisher.publish(pc_msg)

        # Increase frame ID
        self.frame_id += 1


class DatasetAdapterNodeScanNet:
    '''
    Dataset Adapter Node for ScanNet dataset
    '''
    def __init__(self, path_to_data = '/home/vasileiosv/data/scene0010'):
        rospy.init_node("DatasetAdapterNodeScanNet")

        # Initialize frame ID
        self.frame_id = 0

        # Frequency and period for ROS Timer-based frame publishing
        self.freq = 10
        self.period = 1.0/self.freq

        # FPS to use for sensor data timestamps
        self.fps = 30

        # Names of used frames and topics
        self.map_frame = 'map'
        self.sensor_frame = 'camera'
        self.pcd_topic = "/velodyne_points"

        # Pointcloud publisher
        self.pcd_publisher = rospy.Publisher(self.pcd_topic, PointCloud2, queue_size=1)

        # Frame publisher hooked on a timer
        self.frame_timer = rospy.Timer(rospy.Duration(self.period), self.publish_data)
        
        # Path to data and path to depth data
        self.path_to_data = path_to_data
        self.path_to_depth = os.path.join(self.path_to_data, "depth")

        # Load camera intrinsics
        depth_image = cv2.imread(os.path.join(self.path_to_depth, str(self.frame_id) + ".png"), -1)
        self.intrinsics = np.loadtxt(os.path.join(self.path_to_data, "intrinsic/intrinsic_depth.txt")).reshape(4,4)
        self.depth_scale= 1000.0
        
        # Create Open3D pinhole camera based on intrinsics
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_image.shape[0],
            depth_image.shape[1],
            self.intrinsics[0, 0],
            self.intrinsics[1, 1],
            self.intrinsics[0, 2],
            self.intrinsics[1, 2],
        )
        
        # Load sensor transforms
        self.transforms = np.loadtxt(os.path.join(self.path_to_data, "traj.txt")).reshape(-1,4,4)

        self.start_spin()

    def start_spin(self):
        try:
            rospy.loginfo("Spinning Node")
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("Could not spin dataset adapter node.")
            pass

    def publish_data(self, event = None):
        if self.frame_id >= self.transforms.shape[0]:
            rospy.logwarn(f"Done with all frames")
            return
        
        rospy.loginfo(f" Publishing Frame: {self.frame_id}")
        
        # Load depth and create pcd
        image_path = os.path.join(self.path_to_depth, str(self.frame_id) + ".png")
        depth_image = o3d.io.read_image(image_path)
        o3d_depth = o3d.geometry.Image(depth_image)
        pcd = np.asarray(o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, self.o3d_intrinsics, depth_scale = self.depth_scale).points)
                
        # Load transform
        transform = self.transforms[self.frame_id]
        
        # Calcuate timestamp based on frame id and FPS 
        seq_time = self.frame_id/self.fps       
        t = rospy.Time.from_sec(seq_time) 
        
        # Transform Broadcaster         
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (transform[0,3], transform[1,3], transform[2,3]),
            tf.transformations.quaternion_from_matrix(transform),
            t,
            self.sensor_frame,
            self.map_frame,
        )

        # Construct pointcloud
        pc_array = np.zeros(len(pcd), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
        ])
        pc_array['x'] = pcd[:, 0]
        pc_array['y'] = pcd[:, 1]
        pc_array['z'] = pcd[:, 2]

        # Publish pointcloud
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp = t, frame_id = self.sensor_frame)
        self.pcd_publisher.publish(pc_msg)

        # Increase frame ID
        self.frame_id += 1

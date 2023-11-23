"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Vasileios Vasilopoulos (vasileios.v@samsung.com; vasilis.vasilop@gmail.com)
Suveer Garg (suveer.garg@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

# General imports
import os
import time
import numpy as np
import torch
import copy
import json
import trimesh
import zmq
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ROS imports
import rospy
import tf
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
np.float = np.float64
import ros_numpy
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array

# Voxblox imports
from voxblox_msgs.msg import DiscreteSDF
import struct

# Brute-force SDF
from pointcloud_sdf.pointcloud_sdf import PointCloud_SDF

# Global SDF trainer
from .global_sdf_training import GlobalSDFTrainer

# Other imports
from .visualization import write_mesh, write_slices
from .utils import load_gt_sdf, eval_sdf

class HIOSDFNode:
    def __init__(self):
        rospy.init_node('hio_sdf_node')

        # Device and type
        self.device = rospy.get_param("~device")
        self.dtype = torch.float

        # Initialize workspace extents
        self.extents = np.array([[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]])

        # Main options
        self.use_local_sdf_data = rospy.get_param("~use_local_sdf_data")
        self.do_eval = rospy.get_param("~do_eval")
        self.save_mesh = rospy.get_param("~save_mesh")
        self.save_slices = rospy.get_param("~save_slices")
        self.publish_local_data = rospy.get_param("~publish_local_data")
        self.publish_sdf_data = rospy.get_param("~publish_sdf_data")
        self.publish_sdf_slices = rospy.get_param("~publish_sdf_slices")
        self.stream_weights_over_zmq = rospy.get_param("~stream_weights_over_zmq")

        # Parameters for Voxfield sampling
        self.num_voxfield_samples_surface = rospy.get_param("~num_voxfield_samples_surface")
        self.num_voxfield_samples_freespace = rospy.get_param("~num_voxfield_samples_freespace")

        # Parameters for local SDF sampling
        self.num_local_pcds = rospy.get_param("~num_local_pcds")
        self.num_local_pcd_points = rospy.get_param("~num_local_pcd_points")
        self.max_raycasting_distance = rospy.get_param("~max_raycasting_distance")
        self.truncation_distance = rospy.get_param("~truncation_distance")
        self.num_raycasting_points = rospy.get_param("~num_raycasting_points")
        self.num_local_samples = rospy.get_param("~num_local_samples")

        # Initialize brute-force SDF
        self.pointcloud_sdf = PointCloud_SDF(np.random.rand(100000,3), dtype = self.dtype, device = self.device)

        # Training parameters
        self.learning_rate = rospy.get_param("~learning_rate")
        self.weight_decay = rospy.get_param("~weight_decay")
        self.sdf_loss_weight = rospy.get_param("~sdf_loss_weight")
        self.eikonal_loss_weight = rospy.get_param("~eikonal_loss_weight")
        self.warm_start_iters = rospy.get_param("~warm_start_iters")
        self.epochs_warm_start = rospy.get_param("~epochs_warm_start")
        self.epochs_nominal = rospy.get_param("~epochs_nominal")

        # Initialize global SDF trainer
        self.global_sdf_trainer = GlobalSDFTrainer(
            learning_rate = self.learning_rate,
            weight_decay = self.weight_decay,
            sdf_loss_weight = self.sdf_loss_weight,
            eikonal_loss_weight = self.eikonal_loss_weight,
            device = self.device,
        )

        # Initialize buffers
        self.esdf_layer_msg = None
        self.trajectory = []
        self.pointcloud_buffer = []

        # Initialize number of iterations
        self.iters = 0

        # Publish options
        self.sdf_slice_z = rospy.get_param("~sdf_slice_z")
        self.sdf_visualization_resolution = rospy.get_param("~sdf_visualization_resolution")
        self.sdf_visualization_period = 10.0
        self.mesh_resource_file = ""

        # Voxfield parameters
        self.kEpsilon = rospy.get_param("~kEpsilon")
        self.voxel_size = rospy.get_param("~voxel_size")
        self.voxel_per_side = rospy.get_param("~voxel_per_side")
        self.block_size = self.voxel_per_side * self.voxel_size
        self.kNumDataPacketsPerVoxel = rospy.get_param("~kNumDataPacketsPerVoxel")

        # Preparing Voxfield meshgrid
        nx, ny, nz = np.meshgrid(
            np.linspace(0, self.voxel_per_side-1, self.voxel_per_side),
            np.linspace(0, self.voxel_per_side-1, self.voxel_per_side),
            np.linspace(0, self.voxel_per_side-1, self.voxel_per_side)
        )
        self.meshgrid = np.zeros((self.voxel_per_side**3, 3))
        self.meshgrid[:, 0] = np.transpose(nx, (2,0,1)).flatten() 
        self.meshgrid[:, 1] = np.transpose(ny, (2,0,1)).flatten()
        self.meshgrid[:, 2] = np.transpose(nz, (2,0,1)).flatten()
        self.meshgrid = (self.meshgrid + 0.5) * self.voxel_size

        # Output path
        self.output_path = rospy.get_param("~output_path")

        # Regular grid for evaluation
        self.grid_resolution = rospy.get_param("~grid_resolution")
        regular_grid_x_min = rospy.get_param("~regular_grid_x_min")
        regular_grid_x_max = rospy.get_param("~regular_grid_x_max")
        regular_grid_y_min = rospy.get_param("~regular_grid_y_min")
        regular_grid_y_max = rospy.get_param("~regular_grid_y_max")
        regular_grid_z_min = rospy.get_param("~regular_grid_z_min")
        regular_grid_z_max = rospy.get_param("~regular_grid_z_max")
        regular_grid_x = torch.arange(regular_grid_x_min, regular_grid_x_max + self.grid_resolution, step = self.grid_resolution)
        regular_grid_y = torch.arange(regular_grid_y_min, regular_grid_y_max + self.grid_resolution, step = self.grid_resolution)
        regular_grid_z = torch.arange(regular_grid_z_min, regular_grid_z_max + self.grid_resolution, step = self.grid_resolution)
        regular_grid = torch.meshgrid(regular_grid_x, regular_grid_y, regular_grid_z)
        self.regular_grid_3d = torch.cat(
            (
                regular_grid[0][..., None],
                regular_grid[1][..., None],
                regular_grid[2][..., None],
            ),
            dim = 3
        ).cpu().numpy()
        self.regular_grid_transform = np.array([[regular_grid_x_min, regular_grid_y_min, regular_grid_z_min]])
        self.regular_grid_observed_voxels = np.zeros(self.regular_grid_3d.shape[:-1]).astype(bool)

        # Preparing ground truth interpolator for evaluation
        if self.do_eval:
            gt_sdf_file = rospy.get_param("~gt_sdf_file")
            sdf_transf_file = rospy.get_param("~sdf_transf_file")
            self.eval_point_samples = rospy.get_param("~eval_point_samples")
            self.dataset_format = rospy.get_param("~dataset_format")
            self.scene = rospy.get_param("~scene")
            self.sdf_loss = []
            self.eikonal_loss = []
            self.cosdist_loss = []
            self.local_sdf_loss = []
            self.local_eikonal_loss = []
            self.local_cosdist_loss = []
            self.sequence_time = []
            self.gt_sdf_interp, self.sdf_grid, self.grid_3d = load_gt_sdf(gt_sdf_file, sdf_transf_file, dataset_format = self.dataset_format,)
        
        # ZMQ parameters
        if self.stream_weights_over_zmq:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind("tcp://*:5555")

        # Initialize tf2 listener
        self.tf_listener = tf.TransformListener()
        self.map_frame = rospy.get_param("~map_frame")
        
        # Initialize subscribers and publishers
        self.init_subscribers()
        self.init_publishers()

        # Spin
        rospy.loginfo("Finished initializing HIO-SDF node.")
        rospy.on_shutdown(self.plot)
        self.start_spin()
        
    def init_subscribers(self):
        rospy.Subscriber(rospy.get_param("~coarse_sdf_topic"), DiscreteSDF,  self.esdf_callback, queue_size = 1, buff_size = 2**24)
        rospy.Subscriber(rospy.get_param("~pointcloud_topic"), PointCloud2,  self.pointcloud_callback, queue_size = 1, buff_size = 2**24)
        rospy.loginfo('Subscribers initialized')

    def init_publishers(self):
        self.data_slice_publisher = rospy.Publisher('/data_slice', PointCloud2, queue_size = 1)
        self.sdf_slice_publisher = rospy.Publisher('/sdf_slice', PointCloud2, queue_size = 1)
        self.local_points_publisher = rospy.Publisher('/local_points', PointCloud2, queue_size = 1)
        self.mesh_publisher = rospy.Publisher('/global_sdf_mesh', Marker, queue_size = 1)

        # Create a ROS Timer for sending SDF weights over ZMQ
        if self.stream_weights_over_zmq:
            self.zmq_timer = rospy.Timer(rospy.Duration(1.0/30.0), self.zmq_callback)
        
        # Frame publisher hooked on a timer
        if self.publish_sdf_slices:
            self.sdf_slice_timer = rospy.Timer(rospy.Duration(0.1), self.publish_slices)

    def start_spin(self):
        try:
            rospy.loginfo("Spinning Node")
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("Could not spin HIO-SDF node.")
            pass
    
    def frame_transformation_matrix(self, target_frame, source_frame, timestamp=rospy.Time(0)):
        # Query the TF tree
        try:
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, timestamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(e)
            return None
        
        # Construct 4x4 transformation matrix
        position = np.array(trans)
        orientation = tf.transformations.quaternion_matrix(rot)
        transformation = np.eye(4)
        transformation[:3, :3] = orientation[:3, :3]
        transformation[:3, 3] = position

        return transformation
    
    # Definition of ZMQ publisher callback
    def zmq_callback(self, event = None):
        # Publish weights over ZMQ
        send_data={}
        data = self.socket.recv_pyobj()
        srt_time = time.time()
        send_data["model_state_dict"] = self.global_sdf_trainer.model.state_dict()
        self.socket.send_pyobj(send_data)
        rospy.loginfo(f"Sent SDF weights in {time.time()-srt_time}")

        return
    
    # Definition of Voxfield callback
    def esdf_callback(self, esdf_layer_msg):
        self.esdf_layer_msg = esdf_layer_msg
        return
        
    # Definition of main callback
    def pointcloud_callback(self, pointcloud_msg):
        # Check if there exists Voxfield data
        if self.esdf_layer_msg is None:
            rospy.logerr("No Voxfield data received yet - returning")
            return
        
        # Store the time
        start_total = time.time()
        msg_time = pointcloud_msg.header.stamp

        # Find 4x4 transformation of sensor frame in map frame
        self.sensor_transformation = self.frame_transformation_matrix(self.map_frame, pointcloud_msg.header.frame_id, pointcloud_msg.header.stamp)
        if self.sensor_transformation is None:
            rospy.logerr(f"Could not find transform from {pointcloud_msg.header.frame_id} to {self.map_frame}")
            return
        
        

        ################## LOCAL SDF DATA ##################
        # Store time
        start_local = time.time()

        # Get latest point cloud
        pcd_sensor_frame = pointcloud2_to_xyz_array(pointcloud_msg)

        # Transform point cloud to the map frame
        pcd = np.hstack((pcd_sensor_frame, np.ones((pcd_sensor_frame.shape[0],1))))
        pcd = np.transpose(
            np.matmul(
                self.sensor_transformation,
                np.transpose(pcd)
            )
        )[:, :3]

        # Add point cloud to the buffer
        if len(self.pointcloud_buffer) >= self.num_local_pcds:
            self.pointcloud_buffer.pop(0)
        self.pointcloud_buffer.append(pcd)

        # Concatenate point clouds and randomly sample points
        local_pcd = np.concatenate(self.pointcloud_buffer, axis = 0)
        if local_pcd.shape[0] > self.num_local_pcd_points:
            idx = np.random.choice(local_pcd.shape[0], size = self.num_local_pcd_points, replace = False)
            local_pcd = local_pcd[idx, :]

        # Update pointcloud SDF with current point cloud
        self.pointcloud_sdf.update_pcd(local_pcd)

        # Point cloud rays and distances
        local_pcd_pts_to_use = self.num_local_samples // self.num_raycasting_points
        idx = np.random.choice(local_pcd.shape[0], size = local_pcd_pts_to_use, replace = False)
        local_pcd_for_raycasting = local_pcd[idx]
        pcd_rays = (local_pcd_for_raycasting - self.sensor_transformation[:3, 3]) / np.maximum(np.linalg.norm(local_pcd_for_raycasting - self.sensor_transformation[:3, 3], axis = 1, keepdims = True), 1e-12)
        pcd_distances = np.linalg.norm(local_pcd_for_raycasting - self.sensor_transformation[:3, 3], axis = 1)

        # Raycasting
        raycast_distance = min(np.max(pcd_distances), self.max_raycasting_distance)
        raycast_endpoints = self.sensor_transformation[:3, 3] + (raycast_distance + self.truncation_distance) * pcd_rays
        local_points = np.linspace(self.sensor_transformation[:3, 3], raycast_endpoints, self.num_raycasting_points)

        # Find signs of raycasted points
        local_points_signs = np.ones((local_points.shape[0],local_points.shape[1]))
        cond_sign = (np.linalg.norm(local_points - self.sensor_transformation[:3, 3], axis = -1) > np.expand_dims(pcd_distances, axis = 0)).astype(bool)
        local_points_signs[cond_sign] = -1.0

        # Truncate at a fixed distance behind the point cloud
        cond_trunc = (np.linalg.norm(local_points - self.sensor_transformation[:3, 3], axis = -1) <= np.expand_dims(pcd_distances + self.truncation_distance, axis = 0)).astype(bool)
        local_points = local_points[cond_trunc]
        local_points_signs = local_points_signs[cond_trunc]
        local_points = local_points.reshape(-1, 3)
        local_points_signs = local_points_signs.reshape(-1, 1)

        # Construct local samples and signs
        local_points_tensor = torch.from_numpy(local_points).to(dtype = self.dtype, device = self.device)
        local_points_signs = torch.from_numpy(local_points_signs).to(dtype = self.dtype, device = self.device)

        # Pass local points through brute-force SDF
        local_sdf_values = self.pointcloud_sdf.forward(local_points_tensor, signs = local_points_signs)

        # Construct local SDF data
        if self.use_local_sdf_data:
            local_sdf_data = torch.cat((local_points_tensor, local_sdf_values), dim = 1)
            local_sdf_data = local_sdf_data[local_sdf_data[:, 3] < self.truncation_distance, :]

            # Publish local points
            if self.publish_local_data:
                self.publish_local_points(local_sdf_data[:, :3].detach().cpu().numpy(), msg_time)
        else:
            local_sdf_data = torch.tensor([], dtype = self.dtype, device = self.device)

        # Update regular grid
        local_points_idx = ((local_points - self.regular_grid_transform) / self.grid_resolution).astype(int)
        self.regular_grid_observed_voxels[local_points_idx[:, 0], local_points_idx[:, 1], local_points_idx[:, 2]] = True

        # Find total time for constructing local data
        time_local = time.time() - start_local
        ####################################################

        
        
        ################## VOXFIELD DATA ##################
        # Store the time
        start_voxfield = time.time()

        # Get number of blocks and voxels in the message
        indices = np.asarray(self.esdf_layer_msg.indices).reshape(-1, 3)
        num_blocks = indices.shape[0]
        num_voxels = num_blocks * self.voxel_per_side**3

        # Initialize SDF data: array where each row is (x,y,z,SDF)
        voxfield_sdf_data = np.zeros((num_voxels, 4))
    
        # Find origin of each block
        origin = indices * self.block_size

        # Get SDF data and convert to float
        sdf_data_unit = self.esdf_layer_msg.data[::self.kNumDataPacketsPerVoxel]
        flag_data_unit = self.esdf_layer_msg.data[1::self.kNumDataPacketsPerVoxel]
        sdf = list(struct.unpack("<%df" % len(sdf_data_unit), struct.pack("<%dI" % len(sdf_data_unit), *sdf_data_unit)))
        observed = np.bitwise_and(flag_data_unit, True).astype(bool)

        # Find coordinates of voxels
        voxel_centers = np.expand_dims(self.meshgrid, 0) + np.expand_dims(origin, 1)
        voxel_centers = voxel_centers.reshape(-1, 3)
        voxfield_sdf_data[:, :3] = voxel_centers

        # Assign SDF values
        voxfield_sdf_data[:, 3] = sdf
        
        # Use only the observed data
        voxfield_sdf_data = voxfield_sdf_data[observed]

        # Find points close to the surface
        close_to_surface = (abs(voxfield_sdf_data[:, 3]) < 0.05).astype(bool)
        sdf_data_surface = voxfield_sdf_data[close_to_surface, :]
        sdf_data_freespace = voxfield_sdf_data[~close_to_surface, :]

        # Downsample if necessary
        if sdf_data_surface.shape[0] > self.num_voxfield_samples_surface:
            idx = np.random.choice(sdf_data_surface.shape[0], size = self.num_voxfield_samples_surface, replace = False)
            sdf_data_surface = sdf_data_surface[idx, :]
        sdf_data_surface = torch.from_numpy(sdf_data_surface).to(dtype = self.dtype, device = self.device)
        if sdf_data_freespace.shape[0] > self.num_voxfield_samples_freespace:
            idx = np.random.choice(sdf_data_freespace.shape[0], size = self.num_voxfield_samples_freespace, replace = False)
            sdf_data_freespace = sdf_data_freespace[idx, :]
        sdf_data_freespace = torch.from_numpy(sdf_data_freespace).to(dtype = self.dtype, device = self.device)
        voxfield_sdf_data_downsampled = torch.cat((sdf_data_surface, sdf_data_freespace), dim = 0)

        # Update extents if necessary
        self.extents[0][0] = min(self.extents[0][0], torch.min(voxfield_sdf_data_downsampled[:,0]))
        self.extents[0][1] = min(self.extents[0][1], torch.min(voxfield_sdf_data_downsampled[:,1]))
        self.extents[0][2] = min(self.extents[0][2], torch.min(voxfield_sdf_data_downsampled[:,2]))
        self.extents[1][0] = max(self.extents[1][0], torch.max(voxfield_sdf_data_downsampled[:,0]))
        self.extents[1][1] = max(self.extents[1][1], torch.max(voxfield_sdf_data_downsampled[:,1]))
        self.extents[1][2] = max(self.extents[1][2], torch.max(voxfield_sdf_data_downsampled[:,2]))

        # Find total time for constructing Voxfield data
        time_voxfield = time.time() - start_voxfield



        ############### GLOBAL SDF TRAINING ################
        # Store time
        start_train = time.time()

        # Data for global SDF training
        self.sdf_data = torch.cat((local_sdf_data, voxfield_sdf_data_downsampled))

        # Update data for SDF trainer
        self.global_sdf_trainer.update_dataset(self.sdf_data)

        # Iterate
        epochs = self.epochs_warm_start if (self.iters < self.warm_start_iters) else self.epochs_nominal
        _ = self.global_sdf_trainer.step(epochs = epochs)
        self.iters += 1

        # Append sensor transformation to trajectory
        self.trajectory.append(self.sensor_transformation)

        # Find total time for training
        time_train = time.time() - start_train
        time_total = time.time() - start_total
        rospy.loginfo(f"===============================================================================")
        rospy.loginfo(f"Time to construct local SDF data with {local_sdf_data.shape[0]} points: {time_local}")
        rospy.loginfo(f"Time to unpack coarse global SDF data with {voxfield_sdf_data_downsampled.shape[0]} points: {time_voxfield}")
        rospy.loginfo(f"Training time for {epochs} epochs: {time_train}")
        rospy.loginfo(f"Total global SDF network update time: {time_total}")
        rospy.loginfo(f"===============================================================================")


        # Perform evaluation if necessary
        if self.do_eval:
            # Make a copy of global network
            global_sdf_copy = copy.deepcopy(self.global_sdf_trainer.model)

            # Choose data for global evaluation
            # Sample observed points
            regular_grid_idx = (self.regular_grid_observed_voxels == True)
            regular_grid_points = self.regular_grid_3d[regular_grid_idx]
            regular_grid_points = regular_grid_points.reshape(-1, 3)
            idx = np.random.choice(regular_grid_points.shape[0], size = min(regular_grid_points.shape[0], self.eval_point_samples), replace = False)
            eval_data_pts = regular_grid_points[idx, :3]

            # Choose data for local evaluation
            local_points_close_idx = (np.linalg.norm(local_points - self.sensor_transformation[:3, 3], axis = 1) < 3.0).astype(bool)
            local_points_eval = local_points[local_points_close_idx, :]
            idx_local = np.random.choice(local_points_eval.shape[0], size = min(local_points_eval.shape[0], self.eval_point_samples), replace = False)
            eval_data_pts_local = local_points_eval[idx_local, :3] 
            
            # Correct ground truth frame rotation for ReplicaCAD
            if self.dataset_format == 'ReplicaCAD':
                r = R.from_rotvec(-np.pi/2 * np.array([1, 0, 0]))
                rotation = r.as_matrix()
            else:
                rotation = np.eye(3)

            # Evaluate
            res = eval_sdf(self.gt_sdf_interp, eval_data_pts, model = global_sdf_copy, eval_pts_rotation = rotation, dtype = self.dtype, device = self.device)
            res_local = eval_sdf(self.gt_sdf_interp, eval_data_pts_local, model = global_sdf_copy, eval_pts_rotation = rotation, dtype = self.dtype, device = self.device)

            # Append to result arrays
            self.sequence_time.append(msg_time.to_sec())
            self.sdf_loss.append(res['sdf_loss'])
            self.eikonal_loss.append(res['eikonal_loss'])
            self.cosdist_loss.append(res['cosdist_loss'])
            self.local_sdf_loss.append(res_local['sdf_loss'])
            self.local_eikonal_loss.append(res_local['eikonal_loss'])
            self.local_cosdist_loss.append(res_local['cosdist_loss'])
            rospy.loginfo(f"Global evaluation against GT with {eval_data_pts.shape[0]} points: Running SDF error: {res['sdf_loss']} - Running Eikonal error: {res['eikonal_loss']} - Running cos dist error: {res['cosdist_loss']}")
            rospy.loginfo(f"Local evaluation against GT with {eval_data_pts_local.shape[0]} points: Running local SDF error: {res_local['sdf_loss']} - Running local Eikonal error: {res_local['eikonal_loss']} - Running local cos dist error: {res_local['cosdist_loss']}")

        # Draw and publish mesh
        if self.save_mesh:
            write_mesh(self.extents, self.global_sdf_trainer.model, os.path.join(self.output_path, "global_sdf_mesh.ply"), dtype = self.dtype, device = self.device)
            self.publish_mesh()
        
        # Save slices
        if self.save_slices:
            write_slices(extents = self.extents, model = self.global_sdf_trainer.model, t = self.iters, save_path = os.path.join(self.output_path, "slices"), camera_pose = self.trajectory[-1], trajectory = np.array(self.trajectory), dtype = self.dtype, device = self.device)

        # Publish SDF data if necessary
        if self.publish_sdf_data:
            self.publish_data_slice(msg_time)
        
        return self.sdf_data
    
    def publish_data_slice(self, stamp):
        # Extract indices of desired slice from SDF data
        sdf_data = self.sdf_data.detach().cpu().numpy()
        slice_idx = np.logical_and(sdf_data[:, 2] < self.sdf_slice_z + 0.07, sdf_data[:, 2] > self.sdf_slice_z - 0.07)
        pc = sdf_data[slice_idx, :]

        # Populate point cloud and convert to ROS msg
        pc_array = np.zeros(len(pc), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        
        pc_array['x'] = pc[:, 0]
        pc_array['y'] = pc[:, 1]
        pc_array['z'] = pc[:, 2]
        pc_array['intensity'] = pc[:, 3] # Intensity = sdf

        # Publish
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp = stamp, frame_id = self.map_frame)
        self.data_slice_publisher.publish(pc_msg)

        return

    def publish_local_points(self, pc, stamp):
        # Populate point cloud and convert to ROS msg
        pc_array = np.zeros(len(pc), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])
        
        pc_array['x'] = pc[:, 0]
        pc_array['y'] = pc[:, 1]
        pc_array['z'] = pc[:, 2]

        # Publish
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp = stamp, frame_id = self.map_frame)
        self.local_points_publisher.publish(pc_msg)

        return
    
    def publish_slices(self, event = None):
        # Make a copy of global network
        global_sdf_copy = copy.deepcopy(self.global_sdf_trainer.model)

        # Get current stamp and convert to seconds
        current_time = rospy.Time().now()
        current_time_sec = current_time.to_sec()

        # Find current SDF slice height
        sdf_slice_height = self.extents[0][2] + (self.extents[1][2]-self.extents[0][2]) * (1.0 + np.cos(2.0 * np.pi * current_time_sec / self.sdf_visualization_period)) / 2.0
        
        # Creating meshgrid
        try:
            x = np.linspace(self.extents[0][0], self.extents[1][0], int((self.extents[1][0] - self.extents[0][0])//self.sdf_visualization_resolution))
            y = np.linspace(self.extents[0][1], self.extents[1][1], int((self.extents[1][1] - self.extents[0][1])//self.sdf_visualization_resolution))
            xv, yv = np.meshgrid(x, y)
            xv = xv.flatten()
            yv = yv.flatten()
            pcd = np.zeros((len(xv),3))
            pcd[:,0] = xv
            pcd[:,1] = yv
            pcd[:,2] = sdf_slice_height
        except:
            rospy.logerr(f"Training has not started yet - returning...")
            return

        # Pass PCD through model
        sdf = global_sdf_copy(torch.tensor(pcd, device = self.device, dtype = self.dtype))
        sdf = sdf.squeeze(-1)

        # Creating Pointcloud2 message
        pc_array = np.zeros(len(pcd), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ])
        
        pc_array['x'] = pcd[:, 0]
        pc_array['y'] = pcd[:, 1]
        pc_array['z'] = sdf_slice_height
        pc_array['intensity'] = sdf.detach().cpu().numpy()

        # Publish Pointcloud
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp = current_time, frame_id = self.map_frame)
        self.sdf_slice_publisher.publish(pc_msg)
    
    def publish_mesh(self):
        # Remove the previous marker mesh file if it exists
        if os.path.exists(self.mesh_resource_file):
            os.remove(self.mesh_resource_file)
        
        # Open the saved mesh file and copy it (unfortunately, RViz does not update the mesh without custom messages otherwise)
        mesh = trimesh.load(os.path.join(self.output_path, "global_sdf_mesh.ply"), force='mesh')
        data = trimesh.exchange.ply.export_ply(mesh)
        self.mesh_resource_file = os.path.join(self.output_path, "global_sdf_mesh_" + str(self.iters) + ".ply")
        out = open(self.mesh_resource_file, "wb+")
        out.write(data)
        out.close()

        # Initialize marker
        self.marker = Marker()
        self.marker.header.frame_id = self.map_frame
        self.marker.header.stamp = rospy.Time.now()
        self.marker.ns = ""

        # Define marker type, ID and action
        self.marker.type = 10
        self.marker.id = 0
        self.marker.action = 0

        # Note: Must set mesh_resource to a valid URL for a model to appear
        self.marker.mesh_resource = "file://" + self.mesh_resource_file
        self.marker.mesh_use_embedded_materials = True

        # Scale
        self.marker.scale.x = (self.extents[1][0] - self.extents[0][0]) / (mesh.bounding_box.bounds[1][0] - mesh.bounding_box.bounds[0][0])
        self.marker.scale.y = (self.extents[1][1] - self.extents[0][1]) / (mesh.bounding_box.bounds[1][1] - mesh.bounding_box.bounds[0][1])
        self.marker.scale.z = (self.extents[1][2] - self.extents[0][2]) / (mesh.bounding_box.bounds[1][2] - mesh.bounding_box.bounds[0][2])

        # Color
        self.marker.color.r = 1.0
        self.marker.color.g = 1.0
        self.marker.color.b = 1.0
        self.marker.color.a = 1.0

        # Pose
        self.marker.pose.position.x = self.extents[0][0]
        self.marker.pose.position.y = self.extents[0][1]
        self.marker.pose.position.z = self.extents[0][2]
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # Publish
        self.mesh_publisher.publish(self.marker)
    
    def plot(self):
        # Remove the mesh marker
        if os.path.exists(self.mesh_resource_file):
            os.remove(self.mesh_resource_file)
        
        # Save mesh
        write_mesh(self.extents, self.global_sdf_trainer.model, os.path.join(self.output_path, "global_sdf_mesh.ply"), dtype = self.dtype, device = self.device)
        
        # Save slices
        if self.save_slices:
            write_slices(extents = self.extents, model = self.global_sdf_trainer.model, t = self.iters, save_path = os.path.join(self.output_path, "slices"), dtype = self.dtype, device = self.device)
        
        # Plot evaluation figures
        if self.do_eval:
            error_dict = {}
            error_dict['times'] = self.sequence_time
            error_dict['errors'] = self.sdf_loss
            error_dict['cosine_dist'] = self.cosdist_loss
            error_dict['eikonal_loss'] = self.eikonal_loss
            error_dict['local_errors'] = self.local_sdf_loss
            error_dict['local_cosine_dist'] = self.local_cosdist_loss
            error_dict['local_eikonal_loss'] = self.local_eikonal_loss
            f = open(self.output_path + str(self.voxel_size) + '/' + self.scene + '.json', 'w')
            json_object = json.dumps(error_dict)
            f.write(json_object)
            f.close()

            plt.plot(self.sequence_time, self.sdf_loss)
            plt.xlabel('Sequence Time [s]')
            plt.ylabel('Global SDF Error [m]')
            plt.ylim(0, 0.5)
            plt.title(self.scene)
            plt.show()
            plt.clf()

            plt.plot(self.sequence_time, self.cosdist_loss)
            plt.xlabel('Sequence Time [s]')
            plt.ylabel('Global Cosine Distance')
            plt.title(self.scene)
            plt.show()
            plt.clf()

            plt.plot(self.sequence_time, np.asarray(self.eikonal_loss))
            plt.xlabel('Sequence Time [s]')
            plt.ylabel('Global Eikonal Loss')
            plt.title(self.scene)
            plt.show()
            plt.clf()

            plt.plot(self.sequence_time, self.local_sdf_loss)
            plt.xlabel('Sequence Time [s]')
            plt.ylabel('Local SDF Error [m]')
            plt.ylim(0, 0.5)
            plt.title(self.scene)
            plt.show()
            plt.clf()

            plt.plot(self.sequence_time, self.local_cosdist_loss)
            plt.xlabel('Sequence Time [s]')
            plt.ylabel('Local Cosine Distance')
            plt.title(self.scene)
            plt.show()
            plt.clf()

            plt.plot(self.sequence_time, np.asarray(self.local_eikonal_loss))
            plt.xlabel('Sequence Time [s]')
            plt.ylabel('Local Eikonal Loss')
            plt.title(self.scene)
            plt.show()
            plt.clf()

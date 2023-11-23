"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
Vasileios Vasilopoulos (vasileios.v@samsung.com; vasilis.vasilop@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""

# General imports
import logging
import numpy as np
import random
import time

# NN imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from .chamfer_distance import ChamferDistance


class PointCloud_SDF(nn.Module):
    def __init__(
        self,
        pcd: np.ndarray,
        sphere_radius: float = 0.0,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float,
    ):
        
        """
        Brute force estimation of SDF for a point cloud.
        
        :param pcd: N x 3 numpy array describing the point cloud
        :param sphere_radius: Radius of sphere enclosing each point in the point cloud
        :param device: Device to load the point cloud
        :param dtype: Type of tensors
        """
        
        super().__init__()
        
        self._device = device
        self._dtype = dtype
        self._sphere_radius = sphere_radius
        
        self.pcd = torch.from_numpy(pcd).to(dtype = self._dtype, device = self._device)
        
        self.chamfer_distance = ChamferDistance()
    
    def update_pcd(
        self,
        pcd: np.ndarray,
    ):
        self.pcd = torch.from_numpy(pcd).to(dtype = self._dtype, device = self._device)
        
    def forward(
        self,
        x: torch.Tensor,
        signs: torch.Tensor = None
    ):
        """
        Main SDF computation function.

        :param x: M x 3 tensor describing the query points
        :param signs: Optional M x 1 tensor that specifies the signs of the outputs
        :returns: M x 1 tensor giving the raw distances of the query points to the point cloud
        """
        
        # Compute distance using ChamferDistance (returns the squared distance between the point clouds)
        dist_x_to_pcd, idx = self.chamfer_distance(x.unsqueeze(0), self.pcd.unsqueeze(0))
        dist_x_to_pcd = torch.sqrt(dist_x_to_pcd.transpose(0,1)) - self._sphere_radius

        # Apply the correct signs
        if signs is not None:
            dist_x_to_pcd = dist_x_to_pcd * signs

        return dist_x_to_pcd


if __name__ == "__main__":
    DEVICE = "cuda"
    
    pcd = np.random.rand(100000,3)

    model = PointCloud_SDF(pcd, device=DEVICE)
    model.eval()
    model.to(DEVICE)
    
    since = time.time()
    points = torch.rand((500, 3), device=DEVICE, requires_grad=True)

    # with torch.no_grad():
    sdf_values = model(points)
    print(f'Total time to compute the SDF value: {time.time()-since}')
    
    total_sdf_value = sdf_values.sum()
    since = time.time()
    total_sdf_value.backward()
    sdf_gradient = points.grad
    sdf_gradient = sdf_gradient[torch.nonzero(sdf_gradient).data[0][0]]
    print(f'Total time to compute the SDF gradient: {time.time()-since}')
    print(f'SDF values: {sdf_values.cpu().detach().numpy()}')
    print(f'SDF gradient: {sdf_gradient.cpu().detach().numpy()}')

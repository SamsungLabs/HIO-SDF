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

import torch

from .global_sdf import HIOSDFNet
from .global_sdf_dataset import Dataset
from .global_sdf_losses import sdf_loss, eikonal_loss
from .utils import gradient

class GlobalSDFTrainer():
    def __init__(
        self,
        learning_rate = 0.0004,
        weight_decay = 0.012,
        sdf_loss_weight = 5.0,
        eikonal_loss_weight = 2.0,
        device = 'cuda',
        dtype = torch.float,
    ):
        # Device
        self.device = device
        self.dtype = dtype

        # Optimization parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss parameters
        self.loss_weights = dict()
        self.loss_weights['sdf_loss'] = sdf_loss_weight
        self.loss_weights['eikonal_loss'] = eikonal_loss_weight

        # Model
        self.model = HIOSDFNet()
        self.model.to(self.device)
        self.model.train()

        # Dataset
        initial_data = torch.rand(1000,7)
        self.dataset = Dataset(initial_data, dtype = self.dtype, device = self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay
        )

        # Initialize losses
        self.losses = []

    def update_dataset(self, np_data):
        self.dataset.update_dataset(np_data)

    def step(self, epochs = 50):
        epoch = 0

        # Extract data and initialize
        batch_points = self.dataset.pc_data
        batch_sdf = self.dataset.sdf_data
        batch_points.requires_grad_()
        
        while epoch < epochs:
            self.optimizer.zero_grad()

            # Compute SDF predictions and target SDF values
            sdf_preds = self.model(batch_points)
            target_sdf_preds = batch_sdf

            # Compute predicted and groundtruth SDF gradients
            sdf_gradient_preds = gradient(batch_points, sdf_preds)

            # Compute loss
            self.total_sdf_loss = sdf_loss(sdf_preds, target_sdf_preds)
            self.total_eikonal_loss = eikonal_loss(sdf_gradient_preds)
            self.total_loss = self.loss_weights['sdf_loss'] * torch.mean(self.total_sdf_loss) + self.loss_weights['eikonal_loss'] * torch.mean(self.total_eikonal_loss)
            self.total_loss.backward()

            # Update the weights
            self.optimizer.step()

            # Append the loss
            self.losses.append(self.total_sdf_loss.mean().detach().cpu().numpy())
            
            # print(f"Epoch: {epoch}, mean SDF loss: {torch.mean(self.total_sdf_loss)}, mean Eikonal loss: {torch.mean(self.total_eikonal_loss)}")
            epoch += 1
        
        return self.losses[-1]
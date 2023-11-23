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

class Dataset(torch.utils.data.Dataset):
	def __init__(self, data, dtype = torch.float, device = 'cuda'):
		self.dtype = dtype
		self.device = device
		self.pc_data = data[:, :3]   # PC info
		self.sdf_data = data[:, 3].unsqueeze(1)  # SDF

	def __getitem__(self, index):
		point = self.pc_data[index,:]
		sdf = self.sdf_data[index]
		return point, sdf

	def __len__(self):
		return len(self.pc_data)

	def update_dataset(self, data):
		self.pc_data = data[:, :3]   # PC info
		self.sdf_data = data[:, 3].unsqueeze(1)  # SDF
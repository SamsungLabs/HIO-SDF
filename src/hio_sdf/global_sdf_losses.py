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

def sdf_loss(sdf, target_sdf):
    """
    L1 Loss on sdf values : |target_sdf - sdf|
    
    :param sdf: (n,) predicted SDF values
    :param target_sdf: (n,) target SDF values
    """
    return torch.abs(sdf-target_sdf)

def eikonal_loss(grad_sdf):
    """
    Eikonal Loss | || grad_sdf || -1 |
    
    :param grad_sdf: (n,3) predicted SDF gradients
    """

    return torch.abs(torch.norm(grad_sdf, dim = -1, keepdim = True) - 1)

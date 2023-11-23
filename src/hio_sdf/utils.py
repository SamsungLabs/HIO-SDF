# MIT License

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by: Vasileios Vasilopoulos (vasileios.v@samsung.com)
#              Suveer Garg (suveer.garg@samsung.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
import torch
import time
from torch.autograd import grad


def gradient(inputs, outputs):
    """
    Returns the gradients of the outputs with respect to the inputs

    :param inputs: Input tensors
    :param outputs: Output tensors
    """

    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)

    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    return points_grad


def get_colormap(sdf_range=[-2, 2], surface_cutoff=0.01):
    white = np.array([1., 1., 1., 1.])
    sdf_range[1] += surface_cutoff - (sdf_range[1] % surface_cutoff)
    sdf_range[0] -= surface_cutoff - (-sdf_range[0] % surface_cutoff)

    positive_n_cols = int(sdf_range[1] / surface_cutoff)
    viridis = cm.get_cmap('viridis', positive_n_cols)
    positive_colors = viridis(np.linspace(0.2, 1, int(positive_n_cols)))
    positive_colors[0] = white

    negative_n_cols = int(-sdf_range[0] / surface_cutoff)
    redpurple = cm.get_cmap('RdPu', negative_n_cols).reversed()
    negative_colors = redpurple(np.linspace(0., 0.7, negative_n_cols))
    negative_colors[-1] = white

    colors = np.concatenate(
        (negative_colors, white[None, :], positive_colors), axis=0)
    sdf_cmap = ListedColormap(colors)

    norm = mpl.colors.Normalize(sdf_range[0], sdf_range[1])
    sdf_cmap_fn = cm.ScalarMappable(norm=norm, cmap=sdf_cmap)
    # plt.colorbar(sdf_cmap_fn)
    # plt.show()
    return sdf_cmap_fn

def get_grid_pts(dims, transform):
    x = np.arange(dims[0])
    y = np.arange(dims[1])
    z = np.arange(dims[2])
    x = x * transform[0, 0] + transform[0, 3]
    y = y * transform[1, 1] + transform[1, 3]
    z = z * transform[2, 2] + transform[2, 3]

    return x, y, z

def sdf_interpolator(sdf_grid, transform):
    x, y, z = get_grid_pts(sdf_grid.shape, transform)
    start = time.time()
    
    sdf_interp = scipy.interpolate.RegularGridInterpolator(
        (x, y, z), sdf_grid)
    
    print(f'Time to create regular grid interpolator {time.time() - start}')

    return sdf_interp

def eval_sdf_interp(sdf_interp, pc, handle_oob='except', oob_val=0.):
    """
    :param sdf_interp: SDF interpolator
    :param pc: Point cloud data points
    :param handle_oob: dictates what to do with out of bounds points. Must take either 'except', 'mask' or 'fill'
    :param oob_val: Out-of-bounds default value
    """
    reshaped = False
    if pc.ndim != 2:
        reshaped = True
        pc_shape = pc.shape[:-1]
        pc = pc.reshape(-1, 3)

    if handle_oob == 'except':
        sdf_interp.bounds_error = True
    elif handle_oob == 'mask':
        dummy_val = 1e99
        sdf_interp.bounds_error = False
        sdf_interp.fill_value = dummy_val
    elif handle_oob == 'fill':
        sdf_interp.bounds_error = False
        sdf_interp.fill_value = oob_val
    else:
        assert True, "handle_oob must take a recognised value."

    sdf = sdf_interp(pc)

    if reshaped:
        sdf = sdf.reshape(pc_shape)

    if handle_oob == 'mask':
        valid_mask = sdf != dummy_val
        return sdf, valid_mask

    return sdf

def load_gt_sdf(gt_sdf_file, sdf_transf_file, dataset_format):
    sdf_grid = np.load(gt_sdf_file)
    if dataset_format == "ScanNet":
        sdf_grid = np.abs(sdf_grid)
    sdf_transform = np.loadtxt(sdf_transf_file)
    gt_sdf_interp = sdf_interpolator(
        sdf_grid, sdf_transform)
    sdf_dims = torch.tensor(sdf_grid.shape)
    x,y,z    = get_grid_pts(sdf_grid.shape, sdf_transform)
    grid = torch.meshgrid(torch.tensor(x), torch.tensor(y), torch.tensor(z))
    grid_3d = torch.cat((grid[0][..., None],
                        grid[1][..., None],
                        grid[2][..., None]), 
                        dim=3)

    return gt_sdf_interp, sdf_grid, grid_3d

def eval_sdf(gt_sdf_interp, eval_pts, pred_sdf = None, pred_sdf_interp = None, model = None, eval_pts_rotation = np.eye(3), dtype = torch.float, device = 'cuda'):
    if model is not None:
        eval_pts_tensor = torch.from_numpy(eval_pts).to(dtype = dtype, device = device)
        eval_pts_tensor.requires_grad_()
        pred_sdf = model(eval_pts_tensor)
        pred_sdf_grad = gradient(eval_pts_tensor, pred_sdf)
        pred_sdf = pred_sdf.squeeze().detach().cpu().numpy()
        pred_sdf_grad = pred_sdf_grad.detach().cpu().numpy()
        pred_sdf_grad = (eval_pts_rotation @ pred_sdf_grad.T).T
        
    eval_pts = (eval_pts_rotation @ eval_pts.T).T
    
    gt_sdf, valid_mask_gt = eval_sdf_interp(
        gt_sdf_interp, eval_pts,
        handle_oob='mask')
    
    # gt sdf gives value 0 inside the walls. Don't include this in loss
    valid_mask_gt = np.logical_and(gt_sdf != 0., valid_mask_gt)

    if not isinstance(pred_sdf, np.ndarray):
        pred_sdf, valid_mask_sdf = eval_sdf_interp(
            pred_sdf_interp, eval_pts,
            handle_oob='mask')
        valid_mask_sdf = np.logical_and(valid_mask_sdf, ~np.isnan(pred_sdf))
        valid_mask = np.logical_and(valid_mask_gt, valid_mask_sdf)
    else:
        valid_mask = valid_mask_gt

    gt_sdf = gt_sdf[valid_mask]
    pred_sdf = pred_sdf[valid_mask]
    gt_sdf = torch.from_numpy(gt_sdf).to('cpu')
    pred_sdf = torch.from_numpy(pred_sdf).to('cpu')
    
    # Gradient evaluation
    gt_diff, valid_gt_diff = eval_grad(gt_sdf_interp, eval_pts, 0.05, is_gt_sdf=True, device=device)
    if not pred_sdf_interp is None:
        pred_diff, valid_pred_diff = eval_grad(pred_sdf_interp, eval_pts, 0.05, is_gt_sdf=False, device=device)
    else:
        pred_diff = pred_sdf_grad
        valid_pred_diff = np.ones(pred_diff.shape[0]).astype(bool)

    valid_diffs = np.logical_and(valid_pred_diff, valid_gt_diff)
    
    valid_mask = np.logical_and(valid_diffs, valid_mask)

    pred_diff = pred_diff[valid_mask]
    gt_diff = gt_diff[valid_mask]

    with torch.set_grad_enabled(False):

        sdf_diff = pred_sdf - gt_sdf
        sdf_diff = torch.abs(sdf_diff)
        l1_sdf = sdf_diff.mean()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosdist = 1 - cos(torch.tensor(pred_diff), torch.tensor(gt_diff))
        cosdist_mean = cosdist.mean()

        eikonal_loss = np.mean(np.abs(1.0 - np.linalg.norm(pred_diff, axis=1)))

    res = {
        'sdf_loss': l1_sdf.item(),
        'cosdist_loss' : cosdist_mean.item(),
        'eikonal_loss' : eikonal_loss.item()
    }

    return res

def eval_grad(sdf_interp, pts, delta, is_gt_sdf, device = 'cuda'):
    grad = np.zeros(pts.shape)

    for i in range(3):
        for dx in [-1, 1]:

            offset = np.zeros(3)
            offset[i] += dx * delta
            offset_pts = pts + offset[None, :]

            if is_gt_sdf:
                sdf, valid_mask = eval_sdf_interp(
                    sdf_interp, offset_pts, handle_oob='mask')
                valid_mask = np.logical_and(sdf != 0., valid_mask)
                sdf[~valid_mask] = np.nan
            else:
                sdf = eval_sdf_interp(
                       sdf_interp, offset_pts, handle_oob='fill', oob_val=np.nan)

            grad[:, i] += dx * sdf

    grad /= (2 * delta)

    valid_mask = ~np.isnan(np.linalg.norm(grad, axis=1))

    return grad, valid_mask
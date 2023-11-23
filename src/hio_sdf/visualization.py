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

import os
import time
import numpy as np
import scipy
import torch
from scipy.spatial import KDTree
import trimesh
import skimage
import cv2
import PIL
from PIL import Image
import imageio

from .utils import get_colormap

file_path = os.path.abspath(os.path.dirname(__file__))

def to_topdown(extents, pts, im_size, up_ix = 2):
    scale = extents[1, :] - extents[0, :]
    cam_td = (pts - extents[0, :]) / scale
    cam_td = np.concatenate((
        cam_td[:, :up_ix], cam_td[:, up_ix + 1:]), axis=1)
    cam_td = cam_td * im_size
    cam_td = cam_td.astype(int)

    return cam_td

def draw_agent(c_im, agent_position, agent_rotation, agent_radius_px):
    sprite_file = os.path.join(file_path, "100x100.png")
    AGENT_SPRITE = imageio.imread(sprite_file)

    AGENT_SPRITE = AGENT_SPRITE.reshape(-1, 4)
    sums = AGENT_SPRITE[:, :3].sum(axis=1)
    ixs = sums > 600
    AGENT_SPRITE[ixs, 3] = 0
    AGENT_SPRITE[~ixs, :3] = [0, 255, 0]
    AGENT_SPRITE = AGENT_SPRITE.reshape(100, 100, 4)

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.interpolation.rotate(
        AGENT_SPRITE, agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = AGENT_SPRITE.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )

    c_im = paste_overlapping_image(c_im, resized_agent, agent_position)

    return c_im

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location,
    mask=None,
):
    """
    https://github.com/facebookresearch/habitat-lab/blob/786a5eec68cf3b4cf7134af615394c981d365a89/habitat/utils/visualizations/utils.py
    Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]): (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]): (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0]: foreground.shape[0] - max_pad[0],
        min_pad[1]: foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0]: foreground.shape[0] - max_pad[0],
            min_pad[1]: foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background

def write_mesh(
    extents,
    model,
    filename,
    grid_dim = 200,
    chunk_size = 100000,
    color_by = "normals",
    device = 'cuda',
    dtype = torch.float
):
    # Find scale
    extent_limits = extents[1, :] - extents[0, :]
    max_extent = np.max(extent_limits)
    scale_multiplier = max_extent/extent_limits

    # Create grid
    x = torch.linspace(extents[0][0], extents[1][0], steps = int(grid_dim // scale_multiplier[0]), device = device, dtype = dtype)
    y = torch.linspace(extents[0][1], extents[1][1], steps = int(grid_dim // scale_multiplier[1]), device = device, dtype = dtype)
    z = torch.linspace(extents[0][2], extents[1][2], steps = int(grid_dim // scale_multiplier[2]), device = device, dtype = dtype)
    grid = torch.meshgrid(x, y, z)
    grid_3d = torch.cat(
        (
            grid[0][..., None],
            grid[1][..., None],
            grid[2][..., None],
        ),
        dim=3
    )
    grid_3d = grid_3d.view(-1, 3)

    # Pass grid through the model to find SDF values
    head = 0
    num_samples = grid_3d.shape[0]
    sdf = torch.zeros(num_samples)
    while head < num_samples:
        sample_subset = grid_3d[head : min(head + chunk_size, num_samples), :].to(device)

        sdf[head : min(head + chunk_size, num_samples)] = (
            model(sample_subset)
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += chunk_size
    sdf = sdf.reshape(int(grid_dim // scale_multiplier[0]), int(grid_dim // scale_multiplier[1]), int(grid_dim // scale_multiplier[2]))

    # Run marching cubes
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(sdf, level=0.0)

    dim = sdf.shape[0]
    vertices = vertices / (dim - 1)
    vertices = vertices * extent_limits * scale_multiplier
    mesh = trimesh.Trimesh(
        vertices=vertices,
        vertex_normals=vertex_normals,
        faces=faces
    )

    # Coloring
    if color_by == "normals":
        norm_cols = (- mesh.vertex_normals + 1) / 2
        norm_cols = np.clip(norm_cols, 0., 1.)
        norm_cols = (norm_cols * 255).astype(np.uint8)
        alphas = np.full([norm_cols.shape[0], 1], 255, dtype=np.uint8)
        cols = np.concatenate((norm_cols, alphas), axis=1)
        mesh.visual.vertex_colors = cols
    elif color_by == "height":
        zs = mesh.vertices[:, 1]
        cols = trimesh.visual.interpolate(zs, color_map='viridis')
        mesh.visual.vertex_colors = cols
    else:
        mesh.visual.face_colors = [160, 160, 160, 255]

    # Export mesh
    data = trimesh.exchange.ply.export_ply(mesh)
    out = open(filename, "wb+")
    out.write(data)
    out.close()

def write_slices(
    extents,
    model,
    t,
    save_path,
    grid_dim = 200,
    chunk_size = 100000,
    prefix="",
    n_slices=20,
    sdf_range=[-2, 2],
    trajectory = None,
    camera_pose = None,
    device = 'cuda',
    dtype = torch.float
):
    slices = compute_slices(
        extents,
        model,
        grid_dim = grid_dim,
        chunk_size = chunk_size,
        z_ixs=None,
        n_slices=n_slices,
        sdf_range=sdf_range,
        trajectory=trajectory,
        camera_pose=camera_pose,
        device = device,
        dtype = dtype,
    )

    for s in range(n_slices):
        # Not ScanNet
        cv2.imwrite(
            os.path.join(save_path, prefix + f"pred_{s}_{t}.png"),
            slices["pred_sdf"][s][..., ::-1])
        
        # ScanNet
        # image = slices["pred_sdf"][s][..., ::-1]
        # pil_image = Image.fromarray(image)
        # pil_image = pil_image.rotate(-77, PIL.Image.NEAREST, expand = 0)
        # result = np.array(pil_image)
        # idx = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
        # result[idx] = 255
        # cv2.imwrite(
        #     os.path.join(save_path, prefix + f"pred_{s}_{t}.png"),
        #     result)
        
def compute_slices(
    extents,
    model,
    grid_dim = 200,
    chunk_size = 100000,
    z_ixs=None,
    n_slices=20,
    sdf_range=[-2, 2],
    trajectory = None,
    camera_pose = None,
    device = 'cuda',
    dtype = torch.float
):
    # Index for z values
    up_idx = 2

    # Find scale
    scale = extents[1, :] - extents[0, :]

    # Compute points to query
    if z_ixs is None:
        z_ixs = torch.linspace(30, grid_dim - 30, n_slices)
        z_ixs = torch.round(z_ixs).long()
    z_ixs = z_ixs.to(device)

    # Create grid
    x = torch.linspace(extents[0][0], extents[1][0], steps = grid_dim, device = device, dtype = dtype)
    y = torch.linspace(extents[0][1], extents[1][1], steps = grid_dim, device = device, dtype = dtype)
    z = torch.linspace(extents[0][2], extents[1][2], steps = grid_dim, device = device, dtype = dtype)
    grid = torch.meshgrid(x, y, z)
    grid_3d = torch.cat(
        (
            grid[0][..., None],
            grid[1][..., None],
            grid[2][..., None],
        ),
        dim=3
    )

    # Select desired z indices
    pc = grid_3d.reshape(
        grid_dim, grid_dim, grid_dim, 3)
    pc = torch.index_select(pc, up_idx, z_ixs)

    # Image parameters
    cmap = get_colormap(sdf_range=sdf_range)
    grid_shape = pc.shape[:-1]
    n_slices = grid_shape[up_idx]
    pc = pc.reshape(-1, 3)
    scales = np.concatenate([scale[:up_idx], scale[up_idx + 1:]])
    im_size = 256 * scales / scales.min()
    im_size = im_size.astype(int)
    for i in range(im_size.shape[0]):
        if im_size[i]%2 != 0:
            im_size[i] -= 1
    slices = {}

    with torch.set_grad_enabled(False):
        # Pass grid through the model to find SDF values
        head = 0
        num_samples = pc.shape[0]
        sdf = torch.zeros(num_samples)
        while head < num_samples:
            sample_subset = pc[head : min(head + chunk_size, num_samples), :].to(device)

            sdf[head : min(head + chunk_size, num_samples)] = (
                model(sample_subset)
                .squeeze()#.squeeze(1)
                .detach()
                .cpu()
            )
            head += chunk_size
    sdf_viz = cmap.to_rgba(sdf.flatten(), alpha=1., bytes=False)
    sdf_viz = (sdf_viz * 255).astype(np.uint8)[..., :3]
    sdf_viz = sdf_viz.reshape(*grid_shape, 3)
    sdf_viz = [
        cv2.resize(np.take(sdf_viz, i, up_idx), im_size[::-1])
        for i in range(n_slices)
    ]
    # slices["pred_sdf"] = np.flip(sdf_viz, axis = 1) # flip for top-down view
    slices["pred_sdf"] = sdf_viz

    if camera_pose is not None:
        camera_xyz = np.expand_dims(camera_pose[:3, 3], axis = 0)
        cam_td = to_topdown(extents, camera_xyz, im_size)[0]

        camera_rotation = camera_pose[:3, :3]
        camera_angle = np.arctan2(camera_rotation[1, 0], camera_rotation[0, 0]) - np.pi/2
        
        for i, im in enumerate(slices["pred_sdf"]):
            draw_agent(im, cam_td, agent_rotation=camera_angle, agent_radius_px=12)
            slices["pred_sdf"][i] = im
    
    if trajectory is not None:
        traj_td = to_topdown(extents, trajectory[:, :3, 3], im_size)
        
        for i, im in enumerate(slices["pred_sdf"]):
            for j in range(len(traj_td) - 1):
                if not (traj_td[j] == traj_td[j + 1]).all():
                    im = im.astype(np.uint8) / 255
                    im = cv2.line(
                        im,
                        traj_td[j][::-1],
                        traj_td[j + 1][::-1],
                        [1., 0., 0.], 2)
                    im = (im * 255).astype(np.uint8)                    
            slices["pred_sdf"][i] = im

    return slices
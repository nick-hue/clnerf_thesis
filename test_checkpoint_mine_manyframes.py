#!/usr/bin/env python
import torch
from torch import nn
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data and model imports from the repo
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGPGv2
from models.rendering_NGPA import render, MAX_SAMPLES
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Utilities for checkpoint loading and argument parsing
from utils.utils import load_ckpt
from opt import get_opts

import warnings
warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return depth_img


class NeRFSystem(torch.nn.Module):
    """
    A simplified version of the NeRFSystem for rendering.
    This script is intended to check that the checkpoint is renderable.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        # The vocab_size is set based on task index (for continual learning)
        self.model = NGPGv2(scale=self.hparams.scale, vocab_size=self.hparams.task_curr+1, 
                             rgb_act=rgb_act, dim_a=self.hparams.dim_a, dim_g=self.hparams.dim_g)
        G = self.model.grid_size
        self.model.register_buffer('density_grid', torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords', 
                                     create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        if split == 'train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            embed_id = batch['ts']
        else:
            poses = batch['pose'].to(self.device)
            directions = self.directions
            embed_id = batch['ts'][0].to(self.device) * torch.ones(self.directions.flatten().size(0),
                                                                    dtype=batch['ts'].dtype,
                                                                    device=self.device)
        # If external optimization is enabled, adjust poses (if applicable)
        if self.hparams.optimize_ext and 'img_idxs' in batch:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)
        kwargs = {'test_time': split != 'train', 'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        return render(self.model, rays_o, rays_d, embed_id, **kwargs)

    def setup(self, stage):
        # Set up the dataset(s)
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {
            'root_dir': self.hparams.root_dir,
            'downsample': self.hparams.downsample,
            'task_number': self.hparams.task_number,
            'task_curr': self.hparams.task_curr,
            'task_split_method': self.hparams.task_split_method,
            'rep_size': self.hparams.rep_size,
            'rep_dir': f'results/NGPGv2/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep',
            'nerf_rep': self.hparams.nerf_rep
        }
        # Here we only need the test dataset to render a frame
        self.test_dataset = dataset(split='test', **kwargs)
        # Register buffers so that forward() has access to them.
        self.register_buffer('directions', self.test_dataset.directions.to(self.device))
        self.register_buffer('poses', self.test_dataset.poses.to(self.device))


def main():
    # Parse command-line arguments (using the repo's opt.py)
    hparams = get_opts()
    # Ensure we're in validation (rendering) mode and a checkpoint path is provided
    hparams.val_only = True
    if not hparams.weight_path:
        raise ValueError("Please provide a checkpoint path using --weight_path.")
    
    # Instantiate the system and move it to device
    system = NeRFSystem(hparams).to('cuda' if torch.cuda.is_available() else 'cpu')
    # Set up the system (loads test dataset, registers buffers, etc.)
    system.setup(stage="test")
    
    # Load checkpoint weights into the model
    load_ckpt(system.model, hparams.weight_path)
    print("Checkpoint loaded successfully.")

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(system.test_dataset, batch_size=None, num_workers=4, pin_memory=True)
    
    # Create directory for saving frames
    output_dir = "counter_rendered_frames_v2"
    os.makedirs(output_dir, exist_ok=True)
    rendered_frames = []
    
    # Render 10 frames from the test set
    for i, batch in enumerate(test_loader):
        if i >= 10:
            break
        system.eval()
        with torch.no_grad():
            results = system(batch, split='test')
        w, h_img = system.test_dataset.img_wh
        rgb_image = rearrange(results['rgb'], '(h w) c -> h w c', h=h_img)
        rgb_image = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
        
        # Save each frame as a separate image
        frame_fname = os.path.join(output_dir, f"frame_{i:02d}.png")
        imageio.imsave(frame_fname, rgb_image)
        print(f"Frame {i} saved to {frame_fname}")
        rendered_frames.append(rgb_image)
    
    # Concatenate frames horizontally (if you want one image with frames side by side)
    if rendered_frames:
        combined_image = np.concatenate(rendered_frames, axis=1)
        combined_fname = os.path.join(output_dir, "combined.png")
        imageio.imsave(combined_fname, combined_image)
        print(f"Combined image saved to {combined_fname}")



if __name__ == '__main__':
    main()

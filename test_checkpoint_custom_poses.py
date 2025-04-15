#!/usr/bin/env python
import torch
from torch import nn
import os
import imageio
import numpy as np
import cv2
from einops import rearrange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import datetime
import os

# Data and model imports from the repo
from datasets.ray_utils import get_rays
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
    This version is used for rendering from custom camera poses.
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
        from kornia.utils.grid import create_meshgrid3d
        self.model.register_buffer('grid_coords', 
                                     create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        # directions will be set up using the test dataset intrinsics.
        self.directions = None

    def forward(self, batch, split):
        if split == 'train':
            raise ValueError("Rendering from custom poses should be done in test mode")
        else:
            poses = batch['pose'].to(self.device)
            directions = self.directions
            embed_id = batch['ts'][0].to(self.device) * torch.ones(
                directions.flatten().size(0),
                dtype=batch['ts'].dtype,
                device=self.device
            )
        rays_o, rays_d = get_rays(directions, poses)
        kwargs = {'test_time': True, 'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch.get('exposure', None)
        return render(self.model, rays_o, rays_d, embed_id, **kwargs)

    def setup_from_test(self):
        """
        Set up intrinsics and ray directions using the test dataset.
        """
        from datasets import dataset_dict
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
        test_dataset = dataset(split='test', **kwargs)
        self.directions = test_dataset.directions.to(self.device)
        self.img_wh = test_dataset.img_wh

def write_experiment_log(output_dir, experiment_info):
    """
    Write key experiment information to a log file inside output_dir.
    experiment_info should be a dictionary containing the parameters to log.
    """
    log_file = os.path.join(output_dir, "experiment.log")
    with open(log_file, "w") as f:
        f.write("Experiment Log\n")
        f.write("==============\n")
        for key, value in experiment_info.items():
            f.write(f"{key}: {value}\n")
    print(f"Experiment log written to {log_file}")

def generate_custom_poses(num_frames, radius=1.0, center=np.array([0, 0, 0]), vertical_amplitude=0):
    """
    Generate an orbit of camera poses around the given center.
    Each pose is a 4x4 camera-to-world matrix.
    """
    poses = []
    angles = []
    render_angles = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        angles.append(angle)
        
        if not (3.24 <= angle <= 6.09):
            print(f"Skipping frame {i} with angle {angle:.3f}")
            continue
        render_angles.append((i, angle))

        # x_offset = 0.5 + i*0.1 # the desired constant shift along x
        # print(x_offset)
        # cam_x = center[0] + radius * np.cos(angle) + x_offset
        cam_x = center[0] + radius * np.cos(angle)
        cam_z = center[2] + radius * np.sin(angle)
        cam_y = center[1] + vertical_amplitude * np.sin(angle)
        position = np.array([cam_x, cam_y, cam_z])
        
        forward = (center - position)
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        R = np.stack([right, up, forward], axis=1)
        
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = position
        poses.append(pose)
    return poses, angles, render_angles

def get_directories_number(dir):
    return len(next(os.walk(dir))[1])

def make_dir(exp_info):
    # Create a unique output directory based on a timestamp to store the rendered frames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(exp_info['base_output_dir'], timestamp+f"_v{str(exp_info['id'])}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def render_frames(system, custom_poses, output_dir):    
    # Render each custom frame.
    for i, pose in enumerate(custom_poses):
        system.eval()
        with torch.no_grad():
            batch = {
                'pose': torch.from_numpy(pose).unsqueeze(0).float().to(system.device),
                'ts': torch.tensor([0], dtype=torch.int64, device=system.device)
            }
            results = system(batch, split='test')
        
        w, h_img = system.img_wh
        rgb_image = rearrange(results['rgb'], '(h w) c -> h w c', h=h_img)
        rgb_image = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
        
        frame_fname = os.path.join(output_dir, f"custom_frame_{i:02d}.png")
        imageio.imsave(frame_fname, rgb_image)
        print(f"Custom frame {i:02d} saved to {frame_fname}")

    print(f"{len(custom_poses)} frames successfully rendered at {output_dir}")


def run_experiment(system, custom_poses, experiment_info):
    output_dir = make_dir(experiment_info)
    # Render frames from generated camera poses
    render_frames(system, custom_poses, output_dir)

    # Log experiment information
    write_experiment_log(output_dir, experiment_info)


def main():
    # Parse options and update hparams for rendering.
    hparams = get_opts()
    hparams.val_only = True
    if not hparams.weight_path:
        raise ValueError("Please provide a checkpoint path using --weight_path.")
    
    system = NeRFSystem(hparams).to('cuda' if torch.cuda.is_available() else 'cpu')
    system.setup_from_test() # Set up directions and intrinsics using the test dataset.
    
    # Load checkpoint weights
    load_ckpt(system.model, hparams.weight_path)
    print("Checkpoint loaded successfully.")

    base_output_dir = "custom_rendered_frames"
    experiment_id = get_directories_number(base_output_dir) + 1
    num_frames = 31
    radius = 1.5
    # vertical_amplitude = 0.25  * radius
    vertical_amplitude = 0

    # Generate custom camera poses
    custom_poses, custom_angles, good_angles = generate_custom_poses(num_frames=num_frames, radius=radius, center=np.array([0, 0, 0]), vertical_amplitude=vertical_amplitude)

    print("All angles")
    for i, angle in enumerate(custom_angles):
        print(f"Frame {i:02d} : {angle=}")

    print("Good angles")
    for i, angle in good_angles:
        print(f"Frame {i:02d} : {angle=}")

    experiment_info = {
        "id": experiment_id,
        "Requested num_frames": num_frames,
        "Radius": radius,
        "Vertical Amplitued": vertical_amplitude,
        "Poses generated": len(custom_poses),
        "Frames rendered": len(custom_angles),
        "base_output_dir" : base_output_dir,
    }

    # print(experiment_info)

    run_experiment(system, custom_poses, experiment_info)
    

if __name__ == '__main__':
    main()

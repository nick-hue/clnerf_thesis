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
import time  


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
        # self.model = NGPGv2(scale=self.hparams.scale, vocab_size=self.hparams.task_curr+1, 
        #                      rgb_act=rgb_act, dim_a=self.hparams.dim_a, dim_g=self.hparams.dim_g)
        self.model = NGPGv2(scale=self.hparams.scale,    vocab_size=self.hparams.vocab_size, rgb_act=rgb_act,
                            dim_a=self.hparams.dim_a,dim_g=self.hparams.dim_g)

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
        # print('bef loading')
        test_dataset = dataset(split='test', **kwargs)
        # print('after loading')
        self.directions = test_dataset.directions.to(self.device)
        self.img_wh = test_dataset.img_wh

    def setup_intrinsics(self, w, h):
        @torch.cuda.amp.autocast(dtype=torch.float32)
        def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True, crop_region = 'full'):
            """
            Get ray directions for all pixels in camera coordinate [right down front].
            Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                    ray-tracing-generating-camera-rays/standard-coordinate-systems

            Inputs:
                H, W: image height and width
                K: (3, 3) camera intrinsics
                random: whether the ray passes randomly inside the pixel
                return_uv: whether to return uv image coordinates

            Outputs: (shape depends on @flatten)
                directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
                uv: (H, W, 2) or (H*W, 2) image coordinates
            """
            from kornia import create_meshgrid
            grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
            u, v = grid.unbind(-1)

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if random:
                directions = \
                    torch.stack([(u-cx+torch.rand_like(u))/fx,
                                (v-cy+torch.rand_like(v))/fy,
                                torch.ones_like(u)], -1)
            else: # pass by the center
                directions = \
                    torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
            
            if crop_region  == 'left':
                directions = directions[:, :directions.shape[1]//2]

            elif crop_region == 'right':
                directions = directions[:, directions.shape[1]//2:]


            if flatten:
                directions = directions.reshape(-1, 3)
                grid = grid.reshape(-1, 2)

            if return_uv:
                return directions, grid
            return directions
        
        # Step 1: read and scale intrinsics (same for all images)
        from datasets.colmap_utils import read_cameras_binary
        camdata = read_cameras_binary(os.path.join(self.hparams.root_dir, 'sparse/0/cameras.bin'))
        
        # Original width and height of the camera
        original_w = camdata[1].width
        original_h = camdata[1].height
        
        # Compute scaling factors from original resolution to new resolution.
        w_factor = w / original_w
        h_factor = h / original_h
        print(f"{w_factor=}")
        print(f"{h_factor=}")

        self.img_wh = (w, h)

        # Read the intrinsic parameters without applying any downsample
        if camdata[1].model == 'SIMPLE_RADIAL':
            original_fx = original_fy = camdata[1].params[0]
            original_cx = camdata[1].params[1]
            original_cy = camdata[1].params[2]
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            original_fx = camdata[1].params[0]
            original_fy = camdata[1].params[1]
            original_cx = camdata[1].params[2]
            original_cy = camdata[1].params[3]
        else:
            raise ValueError(f"Unsupported camera model: {camdata[1].model}")

        # Scale the intrinsic parameters to adapt to the new resolution.
        fx = original_fx * w_factor
        fy = original_fy * h_factor
        cx = original_cx * w_factor
        cy = original_cy * h_factor

        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K, device=self.device)
        

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

def get_initial_pose(starting_angle, radius=1.0, center=np.array([0, 0, 0]), vertical_amplitude=0):
    """
    Generate an orbit of camera poses around the given center.
    Each pose is a 4x4 camera-to-world matrix.
    """
    cam_x = center[0] + radius * np.cos(starting_angle)
    cam_z = center[2] + radius * np.sin(starting_angle)
    cam_y = center[1] + vertical_amplitude * np.sin(starting_angle)
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
    return pose

def get_directories_number(dir):
    # print(f"{dir=}")
    return len(next(os.walk(dir))[1])

def make_dir(exp_info):
    # Create a unique output directory based on a timestamp and experiment index.
    experiment_dir = exp_info['exp_dir']
    os.makedirs(experiment_dir, exist_ok=True)
    exp_id = get_directories_number(experiment_dir) + 1
    exp_info['exp_id'] = exp_id
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(experiment_dir, f"{timestamp}_v{exp_id}")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def render_frame(system, custom_pose, output_dir, frame_name):    
    # Render one frame from the provided custom_pose.
    system.eval()
    with torch.no_grad():
        # batch = {
        #     'pose': torch.from_numpy(custom_pose).unsqueeze(0).float().to(system.device),
        #     'ts': torch.tensor([0], dtype=torch.int64, device=system.device)
        # }
        batch = {
            'pose': torch.from_numpy(custom_pose)[None].to(system.device),
            'ts':   torch.tensor([system.hparams.task_curr], device=system.device)
        }
        results = system(batch, split='test')
    
    w, h_img = system.img_wh
    rgb_image = rearrange(results['rgb'], '(h w) c -> h w c', h=h_img)
    rgb_image = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
    
    frame_fname = os.path.join(output_dir, frame_name)
    imageio.imsave(frame_fname, rgb_image)
    # print(f"Custom frame -> {frame_fname} <- saved.")

def run_experiment(system, custom_pose, experiment_info, output_dir):
    # Render frame from the provided custom_pose.
    render_frame(system, custom_pose, output_dir)
    # Log experiment information.
    write_experiment_log(output_dir, experiment_info)


def interactive_mode(system, initial_pose, output_dir, center=np.array([0,0,0]), move_step=0.1, zoom_step=0.1, yaw_step=0.1, pitch_step=0.1):
    import curses
    import numpy as np

    # current_pose = initial_pose.copy()
    def display_text(stdscr, current_pose, frame_counter, last_key_pressed, frame_name=""):
        stdscr.clear()
        stdscr.addstr(0, 0, "Interactive Mode: WASD to move; ^/v to zoom in/out, q/e to look left/right, r/f to look up/down; ENTER to render frame from current pose; ESC to exit")
        stdscr.addstr(2, 0, f"Current position: {current_pose[:3, 3]}")
        stdscr.addstr(3, 0, f"Current rotation:\n{current_pose[:3, :3]}")
        stdscr.addstr(7, 0, f"Frames rendered this session: {frame_counter}")
        stdscr.addstr(8, 0, f"Last frame rendered: {frame_name}")
        stdscr.addstr(9, 0, f"Last key pressed: [{last_key_pressed}]")
        stdscr.refresh()

    # helper: build a 3×3 rotation from axis (3,) and angle (rad)
    def axis_angle_to_matrix(axis, theta):
        axis = axis/np.linalg.norm(axis)
        a = np.cos(theta/2)
        b, c, d = -axis*np.sin(theta/2)
        return np.array([
            [a*a + b*b - c*c - d*d, 2*(b*c - a*d),     2*(b*d + a*c)],
            [2*(b*c + a*d),         a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
            [2*(b*d - a*c),         2*(c*d + a*b),     a*a + d*d - b*b - c*c],
        ], dtype=np.float32)

    def curses_loop(stdscr):
        curses.curs_set(0)        # hide cursor
        stdscr.nodelay(True)
        stdscr.timeout(100)

        frame_counter       = 0
        current_pose        = initial_pose.copy()
        last_key_pressed    = ""
        last_frame_rendered = ""
        digit_rounding = 4

        while True:
            # redraw status
            display_text(
                stdscr=stdscr,
                current_pose=current_pose,
                frame_counter=frame_counter,
                last_key_pressed=last_key_pressed,
                frame_name=last_frame_rendered
            )

            key = stdscr.getch()
            if key == -1:
                time.sleep(0.05)
                continue

            # —— ROTATION CONTROLS ——
            if key == ord('q'):       # roll left (positive about Z)
                axis = current_pose[:3,:3][:,1]
                R = axis_angle_to_matrix(axis,  yaw_step)
                current_pose[:3,:3] = np.round(R.dot(current_pose[:3,:3]), digit_rounding)
                last_key_pressed = 'q'
            elif key == ord('e'):     # roll right (negative about Z)
                axis = current_pose[:3,:3][:,1]
                R = axis_angle_to_matrix(axis, -yaw_step)
                current_pose[:3,:3] = np.round(R.dot(current_pose[:3,:3]), digit_rounding)
                last_key_pressed = 'e'
            # pitch up/down around camera-right axis
            elif key == ord('f'):     # look up
                right = current_pose[:3,:3][:,0]   # first column
                R = axis_angle_to_matrix(right,  pitch_step)
                current_pose[:3,:3] = np.round(R.dot(current_pose[:3,:3]), digit_rounding)
                last_key_pressed = 'f'
            elif key == ord('r'):     # look down
                right = current_pose[:3,:3][:,0]
                R = axis_angle_to_matrix(right, -pitch_step)
                current_pose[:3,:3] = np.round(R.dot(current_pose[:3,:3]), digit_rounding)
                last_key_pressed = 'r'

            # —— TRANSLATION CONTROLS —— 
            if key == ord('w'):            # forward
                current_pose[1,3] = round(current_pose[1,3] - move_step, digit_rounding)
                last_key_pressed = 'w'
            elif key == ord('s'):          # backward
                current_pose[1,3] = round(current_pose[1,3] + move_step, digit_rounding)
                last_key_pressed = 's'
            elif key == ord('a'):          # left
                current_pose[0,3] = round(current_pose[0,3] - move_step, digit_rounding)
                last_key_pressed = 'a'
            elif key == ord('d'):          # right
                current_pose[0,3] = round(current_pose[0,3] + move_step, digit_rounding)
                last_key_pressed = 'd'     
            elif key == curses.KEY_UP:     # zoom in
                current_pose[2,3] = round(current_pose[2,3] + zoom_step, digit_rounding)
                last_key_pressed = '^'     
            elif key == curses.KEY_DOWN:   # zoom out
                current_pose[2,3] = round(current_pose[2,3] - zoom_step, digit_rounding)
                last_key_pressed = 'v'
            elif key in (10, 13):          # ENTER
                last_key_pressed = 'ENTER'
                frame_counter += 1
                last_frame_rendered = f"custom_frame_{frame_counter:02d}.png"
                display_text(
                    stdscr, current_pose, frame_counter,
                    last_key_pressed, last_frame_rendered
                )
                render_frame(system, current_pose, output_dir, last_frame_rendered)
            elif key == 27:                # ESCAPE, exit session
                break

            time.sleep(0.05)

        return frame_counter


    num_frames_rendered = curses.wrapper(curses_loop)
    return num_frames_rendered

def main():
    # Parse options and update hparams for rendering.
    hparams = get_opts()
    hparams.val_only = True
    if not hparams.weight_path:
        raise ValueError("Please provide a checkpoint path using --weight_path.")
    
    system = NeRFSystem(hparams).to('cuda' if torch.cuda.is_available() else 'cpu')
    # system.setup_from_test()  # Set up directions and intrinsics using the test dataset.

    # width, height = 810, 1440
    width, height = 1080, 1920
    # width, height = 540, 960

    system.setup_intrinsics(width, height)   # my own setup function in order to prevent test dataset loading...
    
    # Load checkpoint weights.
    load_ckpt(system.model, hparams.weight_path)
    print("Checkpoint loaded successfully.")
    
    base_output_dir = "keyboard_rendered_frames"
    experiment_dir = base_output_dir + f"/{hparams.exp_name}"

    starting_angle = 4.71 # (3/2) * pi 
    radius = 1.5
    vertical_amplitude = 0 
    move_step_size = 0.1
    radius_step_size = 0.05

    experiment_info = {
        "base_output_dir": base_output_dir,
        "exp_dir": experiment_dir,
        "exp_name": hparams.exp_name,
        "starting_angle" : starting_angle,
        "downsample" : system.hparams.downsample,
        "Rendered frame width-height" : system.img_wh,
        "Radius": radius,
        "Vertical Amplitued": vertical_amplitude,
        "Task Current": hparams.task_curr,
        "Task Number": hparams.task_number,
        "Vocab Size": hparams.vocab_size,
    }    
    output_dir = make_dir(experiment_info)

    initial_pose = get_initial_pose(starting_angle=starting_angle, radius=radius, center=np.array([0,0,0]), vertical_amplitude=vertical_amplitude)
    # print(initial_pose)

    # Enter interactive mode to adjust the pose with keyboard controls.
    frames_rendered = interactive_mode(system, initial_pose, output_dir, center=np.array([0,0,0]), move_step=move_step_size, zoom_step=radius_step_size)    

    experiment_info['Rendered Frames'] = frames_rendered
    write_experiment_log(output_dir, experiment_info)
    

if __name__ == '__main__':
    main()

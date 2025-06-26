#!/usr/bin/env python
import torch
import os
import imageio
import numpy as np
from einops import rearrange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Imports from original rendering script
from datasets.ray_utils import get_rays
from models.networks import NGPGv2
from models.rendering_NGPA import render
from utils.utils import load_ckpt
from opt_renderer import get_opts
from losses import NeRFLoss
# from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def write_log(output_dir, experiment_info):
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
    print(f"Render log written to {log_file}")

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
        
        # # Original width and height of the camera
        if w == -1 and h == -1:
            w, h = camdata[1].width, camdata[1].height
            original_w, original_h  = camdata[1].width, camdata[1].height
            # Read the intrinsic parameters without applying any downsample
            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0]
                cx = camdata[1].params[1]
                cy = camdata[1].params[2]
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0]
                fy = camdata[1].params[1]
                cx = camdata[1].params[2]
                cy = camdata[1].params[3]
            else:
                raise ValueError(f"Unsupported camera model: {camdata[1].model}")
        else:
            original_w = camdata[1].width
            original_h = camdata[1].height

        # compute a uniform scale so we don't stretch
        orig_aspect = original_w / original_h
        target_aspect = w / h

        if target_aspect > orig_aspect:
            scale = h / original_h
            pad_x = (w - original_w * scale) / 2
            pad_y = 0
        else:
            scale = w / original_w
            pad_x = 0
            pad_y = (h - original_h * scale) / 2

        # print(f"uniform scale = {scale:.4f}, pad_x = {pad_x:.1f}, pad_y = {pad_y:.1f}")

        # now scale intrinsics and shift principal point for the padding
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
 
        # apply the uniform scale and add the padding offset
        fx = original_fx * scale
        fy = original_fy * scale
        cx = original_cx * scale + pad_x
        cy = original_cy * scale + pad_y

        self.img_wh = (w, h)

        self.K = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K, device=self.device)
        

def render_frame(system, custom_pose, output_dir, frame_name):    
    # Render one frame from the provided custom_pose.
    system.eval()
    with torch.no_grad():
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
    print(f"Custom frame -> {frame_fname} <- saved.")


def main():
    # Parse arguments (checkpoint, output settings)
    hparams = get_opts()
    hparams.val_only = True
    hparams.dataset_name = "colmap_ngpa_CLNerf"
    hparams.vocab_size = hparams.task_curr + 1
    hparams.task_number = hparams.vocab_size
    # hparams.scale = 8.0

    if not hparams.weight_path:
        raise ValueError("Please provide --weight_path to a pretrained checkpoint.")
    
    # initialize the system
    system = NeRFSystem(hparams).to('cuda' if torch.cuda.is_available() else 'cpu')
    system.setup_intrinsics(hparams.width, hparams.height)
    load_ckpt(system.model, hparams.weight_path)

    # Load precomputed poses:
    loaded_poses = np.load(hparams.poses_path, allow_pickle=True)   

    # stored_poses=[np.array([[ 0.9635    , -0.2095    ,  0.1662    ,  0.18651286],
    #    [ 0.2675    ,  0.7549    , -0.599     ,  0.3847468 ],
    #    [ 0.        ,  0.6216    ,  0.7832    , -0.9333    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32), np.array([[ 0.5645972 , -0.6465544 ,  0.5130031 , -0.17849863],
    #    [ 0.82529914,  0.4424158 , -0.3510715 ,  0.13555017],
    #    [ 0.        ,  0.6216    ,  0.7832    , -0.9333    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32), np.array([[-0.2955    , -0.6157    ,  0.7306    , -0.41587812],
    #    [ 0.9553    , -0.1903    ,  0.2259    , -0.3408056 ],
    #    [-0.        ,  0.7649    ,  0.6441    , -0.7833    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32), np.array([[-0.9995332 ,  0.018666  , -0.02223311,  0.2723428 ],
    #    [-0.02918967, -0.64416796,  0.7644036 , -0.97983986],
    #    [ 0.        ,  0.7649    ,  0.6441    , -0.7833    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32), np.array([[-0.4274    ,  0.5107    , -0.7459    ,  1.1107528 ],
    #    [-0.904     , -0.2415    ,  0.3528    , -0.64046997],
    #    [-0.        ,  0.8254    ,  0.5645    , -0.8333    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32), np.array([[ 0.68769175,  0.41014358, -0.5991062 ,  0.931634  ],
    #    [-0.725925  ,  0.388483  , -0.56736815,  0.31625235],
    #    [ 0.        ,  0.8254    ,  0.5645    , -0.8333    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32), np.array([[ 0.9160946 ,  0.22648486, -0.33086982,  0.6909284 ],
    #    [-0.40082154,  0.51753396, -0.7558836 ,  0.57034045],
    #    [ 0.        ,  0.8254    ,  0.5645    , -0.8333    ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=np.float32)]
    # print(f"{loaded_poses=}")
    # print(f"{stored_poses=}")
    # print(loaded_poses == stored_poses)

    # make dir for output
    base_output_dir = hparams.base_output_dir
    experiment_dir = os.path.join(base_output_dir, f"render_{hparams.exp_name}")
    os.makedirs(experiment_dir, exist_ok=True)
    dir_num = len(next(os.walk(experiment_dir))[1])
    output_dir = os.path.join(experiment_dir, f"task_{hparams.task_curr:02d}_v{dir_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Render each pose    
    print(f"Rendering {len(loaded_poses)} frames to {output_dir} ...")
    for idx, pose in enumerate(loaded_poses):
        # print(pose)
        frame_name = f"pose_{idx:03d}.png"
        render_frame(system, pose, output_dir, frame_name)
    print("Rendering completed.")

    experiment_info = {
        "base_output_dir": base_output_dir,
        "root_dir": hparams.root_dir,
        "exp_dir": experiment_dir,
        "exp_name": hparams.exp_name,
        "Rendered frame width-height" : system.img_wh,
        "hparams": vars(hparams),
    }    
    write_log(output_dir, experiment_info)


if __name__ == '__main__':
    main()

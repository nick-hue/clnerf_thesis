import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# mine 
from pytorch_lightning.strategies import DDPStrategy 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# torch.cuda.empty_cache()
from pytorch_lightning.profiler import PyTorchProfiler

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGPGv2
from models.rendering_NGPA import render, MAX_SAMPLES
# from models.rendering import render_ori

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils.utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGPGv2(scale=self.hparams.scale, vocab_size=self.hparams.task_curr+1, rgb_act=rgb_act, dim_a = self.hparams.dim_a, dim_g = self.hparams.dim_g)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        
        # for the checkpoint to be loaded once when we have 2 gpus
        load_ckpt(self.model, self.hparams.weight_path)

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            embed_id = batch['ts']
        else:
            poses = batch['pose']
            directions = self.directions
            embed_id = batch['ts'][0].to(self.device) * torch.ones(self.directions.flatten().size(0), dtype=batch['ts'].dtype, device = self.device)
            # print("embed_id = {}".format(embed_id.device))

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, embed_id, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'task_number': self.hparams.task_number,
                  'task_curr': self.hparams.task_curr,
                  'task_split_method': self.hparams.task_split_method,
                  'rep_size': self.hparams.rep_size,
                  'rep_dir': f'results/NGPGv2/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep',
                  'nerf_rep': self.hparams.nerf_rep}

        # os.makedirs(f'results//NGPA/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep', exist_ok=True)
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        # self.test_dataset = dataset(split='test', **kwargs)
        # self.test_dataset = None
        self.rep_dataset = dataset(split='rep', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        # load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.rep_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)
    # def test_dataloader(self):
    #     print("test_dataloader")
    #     return DataLoader(self.rep_dataset,
    #                       num_workers=16,            # number of CPU loader processes
    #                       batch_size=None,              # render 4 images at once (tune up/down)
    #                       pin_memory=True,
    #                       prefetch_factor=4)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):

        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')
        # max_samples = 8192  # default is 1024
        # interval = 2        # default is 16
        # if self.global_step%interval == 0:
        #     self.model.update_density_grid(0.01*max_samples/3**0.5,
        #                                    warmup=self.global_step<self.warmup_steps,
        #                                    erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        # print("start validation")
        if not self.hparams.no_save_test:
            self.val_dir = f'results/NGPGv2/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim, True)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips, True)

    def on_test_start(self):
        torch.cuda.empty_cache()
        self.rep_dir = f'results//NGPGv2/CLNerf/{self.hparams.dataset_name}/{self.hparams.exp_name}/rep'
        os.makedirs(self.rep_dir, exist_ok=True)


    def test_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        fname = batch['fname']

        # => FETCH THE CURRENT POSE
        # batch['pose'] will be a (4×4) tensor (or shaped [1,4,4])
        pose = batch['pose']
        pose_np = pose.detach().cpu().numpy()
        # print(f"[test_step] Saving {fname} with pose:\n{pose_np}")

        txt_path = os.path.join(self.rep_dir, "poses.txt")
        with open(txt_path, "a") as f:
            # write filename then flattened 4×4 pose
            f.write(f"{fname}\t{pose_np.flatten().tolist()}\n")

        results = self(batch, split='test')
        
        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)

        rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        rgb_pred = (rgb_pred*255).astype(np.uint8)
        
        ## tried to save images and render all together at the end
        # print(f"Image rendered {fname}")
        # return {'fname': fname, 'rgb': rgb_pred}

        imageio.imsave(os.path.join(self.rep_dir, fname), rgb_pred)
        return None
    
    ## tried to save images and render all together at the end, maybe could save time
    # def test_epoch_end(self, outputs):
    #     print(f"{outputs[:2]=}")
    #     print(f"Test images saved to {self.rep_dir}")
    #     for out in outputs:
    #         fname = out['fname']
    #         rgb = out['rgb']  # shape (H*W,3)
    #         # w, h = self.train_dataset.img_wh  # or wherever you stored it
    #         # img = rearrange(rgb, '(h w) c -> h w c', h=h).numpy()
    #         # img_uint8 = (img*255).astype(np.uint8)
    #         # imageio.imsave(os.path.join(self.rep_dir, fname), rgb
    #         cv2.imwrite(os.path.join(self.rep_dir, fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    # print(f"{hparams=}")

    # clear cuda cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/NGPGv2_CL/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=False, # was True
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/NGPGv2_CL/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    # Use DDPStrategy for multi-GPU training
    strategy = DDPStrategy(find_unused_parameters=False) if hparams.num_gpus > 1 else None ## added
    # print(f"strategy = {strategy}")

    if hparams.task_curr != hparams.task_number - 1:
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs+1,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=strategy,
                        precision=16,
                        limit_val_batches=0,        # don’t run any validation batches
                        num_sanity_val_steps=0,     # don’t even do sanity‐check val steps at startup
                        val_check_interval=1.0,     # no mid‐epoch validation
                        # limit_test_batches=0,     # don’t run any test batches
                        )
    else:  
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        check_val_every_n_epoch=hparams.num_epochs,
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        accelerator='gpu',
                        devices=hparams.num_gpus,
                        strategy=strategy,
                        precision=16,
                        limit_val_batches=0,    # don’t run any validation batches
                        num_sanity_val_steps=0, # don’t even do sanity‐check val steps at startup
                        val_check_interval=1.0,           # no mid‐epoch validation
                        # limit_test_batches=0,   # don’t run any test batches
                        )
        
    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    # if not hparams.val_only: # save slimmed ckpt for the last epoch
    #     ckpt_ = \
    #         slim_ckpt(f'ckpts/NGPGv2_CL/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
    #                   save_poses=hparams.optimize_ext)
    #     torch.save(ckpt_, f'ckpts/NGPGv2_CL/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    # if last training task, render rgb and depth videos
    if hparams.task_curr == (hparams.task_number -1) and (not hparams.no_save_test): # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)

    # tried to implement dual gpu for rendering
    if hparams.gpu2_render:
        # build a fresh Trainer just for rendering
        render_strategy = DDPStrategy(find_unused_parameters=False)
        render_trainer = Trainer(
            accelerator='gpu',
            devices=2,                         # <- force two GPUs here
            strategy=render_strategy,           # reuse your DDP strategy if any
            precision=16,
            enable_model_summary=False,
            logger=None,
        )
        # render_trainer.test(system, ckpt_path=hparams.ckpt_path)
        render_trainer.test(system)
    else:
        if hparams.task_curr != (hparams.task_number-1):
            trainer.test(system)

    print(f"Experiment : {hparams.exp_name}")
    print(f"Checkpoint saved at : {ckpt_cb.dirpath}")
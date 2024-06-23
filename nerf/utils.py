import os
import glob
import sys

import tqdm
import math
import imageio
import random
import warnings
# import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2

import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
from jittor import Function as F
from nerf.refine_utils import *
from nerf.unet import UNet
import mcubes
from rich.console import Console

# from torch_ema import ExponentialMovingAverage

jt.flags.use_cuda = 1

# import clip
import jittor.transform as T
from contextual_loss.contextual import ContextualLoss
from packaging import version as pver
from copy import deepcopy
# from nerf.provider import NeRFDataset
from transformers import CLIPVisionModelWithProjection, CLIPFeatureExtractor, CLIPTextModelWithProjection, AutoTokenizer


class Resize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def execute(self, img):
        return nn.resize(img, self.size)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        self.mean = jt.array(mean)
        self.std = jt.array(std)
        self.mean = self.mean.view(1, 3, 1, 1)
        self.std = self.std.view(1, 3, 1, 1)

    def execute(self, img):
        return (img - self.mean) / self.std


class CosineSimilarity(nn.Module):
    __constants__ = ['dim', 'eps']

    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def execute(self, x1, x2):
        return self.cosine_similarity(x1, x2, self.dim, self.eps)

    def cosine_similarity(self, x1, x2, dim, eps):
        dot_product = (x1 * x2).sum()

        norm_x1 = jt.norm(x1, p=2)
        norm_x2 = jt.norm(x2, p=2)

        denominator = max(norm_x1, eps) * max(norm_x2, eps)

        similarity_value = dot_product / denominator

        return similarity_value


class PearsonSimilarity(CosineSimilarity):
    def __init__(self, dim=0, eps=1e-8):
        super(PearsonSimilarity, self).__init__(dim, eps)

    def execute(self, x1: jt.Var, x2: jt.Var):
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()
        # x1 = jt.normalize(x1, dim=0)
        # x2 = jt.normalize(x2, dim=0)
        return self.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)


def custom_meshgrid(*args):
    # ref: https://pyjt.org/docs/stable/generated/jt.meshgrid.html?highlight=meshgrid#jt.meshgrid
    if pver.parse(jt.__version__) < pver.parse('1.10'):
        return jt.meshgrid(*args)
    else:
        return jt.meshgrid(*args, indexing='ij')


def safe_normalize(x, eps=1e-20):
    return x / jt.sqrt(jt.clamp(jt.sum(x * x, -1, keepdim=True), min_v=eps, max_v=1e32))


def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(jt.linspace(0, W - 1, W), jt.linspace(0, H - 1, H))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = jt.randint(0, H * W, size=[N])  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = jt.multinomial(error_map, N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + jt.rand(B, N) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + jt.rand(B, N) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = jt.gather(i, -1, inds)
        j = jt.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = jt.arange(H * W).expand([B, H * W])

    zs = jt.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = jt.stack([xs, ys, zs], dim=-1)
    scale = 1 / directions.pow(2).sum(-1).pow(0.5)

    directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['depth_scale'] = scale

    return results


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    jt.set_global_seed(seed)


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(x, jt.array):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def linear_to_srgb(x):
    return jt.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


def srgb_to_linear(x):
    return jt.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 256
    X = jt.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = jt.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = jt.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    # with jt.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = jt.meshgrid(xs, ys, zs)  # for torch < 1.10, should remove indexing='ij'
                pts = jt.concat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [1, N, 3]
                val = query_func(pts).reshape(len(xs), len(ys), len(zs))  # [1, N, 1] --> [x, y, z]
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys),
                zi * N: zi * N + len(zs)] = val.detach().cpu().numpy()
                del val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, use_sdf=False):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    if use_sdf:
        u = - 1.0 * u

    # print(u.mean(), u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    return vertices, triangles


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 depth_model,  # depth model
                 guidance,  # guidance network
                 ref_imgs,
                 ref_depth,
                 ref_mask,
                 ori_imgs=None,
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device
        self.console = Console()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained('./clip-b-16')

        self.tokenizer = AutoTokenizer.from_pretrained('./clip-b-16')
        self.clip_text_model = CLIPTextModelWithProjection.from_pretrained('./clip-b-16')
        self.ref_imgs = ref_imgs
        self.ori_imgs = ori_imgs
        self.depth_prediction = ref_depth
        self.depth_mask = ref_mask

        self.model = model

        self.depth_model = depth_model
        self.depth_model.eval()

        self.depth_transform = T.Compose(
            [
                T.Resize((384, 384)),
                T.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.mean_img = [0.48145466, 0.4578275, 0.40821073]
        self.std_img = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = Normalize(self.mean_img, self.std_img)
        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:

            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()

        else:
            self.text_z = None

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)
        params=[
            {'params':self.model.sigma_net.parameters(),'lr':0.001},
            {'params': self.model.encoder.parameters(), 'lr': 0.01},
        ]
        self.optimizer = jt.nn.Adam(params,lr=0.001)
        if lr_scheduler is None:
            self.lr_scheduler = optim.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        # if ema_decay is not None:
        #     self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        # else:
        #     self.ema = None
        self.ema = None
        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None

        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            self.img_path = os.path.join(self.workspace, 'train')
            os.makedirs(self.img_path, exist_ok=True)

            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        self.pearson = PearsonSimilarity()

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        self.text = []
        self.text_z = []
        self.text.append(self.opt.text)
        self.text_z.append(self.guidance.get_text_embeds([self.opt.text], [self.opt.negative]))

        # # white background
        # white_text = f"{self.opt.text}, white background"
        # self.text.append(white_text)
        # white_text_z = self.guidance.get_text_embeds([white_text], [self.opt.negative])
        # self.text_z.append(white_text_z)

        if self.opt.need_back:
            text = f"{self.opt.text}, back view"
            negative_text = f"{self.opt.negative}"
            # explicit negative dir-encoded text
            if negative_text != '': negative_text += ', '
            negative_text += "face"
            self.text.append(text)
            text_z = self.guidance.get_text_embeds([text], [negative_text])
            self.text_z.append(text_z)
        else:
            self.text.append(self.opt.text)
            self.text_z.append(self.guidance.get_text_embeds([self.opt.text], [self.opt.negative]))

        print(self.text)

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def img_loss(self, rgb1, rgb2):
        # l2_loss = jt.sum(jt.sqrt(1e-8 + jt.sum((rgb1 - rgb2) ** 2, dim=1, keepdims=True)))
        # return l2_loss
        l1_loss = nn.L1Loss()(rgb1, rgb2)
        return l1_loss

    def depth_loss(self, pearson, pred_depth, depth_gt, mask):
        # l2_loss = jt.sum(jt.sqrt(1e-8 + jt.sum((pred_depth*mask - depth_gt*mask) ** 2, dim=1, keepdims=True)))
        # return l2_loss
        pred_depth = pred_depth.squeeze()
        pred_depth = jt.array(pred_depth)
        depth_gt = depth_gt.squeeze().reshape(-1)
        mask = mask.squeeze().reshape(-1)
        pred_depth = pred_depth.reshape(-1)
        mask = (mask == 1)
        co = pearson(pred_depth[mask], depth_gt[mask])
        return 1 - co

    def img_clip_loss(self, rgb1, rgb2):
        rgb1 = nn.resize(rgb1, (224, 224))
        rgb1 = self.normalize(rgb1)
        image_z_1 = self.image_encoder(rgb1)[0]
        rgb2 = nn.resize(rgb2, (224, 224))
        rgb2 = self.normalize(rgb2)
        image_z_2 = self.image_encoder(rgb2)[0]
        image_z_1 = (image_z_1 / image_z_1.norm(dim=(- 1), keepdim=True))
        image_z_2 = (image_z_2 / image_z_2.norm(dim=(- 1), keepdim=True))
        loss = (- (image_z_1 * image_z_2).sum((- 1)).mean())
        return loss

    def img_text_clip_loss(self, rgb, prompt):
        rgb = nn.resize(rgb, (224, 224))
        rgb = self.normalize(rgb)
        image_z_1 = self.image_encoder(rgb)[0]
        image_z_1 = (image_z_1 / image_z_1.norm(dim=(- 1), keepdim=True))
        text = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                              return_tensors='pt')  # .to(self.device)
        text_z = self.clip_text_model(text.input_ids)[0]
        text_z = (text_z / text_z.norm(dim=(- 1), keepdim=True))
        loss = (- (image_z_1 * text_z).sum((- 1)).mean())
        return loss

    def img_cx_loss(self, cx_model, rgb1, rgb2):
        loss = cx_model(rgb1, rgb2)
        return loss

    ### ------------------------------

    def train_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        depth_scale = data['depth_scale']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        # print(data['is_front'])
        self.opt.albedo_iters=1000000
        self.opt.diff_iters=4000
        if self.global_step < self.opt.albedo_iters or data['is_front']:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            rand = random.random()
            if rand > 0.5:
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4:
                shading = 'textureless'
                ambient_ratio = 0.1
            else:
                shading = 'lambertian'
                ambient_ratio = 0.1

        if self.global_step % 10 == 0:
            verbose = True
        else:
            verbose = False

        ref_imgs = self.ref_imgs
        bg_color = jt.rand(3)  # [3], frame-wise random.
        bg_img = bg_color.expand(1, 512, 512, 3).permute(0, 3, 1, 2).contiguous()
        gt_rgb = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + bg_img * (1 - ref_imgs[:, 3:, :, :])

        # _t = time.time()
        outputs = self.model.render(rays_o, rays_d, depth_scale=depth_scale,
                                    bg_color=bg_color, staged=False, perturb=True, ambient_ratio=ambient_ratio,
                                    shading=shading, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        pred_depth = outputs['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()  # [1, 1, H, W]
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)
        if data['is_large']:
            text_z = self.text_z[1]
            text = self.text[1]
        else:
            text_z = self.text_z[0]
            text = self.text[0]
        if self.global_step < self.opt.diff_iters or data['is_front']:
            loss = 0
            de_imgs = None
        else:
            loss, de_imgs = self.guidance.train_step(text_z, pred_rgb, clip_text_model=self.clip_text_model,
                                                     image_encoder=self.image_encoder,
                                                     tokenizer=self.tokenizer, ref_text=text, islarge=data['is_large'],
                                                     ref_rgb=gt_rgb, guidance_scale=self.opt.guidance_scale)
        if self.opt.lambda_opacity > 0:
            loss_opacity = (pred_ws ** 2).mean()
            if data['is_large']:
                loss = loss + self.opt.lambda_opacity * loss_opacity * 10
            else:
                loss = loss + self.opt.lambda_opacity * loss_opacity
        #
        if self.opt.lambda_entropy > 0:
            alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_entropy = (- alphas * jt.log2(alphas) - (1 - alphas) * jt.log2(1 - alphas)).mean()
            if self.global_step < self.opt.diff_iters:
                loss = loss + self.opt.lambda_entropy * loss_entropy
            else:
                loss = loss + self.opt.lambda_entropy * loss_entropy * 10
        verbose=False
        if verbose:
            print(f"loss_entropy: {loss_entropy}, loss_opacity: {loss_opacity}")

        if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.opt.lambda_orient * loss_orient
            if self.global_step < self.opt.diff_iters:
                loss = loss + self.opt.lambda_orient * loss_orient
            else:
                loss = loss + self.opt.lambda_orient * loss_orient * 10

        if self.opt.lambda_smooth > 0 and 'loss_smooth' in outputs:
            loss_smooth = outputs['loss_smooth']
            loss = loss + self.opt.lambda_smooth * loss_smooth

        pred_rgb = jt.nn.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=True)
        pred_depth = jt.nn.interpolate(pred_depth, (512, 512), mode='bilinear', align_corners=True)
        if data['is_front']:
            loss_ref = self.opt.lambda_img * self.img_loss(pred_rgb, gt_rgb)
            loss_depth = self.opt.lambda_depth * self.depth_loss(self.pearson, pred_depth, self.depth_prediction,
                                                                 1 - self.depth_mask)
            if verbose:
                print(f"loss_depth: {loss_depth}, loss_img: {loss_ref}")
            loss_ref += loss_depth

        else:
            loss_ref = self.opt.lambda_clip * self.img_clip_loss(pred_rgb,
                                                                 gt_rgb) + self.opt.lambda_clip * self.img_text_clip_loss(
                pred_rgb, text)

        if self.global_step % 100 == 0 or self.global_step == 1:
            jt.save_image(pred_rgb[0], os.path.join(self.img_path, f'{self.global_step}.png'))
            jt.save_image(gt_rgb[0], os.path.join(self.img_path, f'{self.global_step}_gt.png'))
            jt.save_image(pred_depth[0].repeat(3, 1, 1), os.path.join(self.img_path, f'{self.global_step}_depth.png'))
            jt.save_image((self.depth_prediction * (1 - self.depth_mask))[0].repeat(3, 1, 1),
                          os.path.join(self.img_path, f'{self.global_step}_ref_depth_mask.png'))
            if de_imgs is not None:
                jt.save_image(de_imgs[0], os.path.join(self.img_path, f'{self.global_step}_denoise.png'))

        loss = loss + loss_ref  # loss_depth = 0.01 * self.opt.lambda_img * (self.img_loss(pred_depth, self.depth_prediction) + 1e-2)

        return pred_rgb, pred_ws, loss

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        depth_scale = data['depth_scale']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, bg_color=None, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = 0.0

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        depth_scale = data['depth_scale']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = jt.ones(3)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, depth_scale=depth_scale, staged=True, perturb=perturb,
                                    light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                    bg_color=bg_color, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W)
        if 'normal' in outputs:
            pred_normal = outputs['normal'].reshape(B, H, W, 3)
            return pred_rgb, pred_depth, pred_mask, pred_normal
        else:
            return pred_rgb, pred_depth, pred_mask, None

    def save_mesh(self, save_path=None, resolution=128):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=resolution)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'

        '''if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))'''

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch
            self.train_one_epoch(train_loader)
            # if self.epoch % self.eval_interval == 0:
            #     self.test(valid_loader, write_video=True, write_image=False)
                # self.evaluate_one_epoch(valid_loader)
                # self.save_checkpoint(full=False, best=False)
                # self.save_checkpoint(full=False, best=True)
            self.save_checkpoint(full=False, best=False)  # syh: 怎么保存

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_image=True, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'result')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normal = []

        all_poses = []
        with jt.no_grad():
            for i, data in enumerate(loader):
                preds, preds_depth, preds_mask, preds_normal = self.test_step(data)
                # print(preds_mask)
                mask = (preds_mask > 0.9).int()

                mask = mask[0].detach().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                if preds_normal is not None:
                    preds_normal = preds_normal[0].detach().cpu().numpy()
                    preds_normal = (preds_normal * 255).astype(np.uint8)

                poses = data['poses']
                pose = poses[0].detach().cpu().numpy()
                all_poses.append(pose)

                if write_video:
                    pred_depth_cpu = preds_depth[0].detach().cpu().numpy()
                    pred_depth_cpu = (pred_depth_cpu * 255.).astype(np.uint8)

                    all_preds.append(pred)
                    if preds_normal is not None:
                        all_preds_normal.append(preds_normal)
                    all_preds_depth.append(pred_depth_cpu)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 1000.).astype(np.uint16)

                if write_image:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    if preds_normal is not None:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'),
                                    cv2.cvtColor(preds_normal, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_mask.png'), mask)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            print(os.path.join(save_path, f'{name}_rgb.mp4'))
            all_preds_normal = np.stack(all_preds_normal, axis=0)
            # imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8,
            #                  macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                             macro_block_size=1)

        all_poses = np.stack(all_poses, axis=0)
        np.save(os.path.join(save_path, f'{name}_poses.npy'), all_poses)

        self.log(f"==> Finished Test.")

    def refine(self, load_dir, train_iters, test_loader):

        load_data_folder = load_dir
        outputdir = load_dir.replace("mvimg", "refine")  
        os.makedirs(outputdir,exist_ok=True)
        text = self.opt.text
        image_path = self.opt.ref_path
        fov = self.opt.fov
        H, W = self.opt.H, self.opt.W
        device = 'cuda'
        # Camera intrincs and extrincs
        focal = 1 / (2 * np.tan(np.deg2rad(fov) / 2))
        K = np.array([[focal*W, 0, 0.5*W], [0, focal*H, 0.5*H], [0, 0, 1]])
        cam_files = sorted(glob.glob(load_data_folder+'/*poses.npy'))
        cam_file = cam_files[0]
        cam2world_list = np.load(cam_file)
        cam2world_cano = cam2world_list[(cam2world_list.shape[0]-1)//2]
        image_size = [H, W]
        radius = 2 #8 # points radius in pixel coordinate
        ppp = 8 # points per pixels, for z_buffer
        
        gt_rgb = imageio.imread(image_path)/255.
        gt_rgb = cv2.resize(gt_rgb[:,:,:3],(H, W))
        
        # load images
        depth_files = sorted(glob.glob(load_data_folder+'/*depth.png'))
        mask_files = sorted(glob.glob(load_data_folder+'/*mask.png'))
        rgb_files = sorted(glob.glob(load_data_folder+'/*rgb.png'))
        
        vertices_cano, vertices_color_cano, vertices_novel, vertices_color_novel = load_views(gt_rgb, rgb_files, depth_files, mask_files, cam2world_list, H, W, K, radius, ppp, outputdir, device)
        
        #### colorize point cloud
        # init zeros
        all_v = np.concatenate((vertices_cano, vertices_novel), axis=0)
        all_v_color = np.concatenate((vertices_color_cano, vertices_color_novel), axis=0)
        
        # Save or load
        print("###### Save point cloud ######")
        np.save(outputdir + '/vertices_cano.npy', vertices_cano)
        np.save(outputdir + '/vertices_color_cano.npy', vertices_color_cano)
        np.save(outputdir + '/vertices_novel.npy', vertices_novel)
        np.save(outputdir + '/vertices_color_novel.npy', vertices_color_novel)
        
        # refine stage optimization with SDS loss
        print("###### Optimization with SDS loss ######")
        K = jt.array((K)).float()
        gt_rgb = imageio.imread(image_path)/255.
        gt_mask = cv2.resize(gt_rgb[:,:,3:],(H, W))        
        gt_rgb = cv2.resize(gt_rgb[:,:,:3],(H, W))
        gt_rgb = jt.array(gt_rgb[None,...]).permute(0,3,1,2)
        gt_rgb.requires_grad = False
        
        kernel = np.ones(((5,5)), np.uint8) ##11
        gt_mask = cv2.erode(gt_mask,kernel,iterations=1)
        gt_mask = jt.array(gt_mask).unsqueeze(0).unsqueeze(1)
        gt_mask.requires_grad = False
        
        train_outputdir = outputdir+'/train/'
        os.makedirs(train_outputdir,exist_ok=True)
        
        radius = float(radius) / float(image_size[0]) * 2.0
        unet = UNet(num_input_channels=3+16)
        unet.train()
        cx_model = ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
        
        vertices_cano = jt.array(vertices_cano)
        vertices_cano.requires_grad = False
        vertices_novel = jt.array(vertices_novel)
        vertices_novel.requires_grad = False
        vertices_color_cano = jt.array(vertices_color_cano)
        vertices_color_cano.requires_grad = False
        
        feat_cano = jt.randn((vertices_color_cano.shape[0], 16))
        feat_cano.requires_grad = True
        vertices_color_cano = jt.array(vertices_color_cano)
        vertices_color_cano.requires_grad = True
        vertices_color_novel = jt.array(vertices_color_novel)
        vertices_color_novel.requires_grad = True
        feat_novel = jt.randn((vertices_color_novel.shape[0], 16))
        feat_novel.requires_grad = True
        bg_feat = jt.array(jt.ones((1, 19, 1, 1)))
        bg_feat.requires_grad = True
        
        vertices_color_novel_origin = deepcopy(vertices_color_novel)
        vertices_color_novel_origin.requires_grad = False
        vertices_color_cano_origin = deepcopy(vertices_color_cano)
        vertices_color_cano_origin.requires_grad = False

        point_optimizer = jt.nn.Adam([vertices_color_cano, vertices_color_novel, feat_cano, feat_novel]
                                     + unet.parameters(), 0.001, betas=(0.9, 0.99), eps=1e-15)

        max_pool = jt.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        

        pbar = tqdm.tqdm(range(train_iters))
        for i in pbar:
            rand_c2w, is_front, is_large = fix_poses(i, device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range)
            text_z = self.text_z[0]
            ref_text = self.text[0]
            
            cam2world = rand_c2w[0,:,:]
            world2cam = jt.linalg.inv(cam2world)
            all_v = jt.concat((vertices_cano, vertices_novel),dim=0).float()
            all_xy_cano = jt.concat((vertices_color_cano, feat_cano),dim=-1).float()
            all_xy_novel = jt.concat((vertices_color_novel, feat_novel),dim=-1).float()
            all_v_color = jt.concat((all_xy_cano, all_xy_novel),dim=0).float()
    
            # Unet
            scale = 1
            pred_list = []
            for j in range(3):
                h = H // scale
                w = W // scale
                image_size = (h, w)
                K_ = np.array([[focal*w, 0, 0.5*w], [0, focal*h, 0.5*h], [0, 0, 1]])
                K_ = jt.array((K_)).float()
                pred_rgb = render_point(all_v, all_v_color, h, w, K_, world2cam, image_size, radius, ppp, bg_feat=bg_feat,debug=False)
                scale = scale * 2
                pred_list.append(pred_rgb)
            pred_rgb = unet(pred_list)

            H, W = 800, 800
            image_size = (H, W)
            v_mask_color = jt.ones_like(all_v).float()
            pred_mask = render_point(all_v, v_mask_color, H, W, K, world2cam, image_size, radius, ppp)
            pred_mask_dilate = max_pool(pred_mask)

            if i % 50 == 0:
                jt.save_image(pred_rgb[0], os.path.join(train_outputdir,  f'{i}.png'))
                jt.save_image(pred_mask_dilate[0], os.path.join(train_outputdir,  f'{i}_mask.png'))
            
            if is_front:
                clip_loss = 1000 * self.img_loss(pred_rgb*gt_mask, gt_rgb*gt_mask)
                bg_loss = 0
            else:
                clip_loss, de_imgs = self.guidance.train_step(text_z, pred_rgb, clip_text_model=self.clip_text_model,
                                                              image_encoder=self.image_encoder,
                                                              tokenizer=self.tokenizer,
                                                              ref_text=ref_text, islarge=False, ref_rgb=gt_rgb,
                                                              guidance_scale=5)
                clip_loss += 10 * self.img_clip_loss(pred_rgb, gt_rgb)
                cx_loss = self.img_cx_loss(cx_model, pred_rgb, gt_rgb)
                clip_loss += cx_loss
            
            # background regularization
            bg_loss = 1e-3 * (1 - pred_rgb * (1 - pred_mask_dilate)).sum()
            reg_loss = jt.nn.MSELoss()(vertices_color_novel, vertices_color_novel_origin) * 1e3 + jt.nn.MSELoss()(vertices_color_cano, vertices_color_cano_origin) * 1e5
            loss = clip_loss + reg_loss + bg_loss
            
            pbar.set_description((f"loss: {loss.item():.4f}; reg_loss: {reg_loss.item():.4f}; bg_loss: {bg_loss.item():.4f}"))
            
            point_optimizer.zero_grad()
            point_optimizer.backward(loss)
            point_optimizer.step()

            if i % 1000 == 0:
                jt.save(all_v, outputdir + f'/{i}_v_unet.pt')
                jt.save(all_v_color, outputdir + f'/{i}_v_color_unet.pt')
                jt.save(bg_feat, outputdir + f'/{i}_bg_unet.pt')
                jt.save({'model_state_dict': unet.state_dict(),
                        'optimizer_state_dict': point_optimizer.state_dict(),
                        }, outputdir + f'/{i}_unet.pth')
        
        jt.save(all_v, outputdir + f'/end_v_unet.pt')
        jt.save(all_v_color, outputdir + f'/end_v_color_unet.pt')
        jt.save(bg_feat, outputdir + f'/end_bg_unet.pt')
        jt.save({'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': point_optimizer.state_dict(),
                }, outputdir + f'/end_unet.pth')
        
        # unet.eval()
        # evaluation
        all_transformed_src_alpha = []
        white_bg = jt.ones((1, 19, 1, 1)).to(device)
        white_bg.requires_grad=False
        img_outdir = os.path.join(outputdir, "results")
        os.makedirs(img_outdir, exist_ok=True)
        
        # test
        print("###### Finel Refine Rendering ######")
        pbar = tqdm.tqdm(total=len(test_loader) * test_loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for i, data in enumerate(test_loader):
            cam2world = data['poses'][0].detach().cpu().numpy()
            world2cam = np.linalg.inv(cam2world)
            world2cam = jt.array(world2cam).to(device)
            scale = 1
            pred_list = []
            for j in range(3):
                h = H // scale
                w = W // scale
                image_size = (h, w)
                K_ = np.array([[focal*w, 0, 0.5*w], [0, focal*h, 0.5*h], [0, 0, 1]])
                K_ = jt.array((K_)).float()
                pred_rgb = render_point(all_v, all_v_color, h, w, K_, world2cam, image_size, radius, ppp, bg_feat=bg_feat)
                scale = scale * 2
                pred_list.append(pred_rgb)
            pred_rgb = unet(pred_list)
            transformed_src_alpha = np.array(pred_rgb[0].permute(1,2,0).detach().cpu().numpy() * 255,dtype=np.uint8)
            jt.save_image(pred_rgb[0], img_outdir+f'/render_unet_{i:04d}.png')
            all_transformed_src_alpha.append(transformed_src_alpha)
            pbar.update(test_loader.batch_size)
        
        all_transformed_src_alpha = np.stack(all_transformed_src_alpha, axis=0)
        imageio.mimwrite(img_outdir+'/render_unet_img_clip.mp4', all_transformed_src_alpha, fps=25, quality=8, macro_block_size=1)

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        # sys.exit()
        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pyjt.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            # print(self.global_step, '--------------------------------')
            # if self.global_step>1:
            #     for name,param in self.model.named_parameters():
            #         print(name,param.mean())
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1
            pred_rgbs, pred_ws, loss = self.train_step(data)
            self.optimizer.backward(loss)
            self.optimizer.clip_grad_norm(max_norm=10)
            self.optimizer.step()
            # if self.scheduler_update_every_step:
            #     self.lr_scheduler.step()

            loss_val = loss.detach().item()
            total_loss += loss_val
            if self.local_rank == 0:
                # if self.use_tensorboardX:
                #     self.writer.add_scalar("train/loss", loss_val, self.global_step)
                #     self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)
            # return
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, jt.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        with jt.no_grad():
            for data in loader:
                self.local_step += 1
                preds, preds_depth, loss = self.eval_step(data)
                # save image
                save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pred_depth = preds_depth.reshape(1, self.opt.H, self.opt.W, 1).permute(0, 3, 1,2).contiguous()  # [1, 1, H, W]

                preds = preds.reshape(1, self.opt.H, self.opt.W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 1, H, W]

                jt.save_image(pred_depth[0], save_path_depth)
                jt.save_image(preds[0], save_path)

                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)
            state['model']['density_bitfield'] = state['model']['density_bitfield'].float()  # syh: 方便保存
            jt.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    jt.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return
        checkpoint_dict = jt.load(checkpoint)
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return
        for idx,key in enumerate(checkpoint_dict['model'].keys()):
            params=checkpoint_dict['model'][key]
            if idx == 3:
                checkpoint_dict['model'][key]=params.uint8()


        self.model.load_state_dict(checkpoint_dict['model'])
        self.log("[INFO] loaded model.")
        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

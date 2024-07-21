import argparse

import jittor as jt
import jittor.transform as T
import torch
import torchvision.transforms as TT
from jittor import optim
from jittor import Function as F
from jittor import lr_scheduler

from nerf.provider import NeRFDataset
from nerf.utils import *


from scipy.ndimage import median_filter
from PIL import Image

jt.flags.use_cuda = 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='a teddy bear with a black bow', help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--final', action='store_true', help="final train mode")
    parser.add_argument('--refine', action='store_true', help="refine mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--depth_model', type=str, default='dpt_hybrid', help='choose from [dpt_large, dpt_hybrid]')
    parser.add_argument('--guidance_scale', type=float, default=10)
    parser.add_argument('--need_back', action='store_true', help="use back text prompt")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--ref_path', default=None, type=str, help="use image as referance, only support alpha image")


    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--refine_iters', type=int, default=3000, help="refine iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--no_cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--diff_iters', type=int, default=400, help="training iters that only use albedo shading")
    parser.add_argument('--step_range', type=float, nargs='*', default=[0.2, 0.6])
    
    # model options
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the gaussian density blob")
    parser.add_argument('--blob_radius', type=float, default=0.1, help="control the radius for the gaussian density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='tcnn', choices=['grid', 'tcnn', 'sdf', 'vanilla', 'normal'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=96, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=96, help="render height for NeRF in training")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fov', type=float, default=20, help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[15, 25], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[70, 110], help="training camera phi range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[0, 360], help="training camera phi range")
    
    parser.add_argument('--lambda_entropy', type=float, default=1, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=1e-3, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=1, help="loss scale for surface smoothness")
    parser.add_argument('--lambda_img', type=float, default=1e3, help="loss scale for ref loss")
    parser.add_argument('--lambda_depth', type=float, default=1, help="loss scale for depth loss")
    parser.add_argument('--lambda_clip', type=float, default=1, help="loss scale for clip loss")
    
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    parser.add_argument('--max_depth', type=float, default=10.0, help="farthest depth")
    
    opt = parser.parse_args()
    opt.cuda_ray = True
    if opt.no_cuda_ray:
        opt.cuda_ray = False
    optDict = opt.__dict__
    opt.workspace = os.path.join('results', opt.workspace)
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True)

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'tcnn':
        from nerf.network_tcnn import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)
    seed_everything(opt.seed)

    # load depth network
    device = 'cpu'
    if jt.flags.use_cuda == 1:
        device = 'cuda'
    

    if opt.optim == 'adan':
        from optimizer import Adan

        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.parameters(), lr=5 * opt.lr, eps=1e-8, weight_decay=2e-5,
                                       max_grad_norm=5.0, foreach=False)
    else:  # adam
        optimizer = lambda model: optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        warm_up_with_cosine_lr = lambda iter: iter / 100
        scheduler = lambda optimizer: optim.LambdaLR(optimizer, lambda iter: 1)
        # scheduler = lambda optimizer: optim.LambdaLR(optimizer, warm_up_with_cosine_lr)
    else:
        scheduler = lambda optimizer: optim.LambdaLR(optimizer, lambda iter: 1)  # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    if opt.guidance == 'stable-diffusion':
        from nerf.sd import StableDiffusion

        guidance = StableDiffusion(device, opt.sd_version, opt.hf_key)  # , step_range=opt.step_range
    elif opt.guidance == 'clip':
        from nerf.clip import CLIP

        guidance = CLIP(device)
    else:
        raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

    ref_imgs = cv2.imread(opt.ref_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
    image_pil = Image.open(opt.ref_path).convert("RGB")

    # generated caption
    if opt.text == None:
        with open(os.path.join(opt.workspace, 'preprocess', 'prompt.txt'), 'r') as f:
            caption = f.readline()
        opt.text = caption

    with open(os.path.join(opt.workspace, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in optDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    # only support alpha photo input.
    imgs = cv2.cvtColor(ref_imgs, cv2.COLOR_BGRA2RGBA)
    imgs = cv2.resize(imgs, (512, 512), interpolation=cv2.INTER_AREA)
    # ref_imgs = (torch.from_numpy(imgs)/255.).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    ref_imgs = jt.unsqueeze((jt.array(imgs) / 255.), 0)
    ref_imgs = ref_imgs.permute(0, 3, 1, 2)
    ori_imgs = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + (1 - ref_imgs[:, 3:, :, :])

    mask = imgs[:, :, 3:]
    # mask[mask < 0.5 * 255] = 0
    # mask[mask >= 0.5 * 255] = 1
    kernel = np.ones(((5, 5)), np.uint8)  ##11
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = (mask == 0)
    mask = (jt.array(mask)).unsqueeze(0).unsqueeze(0).to(device)
    depth_mask = mask
    
    # depth load
    disparity = imageio.imread(os.path.join(opt.workspace, 'preprocess','depth.png')) / 65535.
    disparity = median_filter(disparity, size=5)
    depth = 1. / np.maximum(disparity, 1e-2)

    depth_prediction = jt.array(depth)
    # normalize estimated depth
    depth_prediction = depth_prediction * (1 - depth_mask) + jt.ones_like(depth_prediction) * (depth_mask)
    depth_prediction = depth_prediction.float()
    rays_alive = jt.arange(10, dtype=jt.int32)  # [N]
    # Todo:depth_prediction
    depth_prediction = ((depth_prediction - 1.0) / (depth_prediction.max() - 1.0)) * 0.9 + 0.1
    # save_image(ori_imgs, os.path.join(opt.workspace, opt.text.replace(" ", "_") + '_ref.png'))
    model = NeRFNetwork(opt)
    for name, param in model.named_parameters():
        if 'sigma_net' in name:
            param.requires_grad = True
    trainer = Trainer('df', opt, model, guidance,
                      ref_imgs=ref_imgs, ref_depth=depth_prediction,
                      ref_mask=depth_mask, ori_imgs=ori_imgs,
                      device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16,
                      lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval,
                      scheduler_update_every_step=True)

    if opt.test:
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=33).dataloader()
        print("Prepare in Test")
        trainer.test(test_loader, write_video=True)
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)

    if opt.refine:
        mv_loader = NeRFDataset(opt, device=device, type='gen_mv', H=opt.H, W=opt.W, size=33).dataloader()
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader()
        trainer.test(mv_loader, save_path=os.path.join(opt.workspace, 'mvimg'), write_image=True, write_video=False)
        trainer.refine(os.path.join(opt.workspace, 'mvimg'), opt.refine_iters, test_loader)

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
        # valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
        # max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        # trainer.train(train_loader, valid_loader, max_epoch)
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=33).dataloader()
        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, test_loader, max_epoch)

        # also test
        if opt.final:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader()
            trainer.test(test_loader, write_image=False, write_video=True)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)

        if opt.refine:
            mv_loader = NeRFDataset(opt, device=device, type='gen_mv', H=opt.H, W=opt.W, size=33).dataloader()
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader()
            trainer.test(mv_loader, save_path=os.path.join(opt.workspace, 'mvimg'), write_image=True, write_video=False)
            trainer.refine(os.path.join(opt.workspace, 'mvimg'), opt.refine_iters, test_loader)


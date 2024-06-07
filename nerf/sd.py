import jittor as jt
from jittor import init
import jittor.transform as T
from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPVisionModel, CLIPFeatureExtractor
from diffusers import PNDMScheduler, DDIMScheduler
from JDiffusion.models import AutoencoderKL, UNet2DConditionModel
from JDiffusion.pipelines.pipeline_output_jittor import StableDiffusionPipelineOutput
logging.set_verbosity_error()
from JDiffusion.utils import randn_tensor
from jittor import nn
import time
import os
import clip
from jittor import Function


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class backward_fun(Function):
    def execute(self, x, grad):
        self.grad = grad
        return jt.Var(1.0)

    def grad(self, g):
        grad = self.grad
        return g * (grad)

backward_fun = backward_fun.apply




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
        # return nn.normalize(img, self.mean, self.std)


class StableDiffusion(nn.Module):

    def __init__(self, device,sd_version='2.0', hf_key=None, step_range=[0.2, 0.6]):
        super().__init__()
        # self.device = device
        self.sd_version = sd_version
        print(f'[INFO] loading stable diffusion...')
        print(hf_key)
        if (self.sd_version == '2.0'):
            model_key = 'stabilityai/stable-diffusion-2'
        elif (hf_key is not None):
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif (self.sd_version == '2.0'):
            model_key = 'stabilityai/stable-diffusion-2'
        elif (self.sd_version == '1.5'):
            model_key = 'runwayml/stable-diffusion-v1-5'
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder='vae')# .to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder='text_encoder')# .to(self.device)
        self.mean_img = [0.48145466, 0.4578275, 0.40821073]
        self.std_img = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = Normalize(self.mean_img, self.std_img)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder='unet')# .to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder='scheduler')
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.num_inference_steps = 50
        self.min_step = int((self.num_train_timesteps * float(step_range[0])))
        self.max_step = int((self.num_train_timesteps * float(step_range[1])))
        self.alphas = self.scheduler.alphas_cumprod# .to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod# .to(self.device)
        self.ref_imgs = None
        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt=''):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with jt.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]# .to(self.device))[0]
            prompt_embeds_dtype = self.text_encoder.dtype
            text_embeddings = text_embeddings.to(dtype=prompt_embeds_dtype)
            # bs_embed, seq_len, _ = prompt_embeds.shape

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with jt.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0] # .to(self.device))[0]
        text_embeddings = jt.concat([uncond_embeddings, text_embeddings])
        return text_embeddings



    def img_clip_loss(self, image_encoder, rgb1, rgb2):
        rgb1 = nn.resize(rgb1, (224, 224))
        rgb1 = self.normalize(rgb1)
        image_z_1 = image_encoder(rgb1)[0]
        rgb2 = nn.resize(rgb2, (224, 224))
        rgb2 = self.normalize(rgb2)
        image_z_2 = image_encoder(rgb2)[0]
        image_z_1 = (image_z_1 / image_z_1.norm(dim=(- 1), keepdim=True))
        image_z_2 = (image_z_2 / image_z_2.norm(dim=(- 1), keepdim=True))
        loss = (- (image_z_1 * image_z_2).sum((- 1)).mean())
        print("aaa")
        return loss

    def img_text_clip_loss(self, tokenizer, clip_text_model, image_encoder,rgb, prompt):
        rgb = nn.resize(rgb, (224, 224))
        rgb = self.normalize(rgb)
        image_z_1 = image_encoder(rgb)[0]
        image_z_1 = (image_z_1 / image_z_1.norm(dim=(- 1), keepdim=True))
        text = tokenizer(prompt,padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt' )# .to(self.device)
        text_z = clip_text_model(text.input_ids)[0]
        text_z = (text_z / text_z.norm(dim=(- 1), keepdim=True))
        loss = (- (image_z_1 * text_z).sum((- 1)).mean())
        return loss

    def train_step(self, text_embeddings, pred_rgb, ref_rgb=None, noise=None, islarge=False, ref_text=None, clip_text_model=None, image_encoder=None, tokenizer=None, guidance_scale=10):
        loss = 0
        imgs = None
        pred_rgb_512 = nn.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        t = jt.randint(self.min_step, (self.max_step + 1), [1], dtype='int32')
        w_ = 1.0
        latents = self.encode_imgs(pred_rgb_512)
        with jt.no_grad():
            # noise = randn_tensor(latents.shape, seed, dtype=latents.dtype)
            noise = jt.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = jt.concat(([latents_noisy] * 2))
            latent_model_input = latent_model_input.detach().start_grad()
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
            noise_pred = (noise_pred_text + (guidance_scale * (noise_pred_text - noise_pred_uncond)))
        if ((not islarge) and ((t / self.num_train_timesteps) <= 0.4)):
            self.scheduler.set_timesteps(self.num_train_timesteps)
            de_latents = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']
            imgs = self.decode_latents(de_latents)
            loss = ((10 * self.img_clip_loss(image_encoder, imgs, ref_rgb)) + (10 * self.img_text_clip_loss(tokenizer, clip_text_model,image_encoder, imgs, ref_text)))
        else:
            w = (1 - self.alphas[t])
            grad = ((w * (noise_pred - noise)) * w_)
            imgs = None
            # grad = torch.nan_to_num(grad)


            loss = backward_fun(latents, grad)
            # latents.backward(gradient=grad, retain_graph=True)
            # optimizer.backward(loss)
        return (loss, imgs)

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if (latents is None):
            latents = jt.randn(((text_embeddings.shape[0] // 2), self.unet.in_channels, (height // 8), (width // 8)))
        self.scheduler.set_timesteps(num_inference_steps)
        # with torch.autocast('cuda'):
        for (i, t) in enumerate(self.scheduler.timesteps):
            latent_model_input = jt.concat(([latents] * 2))
            with jt.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
            (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
            noise_pred = (noise_pred_text + (guidance_scale * (noise_pred_text - noise_pred_uncond)))
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents

    def decode_latents(self, latents):
        latents = ((1 / 0.18215) * latents)
        with jt.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = jt.clamp(((imgs / 2) + 0.5), 0, 1)
        return imgs

    def encode_imgs(self, imgs):
        imgs = ((2 * imgs) - 1)
        if self.vae.config.force_upcast:
            imgs = imgs.float()
            self.vae.to(dtype=jt.float32)
        posterior = self.vae.encode(imgs).latent_dist
        latents = (posterior.sample() * 0.18215)
        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeds = self.get_text_embeds(prompts, negative_prompts)
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        imgs = self.decode_latents(latents)
        return imgs


if (__name__ == '__main__'):
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--workspace', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help='stable diffusion version')
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seeds', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()
    opt.workspace = os.path.join('test_bench', opt.workspace)
    if (opt.workspace is not None):
        os.makedirs(opt.workspace, exist_ok=True)
    jt.flags.use_cuda = jt.has_cuda

    # device = torch.device('cuda')
    sd = StableDiffusion(opt.sd_version)
    t_embed = sd.get_text_embeds('a teddy bear', '')
    print(2)
    print(t_embed.shape)
    for seed in range(opt.seeds):
        seed_everything(seed)
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, guidance_scale=7.5)
        save_image(imgs, os.path.join(opt.workspace, (opt.prompt.replace(' ', '_') + f'_{seed}.png')))
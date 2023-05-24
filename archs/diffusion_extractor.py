from PIL import Image
import torch

from diffusers import DDIMScheduler
from archs.stable_diffusion.diffusion import (
    init_models, 
    get_tokens_embedding,
    generalized_steps,
    collect_and_resize_feats
)
from archs.stable_diffusion.resnet import init_resnet_func

class DiffusionExtractor:
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """
    def __init__(self, config, device):
        self.device = device
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.num_timesteps = config["num_timesteps"]
        self.scheduler.set_timesteps(self.num_timesteps)
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0))
        self.batch_size = config.get("batch_size", 1)

        self.unet, self.vae, self.clip, self.clip_tokenizer = init_models(device=self.device, model_id=config["model_id"])
        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")
        self.change_cond(self.prompt, "cond")
        self.change_cond(self.negative_prompt, "uncond")
        
        self.diffusion_mode = config.get("diffusion_mode", "generation")
        if "idxs" in config and config["idxs"] is not None:
            self.idxs = config["idxs"]
        else:
            self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.output_resolution = config["output_resolution"]

        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep = config.get("save_timestep", [])

        print(f"diffusion_mode: {self.diffusion_mode}")
        print(f"idxs: {self.idxs}")
        print(f"output_resolution: {self.output_resolution}")
        print(f"prompt: {self.prompt}")
        print(f"negative_prompt: {self.negative_prompt}")

    def change_cond(self, prompt, cond_type="cond"):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                new_cond = new_cond.expand((self.batch_size, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError

    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent,
            self.unet, 
            self.scheduler, 
            run_inversion=False, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond, 
            min_i=min_i,
            max_i=max_i
        )
        return xs
    
    def run_inversion(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent, 
            self.unet, 
            self.scheduler, 
            run_inversion=True, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond,
            min_i=min_i,
            max_i=max_i
        )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = []
            for timestep in self.save_timestep:
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)
            feats = torch.stack(feats, dim=1)
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs

    def latents_to_images(self, latents):
        latents = latents.to(self.device)
        latents = latents / 0.18215
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    def forward(self, images=None, latents=None, guidance_scale=-1, preview_mode=False):
        if images is None:
            if latents is None:
                latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device, generator=self.generator)
            if self.diffusion_mode == "generation":
                if preview_mode:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
                else:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale)
            elif self.diffusion_mode == "inversion":
                raise NotImplementedError
        else:
            images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215
            if self.diffusion_mode == "inversion":
                extractor_fn = lambda latents: self.run_inversion(latents, guidance_scale)
            elif self.diffusion_mode == "generation":
                raise NotImplementedError
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)
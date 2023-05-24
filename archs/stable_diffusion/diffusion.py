import numpy as np
import os
from PIL import Image
import PIL
import torch
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler, 
    StableDiffusionPipeline
)
from transformers import (
    CLIPModel, 
    CLIPTextModel, 
    CLIPTokenizer
)
from archs.stable_diffusion.resnet import set_timestep, collect_feats

"""
Functions for running the generalized diffusion process 
(either inversion or generation) and other helpers 
related to latent diffusion models. Adapted from 
Shape-Guided Diffusion (Park et. al., 2022).
https://github.com/shape-guided-diffusion/shape-guided-diffusion/blob/main/utils.py
"""

def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
  tokens = clip_tokenizer(
    prompt,
    padding="max_length",
    max_length=clip_tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
    return_overflowing_tokens=True,
  )
  input_ids = tokens.input_ids.to(device)
  embedding = clip(input_ids).last_hidden_state
  return tokens, embedding

def latent_to_image(vae, latent):
  latent = latent / 0.18215
  image = vae.decode(latent.to(vae.dtype)).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  image = (image[0] * 255).round().astype("uint8")
  image = Image.fromarray(image)
  return image

def image_to_latent(vae, image, generator=None, mult=64, w=512, h=512):
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32)
  # remove alpha channel
  if len(image.shape) == 2:
    image = image[:, :, None]
  else:
    image = image[:, :, (0, 1, 2)]
  # (b, c, w, h)
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  image = image / 255.0
  image = 2. * image - 1.
  image = image.to(vae.device)
  image = image.to(vae.dtype)
  return vae.encode(image).latent_dist.sample(generator=generator) * 0.18215

def get_xt_next(xt, et, at, at_next, eta):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
  if eta == 0:
    c1 = 0
  else:
    c1 = (
      eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    )
  c2 = ((1 - at_next) - c1 ** 2).sqrt()
  xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
  return x0_t, xt_next

def generalized_steps(x, model, scheduler, **kwargs):
  """
  Performs either the generation or inversion diffusion process.
  """
  seq = scheduler.timesteps
  seq = torch.flip(seq, dims=(0,))
  b = scheduler.betas
  b = b.to(x.device)

  with torch.no_grad():
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    if kwargs.get("run_inversion", False):
      seq_iter = seq_next
      seq_next_iter = seq
    else:
      seq_iter = reversed(seq)
      seq_next_iter = reversed(seq_next)

    x0_preds = [x]
    xs = [x]
    for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
      max_i = kwargs.get("max_i", None)
      min_i = kwargs.get("min_i", None)
      if max_i is not None and i >= max_i:
        break
      if min_i is not None and i < min_i:
        continue
      
      t = (torch.ones(n) * t).to(x.device)
      next_t = (torch.ones(n) * next_t).to(x.device)
      if t.sum() == -t.shape[0]:
        at = torch.ones_like(t)
      else:
        at = (1 - b).cumprod(dim=0).index_select(0, t.long())
      if next_t.sum() == -next_t.shape[0]:
        at_next = torch.ones_like(t)
      else:
        at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
      
      # Expand to the correct dim
      at, at_next = at[:, None, None, None], at_next[:, None, None, None]

      if kwargs.get("run_inversion", False):
        set_timestep(model, len(seq_iter) - i - 1)
      else:
        set_timestep(model, i)

      xt = xs[-1].to(x.device)
      cond = kwargs["conditional"]
      guidance_scale = kwargs.get("guidance_scale", -1)
      if guidance_scale == -1:
        et = model(xt, t, encoder_hidden_states=cond).sample
      else:
        # If using Classifier-Free Guidance, the saved feature maps
        # will be from the last call to the model, the conditional prediction
        uncond = kwargs["unconditional"]
        et_uncond = model(xt, t, encoder_hidden_states=uncond).sample
        et_cond = model(xt, t, encoder_hidden_states=cond).sample
        et = et_uncond + guidance_scale * (et_cond - et_uncond)
      
      eta = kwargs.get("eta", 0.0)
      x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta)

      x0_preds.append(x0_t)
      xs.append(xt_next.to('cpu'))

    return x0_preds

def freeze_weights(weights):
  for param in weights.parameters():
    param.requires_grad = False

def init_models(
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    freeze=True
  ):
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    revision="fp16",
    torch_dtype=torch.float16,
  )
  unet = pipe.unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  unet.to(device)
  vae.to(device)
  clip.to(device)
  if freeze:
    freeze_weights(unet)
    freeze_weights(vae)
    freeze_weights(clip)
  return unet, vae, clip, clip_tokenizer

def collect_and_resize_feats(unet, idxs, timestep, resolution=-1):
  latent_feats = collect_feats(unet, idxs=idxs)
  latent_feats = [feat[timestep] for feat in latent_feats]
  if resolution > 0:
      latent_feats = [torch.nn.functional.interpolate(latent_feat, size=resolution, mode="bilinear") for latent_feat in latent_feats]
  latent_feats = torch.cat(latent_feats, dim=1)
  return latent_feats
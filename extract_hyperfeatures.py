import argparse
import glob
import json
import os
from omegaconf import OmegaConf
from PIL import Image
import random
import torch
from tqdm import tqdm

from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork
from archs.stable_diffusion.resnet import collect_dims
from archs.correspondence_utils import process_image

def load_models(config_path, device="cuda"):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    weights = torch.load(config["weights_path"], map_location="cpu")
    config.update(weights["config"])
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]

    # dims is the channel dim for each layer (12 dims for Layers 1-12)
    # idxs is the (block, sub-block) index for each layer (12 idxs for Layers 1-12)
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=dims,
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"]
    )
    aggregation_network.load_state_dict(weights["aggregation_network"])
    return config, diffusion_extractor, aggregation_network

def extract_hyperfeats(config, diffusion_extractor, aggregation_network, images_or_prompts, save_root, device="cuda", load_size=(512, 512)):
  b = config["batch_size"]
  w = h = config["output_resolution"]
  meta_file = []
  for i in tqdm(range(0, len(images_or_prompts), b)):
    with torch.inference_mode():
      with torch.autocast("cuda"):
        if config["diffusion_mode"] == "inversion":
            imgs, save_names = [], []
            for path in images_or_prompts[i:i+b]:
                img_pil = Image.open(path).convert("RGB")
                img, _ =  process_image(img_pil, res=load_size)
                img = img.to(device)
                imgs.append(img)
                save_names.append(os.path.basename(path).split(".")[0])
            imgs = torch.vstack(imgs)
            feats, _ = diffusion_extractor.forward(imgs)
            diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))
        elif config["diffusion_mode"] == "generation":
          prompt = images_or_prompts[i]
          negative_prompt = config["negative_prompt"]
          diffusion_extractor.change_cond(prompt, "cond")
          diffusion_extractor.change_cond(negative_prompt, "uncond")
          latents = torch.randn((diffusion_extractor.batch_size, diffusion_extractor.unet.in_channels, 512 // 8, 512 // 8), device=diffusion_extractor.device, generator=diffusion_extractor.generator)
          feats, outputs = diffusion_extractor.forward(latents=latents, guidance_scale=config["guidance_scale"])
          diffusion_hyperfeats = aggregation_network(torch.flip(feats, dims=(1,)).float().view(b, -1, w, h))
          save_names = []
          for img in diffusion_extractor.latents_to_images(outputs[-1]):
            hash_name = str(random.getrandbits(32))
            img.save(f"{save_root}/{hash_name}.png")
            save_names.append(hash_name)
            meta_file.append({"source_path": f"{hash_name}.png", "prompt": prompt})
    for j, img_hyperfeats in enumerate(diffusion_hyperfeats):
      torch.save(img_hyperfeats.detach().cpu(), f"{save_root}/{save_names[j]}.pt")
    json.dump(meta_file, open(f"{save_root}/meta.json", "w"))

def pad_to_batch_size(items, batch_size):
    remainder = len(items) % batch_size
    if remainder == 0:
        return items

    padding_length = batch_size - remainder
    padding = items[-1:] * padding_length  # Duplicate the last element
    padded_items = items + padding
    return padded_items

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/real.yaml") # "configs/real.yaml"
    parser.add_argument("--images_or_prompts_path", type=str, help="Path to json with image paths (real) or prompts (synthetic)", default="annotations/spair_71k_test-6.json") # "annotations/synthetic-3.json"
    parser.add_argument("--save_root", type=str, help="Root directory to save the results", default="hyperfeatures")
    parser.add_argument("--image_root", type=str, help="Root directory storing images (real)", default="assets/spair/images") # ""
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)

    batch_size = config["batch_size"]
    images_or_prompts = []
    if args.images_or_prompts_path:
        for ann in json.load(open(args.images_or_prompts_path)):
            if config["diffusion_mode"] == "inversion":
                if type(ann) is dict:
                    images_or_prompts.append(f"{args.image_root}/{ann['source_path']}")
                    images_or_prompts.append(f"{args.image_root}/{ann['target_path']}")
                else:
                    images_or_prompts.append(ann)
            elif config["diffusion_mode"] == "generation":
                images_or_prompts.extend([ann] * batch_size)
    else:
        images_or_prompts = glob.glob(f"{args.image_root}/*")

    # Duplicate the last item in the list until images_or_prompts is a multiple of batch_size
    images_or_prompts = pad_to_batch_size(images_or_prompts, batch_size)
    extract_hyperfeats(config, diffusion_extractor, aggregation_network, images_or_prompts, args.save_root)

if __name__ == "__main__":
    main()
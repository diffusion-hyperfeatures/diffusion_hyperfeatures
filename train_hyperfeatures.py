import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
import wandb

from archs.correspondence_utils import (
    load_image_pair,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
)
from archs.stable_diffusion.resnet import collect_dims
from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork

def get_rescale_size(config):
    output_size = (config["output_resolution"], config["output_resolution"])
    if "load_resolution" in config:
        load_size = (config["load_resolution"], config["load_resolution"])
    else:
        load_size = output_size
    return output_size, load_size

def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    wandb.log({f"mixing_weights": plt})

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
        with torch.autocast("cuda"):
            feats, _ = diffusion_extractor.forward(imgs)
            b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))
    img1_hyperfeats = diffusion_hyperfeats[0][None, ...]
    img2_hyperfeats = diffusion_hyperfeats[1][None, ...]
    return img1_hyperfeats, img2_hyperfeats

def compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size):
    # Assumes hyperfeats are batch_size=1 to avoid complex indexing
    # Compute in both directions for cycle consistency
    source_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img1_hyperfeats, img2_hyperfeats)
    target_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img2_hyperfeats, img1_hyperfeats)
    source_idx = torch.from_numpy(points_to_idxs(source_points, output_size)).long().to(source_logits.device)
    target_idx = torch.from_numpy(points_to_idxs(target_points, output_size)).long().to(target_logits.device)
    loss_source = torch.nn.functional.cross_entropy(source_logits[0, source_idx], target_idx)
    loss_target = torch.nn.functional.cross_entropy(target_logits[0, target_idx], source_idx)
    loss = (loss_source + loss_target) / 2
    return loss

def save_model(config, aggregation_network, optimizer, step):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    results_folder = f"{config['results_folder']}/{wandb.run.name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    torch.save(dict_to_save, f"{results_folder}/checkpoint_step_{step}.pt")

def validate(config, diffusion_extractor, aggregation_network, val_anns):
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    plot_every_n_steps = config.get("plot_every_n_steps", -1)
    pck_threshold = config["pck_threshold"]
    ids, val_dist, val_pck_img, val_pck_bbox = [], [], [], []
    for j, ann in tqdm(enumerate(val_anns)):
        with torch.no_grad():
            source_points, target_points, img1_pil, img2_pil, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"])
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            wandb.log({"val/loss": loss.item()}, step=j)
            # Loss NN correspondences
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            predicted_points = predicted_points.detach().cpu().numpy()
            # Rescale to the original image dimensions
            target_size = ann["target_size"]
            predicted_points = rescale_points(predicted_points, load_size, target_size)
            target_points = rescale_points(target_points, load_size, target_size)
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["target_bounding_box"])
            wandb.log({"val/sample_pck_img": sample_pck_img}, step=j)
            wandb.log({"val/sample_pck_bbox": sample_pck_bbox}, step=j)
            val_dist.append(dist)
            val_pck_img.append(pck_img)
            val_pck_bbox.append(pck_bbox)
            ids.append([j] * len(dist))
            if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
                title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
                title += f"\npck@{pck_threshold}_bbox: {sample_pck_bbox.round(decimals=2)}"
                draw_correspondences(source_points, predicted_points, img1_pil, img2_pil, title=title, radius1=1)
                wandb.log({"val/correspondences": plt}, step=j)
    ids = np.concatenate(ids)
    val_dist = np.concatenate(val_dist)
    val_pck_img = np.concatenate(val_pck_img)
    val_pck_bbox = np.concatenate(val_pck_bbox)
    df = pd.DataFrame({
        "id": ids,
        "distances": val_dist,
        "pck_img": val_pck_img,
        "pck_bbox": val_pck_bbox,
    })
    wandb.log({"val/pck_img": val_pck_img.sum() / len(val_pck_img)})
    wandb.log({"val/pck_bbox": val_pck_bbox.sum() / len(val_pck_bbox)})
    wandb.log({f"val/distances_csv": wandb.Table(dataframe=df)})

def train(config, diffusion_extractor, aggregation_network, optimizer, train_anns, val_anns):
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    np.random.seed(0)
    for epoch in range(config["max_epochs"]):
        epoch_train_anns = np.random.permutation(train_anns)[:config["max_steps_per_epoch"]]
        for i, ann in tqdm(enumerate(epoch_train_anns)):
            step = epoch * config["max_steps_per_epoch"] + i
            optimizer.zero_grad()
            source_points, target_points, _, _, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"])
            if config.get("use_paper_size", False):
                # In the paper we set load_size = 64, output_size = 64 during training
                # and load_size = 224, output_size = 64 during testing to maintain a fair
                # comparison with DINO descriptors.
                # However, one could also set load_size = 512, output_size = 64 to use the
                # max possible resolution supported by Stable Diffusion, which is our
                # recommended setting when training for your use case.
                assert load_size == output_size, "Load and output resolution should be the same for use_paper_size."
                source_points, target_points, _, _, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"])
            else:
                # Resize input images to load_size and rescale points to output_size.
                source_points, target_points, _, _, imgs = load_image_pair(ann, load_size, device, image_path=config["image_path"], output_size=output_size)
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()}, step=step)
            if step > 0 and config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
                with torch.no_grad():
                    log_aggregation_network(aggregation_network, config)
                    save_model(config, aggregation_network, optimizer, step)
                    validate(config, diffusion_extractor, aggregation_network, val_anns)

def load_models(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    device = config.get("device", "cuda")
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = config.get("dims")
    if dims is None:
        dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]
    aggregation_network = AggregationNetwork(
            projection_dim=config["projection_dim"],
            feature_dims=dims,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
    )
    return config, diffusion_extractor, aggregation_network

def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)
    wandb.init(project=config["wandb_project"], name=config["wandb_run"])
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    val_anns = json.load(open(config["val_path"]))
    if config.get("train_path"):
        train_anns = json.load(open(config["train_path"]))
        train(config, diffusion_extractor, aggregation_network, optimizer, train_anns, val_anns)
    else:
        if config.get("weights_path"):
            aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
        validate(config, diffusion_extractor, aggregation_network, val_anns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()
    main(args)
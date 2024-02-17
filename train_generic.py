import argparse
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import wandb

from archs.correspondence_utils import process_image
from train_hyperfeatures import get_rescale_size, load_models, save_model, log_aggregation_network

class ImagePairDataset(torch.utils.data.Dataset):
    # TODO: Write your custom dataset
    def __init__(self, data_path, image_path, load_size):
        super().__init__()
        self.data = json.load(open(data_path))
        self.load_size = load_size
        self.image_path = image_path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        source = Image.open(f"{self.image_path}/{ann['source_path']}").convert("RGB")
        source, _ = process_image(source, res=self.load_size)
        source = source[0]

        target = Image.open(f"{self.image_path}/{ann['target_path']}").convert("RGB")
        target, _ = process_image(target, res=self.load_size)
        target = target[0]

        batch = {
            "source": source,
            "target": target
        }
        return batch
    
def get_loader(config, data_path, shuffle):
    output_size, load_size = get_rescale_size(config)
    image_path = config["image_path"]
    dataset = ImagePairDataset(data_path, image_path, load_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"],
    )
    return dataset, dataloader
    
def loss_fn(pred, target):
    # TODO: Write your custom loss function
    return torch.nn.functional.mse_loss(pred, target)

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
        with torch.autocast("cuda"):
          feats, _ = diffusion_extractor.forward(imgs)
          b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))
    return diffusion_hyperfeats

def validate(config, diffusion_extractor, aggregation_network, val_dataloader):
    # TODO: Write your custom validation function
    pass

def train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader):
    device = config.get("device", "cuda")
    max_epochs = config["max_epochs"]
    np.random.seed(0)
    step = 0
    for epoch in range(max_epochs):
      for batch in tqdm(train_dataloader):
          optimizer.zero_grad()
          imgs, target = batch["source"], batch["target"]
          imgs = imgs.to(device)
          target = target.to(device)
          pred = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
          target = get_hyperfeats(diffusion_extractor, aggregation_network, target)
          loss = loss_fn(pred, target)
          loss.backward()
          optimizer.step()
          wandb.log({"train/loss": loss.item()}, step=step)
          if step > 0 and config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
              with torch.no_grad():
                  log_aggregation_network(aggregation_network, config)
                  save_model(config, aggregation_network, optimizer, step)
                  validate(config, diffusion_extractor, aggregation_network, val_dataloader)
          step += 1

def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)
    wandb.init(project=config["wandb_project"], name=config["wandb_run"])
    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    
    _, train_dataloader = get_loader(config, config["train_path"], True)
    _, val_dataloader = get_loader(config, config["val_path"], False)
    train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # python3 train_generic.py --config_path configs/train.yaml
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()
    main(args)
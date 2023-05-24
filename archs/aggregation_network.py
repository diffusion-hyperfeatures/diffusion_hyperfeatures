import numpy as np
import torch
from torch import nn
from archs.detectron2.resnet import ResNet, BottleneckBlock

class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        return output_feature
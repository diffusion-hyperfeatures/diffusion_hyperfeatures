"""
Function override for Huggingface implementation of latent diffusion models
to cache features. Design pattern inspired by open source implementation 
of Cross Attention Control.
https://github.com/bloc97/CrossAttentionControl
"""
def init_resnet_func(
  unet,
  save_hidden=False,
  use_hidden=False,
  reset=True,
  save_timestep=[],
  idxs=[(1, 0)]
):
  def new_forward(self, input_tensor, temb):
    # https://github.com/huggingface/diffusers/blob/ad9d7ce4763f8fb2a9e620bff017830c26086c36/src/diffusers/models/resnet.py#L372
    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
      input_tensor = self.upsample(input_tensor)
      hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
      input_tensor = self.downsample(input_tensor)
      hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if temb is not None:
      temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
      hidden_states = hidden_states + temb

    hidden_states = self.norm2(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
      input_tensor = self.conv_shortcut(input_tensor)

    if save_hidden:
      if save_timestep is None or self.timestep in save_timestep:
        self.feats[self.timestep] = hidden_states
    elif use_hidden:
      hidden_states = self.feats[self.timestep]
    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    return output_tensor
  
  layers = collect_layers(unet, idxs)
  for module in layers:
    module.forward = new_forward.__get__(module, type(module))
    if reset:
      module.feats = {}
      module.timestep = None

def set_timestep(unet, timestep=None):
  for name, module in unet.named_modules():
    module_name = type(module).__name__
    module.timestep = timestep

def collect_layers(unet, idxs=None):
  layers = []
  for i, up_block in enumerate(unet.up_blocks):
    for j, module in enumerate(up_block.resnets):
      if idxs is None or (i, j) in idxs:
        layers.append(module)
  return layers

def collect_dims(unet, idxs=None):
  dims = []
  for i, up_block in enumerate(unet.up_blocks):
      for j, module in enumerate(up_block.resnets):
          if idxs is None or (i, j) in idxs:
            dims.append(module.time_emb_proj.out_features)
  return dims

def collect_feats(unet, idxs):
  feats = []
  layers = collect_layers(unet, idxs)
  for module in layers:
    feats.append(module.feats)
  return feats

def set_feats(unet, feats, idxs):
  layers = collect_layers(unet, idxs)
  for i, module in enumerate(layers):
    module.feats = feats[i]
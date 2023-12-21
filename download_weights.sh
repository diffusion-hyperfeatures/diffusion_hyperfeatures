mkdir -p weights
# Ours - SDv1-5
wget http://diffusion_hyperfeatures.berkeleyvision.org/weights/aggregation_network.pt -O weights/aggregation_network.pt
# Ours - SDv1-5 (One-Step)
wget http://diffusion_hyperfeatures.berkeleyvision.org/weights/aggregation_network_one-step.pt -O weights/aggregation_network_one-step.pt
# Ours - SDv2-1
wget http://diffusion_hyperfeatures.berkeleyvision.org/weights/aggregation_network_sdv2-1.pt -O weights/aggregation_network_sdv2-1.pt
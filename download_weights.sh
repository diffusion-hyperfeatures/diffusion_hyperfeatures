mkdir -p weights
download_url=https://huggingface.co/g-luo/diffusion-hyperfeatures/resolve/main/weights

# Ours - SDv1-5
wget ${download_url}/aggregation_network.pt?download=true -O weights/aggregation_network.pt
# Ours - SDv1-5 (One-Step)
wget ${download_url}/aggregation_network_one-step.pt?download=true -O weights/aggregation_network_one-step.pt
# Ours - SDv2-1
wget ${download_url}/aggregation_network_sdv2-1.pt?download=true -O weights/aggregation_network_sdv2-1.pt
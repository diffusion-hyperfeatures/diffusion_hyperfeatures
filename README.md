# Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence
This repository contains the code accompanying the paper [Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence](https://diffusion-hyperfeatures.github.io). The code implements Diffusion Hyperfeatures, a framework for consolidating multi-scale and multi-timestep feature maps from a diffusion model into per-pixel feature descriptors.

<img src="assets/approach.png" alt="teaser">

## Setup
This code was tested with Python 3.8. To install the necessary packages, please run:
```
conda env create -f environment.yml
conda activate dhf
```
## Extraction
To extract and save Diffusion Hyperfeatures for your own set of real images, or a set of synthetic images with your own custom prompts, run `extract_hyperfeatures.py`.

To run on real images, you can provide a folder of images with or without corresponding annotations.
```
python3 extract_hyperfeatures.py --save_root hyperfeatures --config_path configs/real.yaml --image_root assets/spair/images --images_or_prompts_path annotations/spair_71k_test-6.json 

python3 extract_hyperfeatures.py --save_root hyperfeatures --config_path configs/real.yaml --image_root assets/spair/images --images_or_prompts_path ""
```

To run on synthetic images, you can provide a json file containing a list of prompts.
```
python3 extract_hyperfeatures.py --save_root hyperfeatures --config_path configs/synthetic.yaml  --image_root "" --images_or_prompts_path annotations/synthetic-3.json
```

## Semantic Keypoint Matching
We also provide demos for the semantic keypoint matching task using Diffusion Hyperfeatures.

For real images, [**real_demo**](real_demo.ipynb) waks through visualizing correspondences using either nearest neighbors or mutual nearest neighbors.

For synthetic images, [**synthetic_demo**](synthetic_demo.ipynb) provides an interactive demo for visualizing correspondences given different prompts and different sets of user annotated source points.

## Citing
```
@article{luo2023dhf,
  author    = {Luo, Grace and Dunlap, Lisa and Park, Dong Huk and Holynski, Aleksander and Darrell, Trevor},
  title     = {Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence},
  journal   = {arXiv},
  year      = {2023},
}
```
## Acknowledgements
Our codebase builds on top of a few prior works, including [Deep ViT Features as Dense Visual Descriptors](https://github.com/ShirAmir/dino-vit-features), [Zero-Shot Category-Level Object Pose Estimation](https://github.com/applied-ai-lab/zero-shot-pose), [Shape-Guided Diffusion](https://github.com/shape-guided-diffusion/shape-guided-diffusion), and [ODISE](https://github.com/NVlabs/ODISE).


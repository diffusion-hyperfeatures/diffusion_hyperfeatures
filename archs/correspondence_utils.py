import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torchvision

from sklearn.cluster import KMeans
import torch.nn.functional as F
from typing import Tuple
from PIL import Image

def process_image(image_pil, res=None, range=(-1, 1)):
    if res:
        image_pil = image_pil.resize(res, Image.BILINEAR)
    image = torchvision.transforms.ToTensor()(image_pil) # range [0, 1]
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min # range [r_min, r_max]
    return image[None, ...], image_pil

"""
Helper functions for computing semantic correspondence via nearest neighbors.
"""
def rescale_points(points, old_shape, new_shape):
    # Assumes old_shape and new_shape are in the format (w, h)
    # and points are in (y, x) order
    x_scale = new_shape[0] / old_shape[0]
    y_scale = new_shape[1] / old_shape[1]
    rescaled_points = np.multiply(points, np.array([y_scale, x_scale]))
    return rescaled_points

def flatten_feats(feats):
    # (b, c, w, h) -> (b, w*h, c)
    b, c, w, h = feats.shape
    feats = feats.view((b, c, -1))
    feats = feats.permute((0, 2, 1))
    return feats

def normalize_feats(feats):
    # (b, w*h, c)
    feats = feats / torch.linalg.norm(feats, dim=-1)[:, :, None]
    return feats

def batch_cosine_sim(img1_feats, img2_feats, flatten=True, normalize=True, low_memory=False):
    if flatten:
        img1_feats = flatten_feats(img1_feats)
        img2_feats = flatten_feats(img2_feats)
    if normalize:
        img1_feats = normalize_feats(img1_feats)
        img2_feats = normalize_feats(img2_feats)
    if low_memory:
        sims = []
        for img1_feat in img1_feats[0]:
            img1_sims = img1_feat @ img2_feats[0].T
            sims.append(img1_sims)
        sims = torch.stack(sims)[None, ...]
    else:
        sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))
    return sims

def find_nn_correspondences(sims):
    """
    Assumes sims is shape (b, w*h, w*h). Returns points1 (w*hx2) which indexes the image1 in column-major order
    and points2 which indexes corresponding points in image2.
    """
    w = h = int(math.sqrt(sims.shape[-1]))
    b = sims.shape[0]
    points1 = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h)), dim=-1)
    points1 = points1.expand((b, w, h, 2))
    # Convert from row-major to column-major order
    points1 = points1.reshape((b, -1, 2))
    
    # Note x = col, y = row
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % h
    points2_y = points2 // h
    points2 = torch.stack([points2_y, points2_x], dim=-1)
    
    points1 = points1.to(torch.float32)
    points2 = points2.to(torch.float32)

    return points1, points2

def find_nn_source_correspondences(img1_feats, img2_feats, source_points, output_size, load_size):
    """
    Precompute nearest neighbor of source_points in img1 to target_points in img2.
    """
    img1_feats = torch.nn.functional.interpolate(img1_feats, load_size, mode="bilinear")
    img2_feats = torch.nn.functional.interpolate(img2_feats, load_size, mode="bilinear")

    source_idx = torch.from_numpy(points_to_idxs(source_points, load_size)).long()
    # Select source_points in the flattened (w, h) dimension as source_idx
    img1_feats = flatten_feats(img1_feats)
    img2_feats = flatten_feats(img2_feats)
    img1_feats = img1_feats[:, source_idx, :]
    img1_feats = normalize_feats(img1_feats)
    img2_feats = normalize_feats(img2_feats)
    sims = torch.matmul(img1_feats, img2_feats.permute((0, 2, 1)))

    # Find nn_correspondences but with points1 = source_points
    num_pixels = int(math.sqrt(sims.shape[-1]))
    points2 = sims.argmax(dim=-1)
    points2_x = points2 % num_pixels
    points2_y = points2 // num_pixels
    points2 = torch.stack([points2_y, points2_x], dim=-1)

    points1 = torch.from_numpy(source_points)
    points2 = points2[0]
    return points1, points2

def points_to_idxs(points, load_size):
    points_y = points[:, 0]
    points_y = np.clip(points_y, 0, load_size[1]-1)
    points_x = points[:, 1]
    points_x = np.clip(points_x, 0, load_size[0]-1)
    idx = load_size[1] * np.round(points_y) + np.round(points_x)
    return idx

def points_to_patches(source_points, num_patches, load_size):
    source_points = np.round(source_points)
    new_H = new_W = num_patches
    # Note that load_size is in (w, h) order and source_points is in (y, x) order
    source_patches_y = (new_H / load_size[1]) * source_points[:, 0]
    source_patches_x = (new_W / load_size[0]) * source_points[:, 1]
    source_patches = np.stack([source_patches_y, source_patches_x], axis=-1)
    # Clip patches for cases where it falls close to the boundary
    source_patches = np.clip(source_patches, 0, num_patches - 1)
    source_patches = np.round(source_patches)
    return source_patches

def compute_pck(predicted_points, target_points, load_size, pck_threshold=0.1):
  distances = np.linalg.norm(predicted_points - target_points, axis=-1)
  pck = distances <= pck_threshold * max(load_size)
  return distances, pck.sum() / len(pck)

"""
Helper functions adapted from https://github.com/ShirAmir/dino-vit-features.
"""
def draw_correspondences(points1, points2, image1, image2, image1_label="", image2_label="", title="", radius1=8, radius2=1):
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax1, ax2 = axs[0], axs[1]
    ax1.set_xlabel(image1_label)
    ax2.set_xlabel(image2_label)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('hsv')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    for number, (point1, point2, color) in enumerate(zip(points1, points2, colors)):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axs

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

def find_best_buddies_correspondences(
    descriptors1, 
    descriptors2,
    saliency_map1, 
    saliency_map2,
    num_pairs: int = 10,
    thresh: float = 0.05,
):
    """
    Adapted from find_correspondences.
    
    Legend: B: batch, T: total tokens (num_patches ** 2), D: Descriptor dim per head.
    Method: Find mutual nearest neighbours from Image1 --> Image2, and Image2 --> Image1.
    :param descriptors1: descriptors of shape B x 1 x T x D.
    :param descriptors2: descriptors of shape B x 1 x T x D. 
    :param saliency_map1: saliency maps of shape B x T.
    :param saliency_map2: saliency maps of shape B x T.
    :param num_pairs: number of outputted corresponding pairs.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    """

    # extracting descriptors for each image
    device = descriptors1.device
    B, _, t, d = descriptors1.size()
    num_patches1 = num_patches2 = (int(np.sqrt(t)), int(np.sqrt(t)))

    # remove batch dim from all tensors
    saliency_map1 = saliency_map1[0]
    saliency_map2 = saliency_map2[0]

    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    # remove batch dim from all tensors
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_bb_descs = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(num_pairs, len(all_bb_descs))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_bb_descs ** 2).sum(axis=1))[:, None]
    normalized_all_bb_descs = all_bb_descs / length
    if len(normalized_all_bb_descs) == 0:
        return [], []

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_all_bb_descs)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    # for each kmeans cluster, find the pairs with the highest rank
    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[bb_indices_to_show] # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1])
    img1_x_to_show = (img1_indices_to_show % num_patches1[1])
    img2_y_to_show = (img2_indices_to_show / num_patches2[1])
    img2_x_to_show = (img2_indices_to_show % num_patches2[1])

    points1 = torch.stack([img1_y_to_show, img1_x_to_show], dim=-1)
    points2 = torch.stack([img2_y_to_show, img2_x_to_show], dim=-1)
    return points1, points2

"""
Helper functions adapted from https://github.com/applied-ai-lab/zero-shot-pose.
"""
def _to_cartesian(coords: torch.Tensor, shape: Tuple):
    """
    Takes raveled coordinates and returns them in a cartesian coordinate frame
    coords: B x D
    shape: tuple of cartesian dimensions
    return: B x D x 2
    """
    i, j = (torch.from_numpy(inds) for inds in np.unravel_index(coords.cpu(), shape=shape))
    return torch.stack([i, j], dim=-1)
    
def find_cyclical_correspondences(
    descriptors1, 
    descriptors2, 
    saliency_map1, 
    saliency_map2,
    num_pairs: int = 10, 
    thresh: float = 0.05,
):
    """
    Adapted from find_correspondences_batch_with_knn.

    Legend: B: batch, T: total tokens (num_patches ** 2), D: Descriptor dim per head
    Method: Compute similarity between all pairs of pixel descriptors
            Find nearest neighbours from Image1 --> Image2, and Image2 --> Image1
            Use nearest neighbours to define a cycle from Image1 --> Image2 --> Image1
            Take points in Image1 (and corresponding points in Image2) which have smallest 'cycle distance'
            Also, filter examples which aren't part of the foreground in both images, as determined by ViT attention maps
    :param descriptors1: descriptors of shape B x 1 x T x D.
    :param descriptors2: descriptors of shape B x 1 x T x D. 
    :param saliency_map1: saliency maps of shape B x T.
    :param saliency_map2: saliency maps of shape B x T.
    :param num_pairs: number of outputted corresponding pairs.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    """
    device = descriptors1.device
    B, _, t, d = descriptors1.size()
    num_patches1 = (int(np.sqrt(t)), int(np.sqrt(t)))
    inf_idx = int(t)

    # -----------------
    # EXTRACT SALIENCY MAPS
    # -----------------
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # -----------------
    # COMPUTE SIMILARITIES
    # calculate similarity between image1 and image2 descriptors
    # -----------------
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # -----------------
    # COMPUTE MUTUAL NEAREST NEIGHBOURS
    # -----------------
    sim_1, nn_1 = torch.max(similarities, dim=-1, keepdim=False)  # nn_1 - indices of block2 closest to block1. B x T
    sim_2, nn_2 = torch.max(similarities, dim=-2, keepdim=False)  # nn_2 - indices of block1 closest to block2. B x T
    nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]

    # Map nn_2 points which are not highlighted by fg_mask to 0
    nn_2[~fg_mask2] = 0     # TODO: Note, this assumes top left pixel is never a point of interest
    cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)  # Intuitively, nn_2[nn_1]

    # -----------------
    # COMPUTE SIMILARITIES
    # Find distance between cyclical point and original point in Image1
    # -----------------
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1])[None, :].repeat(B, 1)
    cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches1).to(device)
    image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches1).to(device)

    # Find which points are mapped to 0, artificially map them to a high value
    zero_mask = (cyclical_idxs_ij - torch.Tensor([0, 0])[None, None, :].to(device)) == 0
    cyclical_idxs_ij[zero_mask] = inf_idx

    # Find negative of distance between cyclical point and original point
    # View to make sure PairwiseDistance behaviour is consistent across torch versions
    b, hw, ij_dim = cyclical_idxs_ij.size()
    cyclical_dists = -torch.nn.PairwiseDistance(p=2)(cyclical_idxs_ij.view(-1, ij_dim), image_idxs_ij.view(-1, ij_dim))
    cyclical_dists = cyclical_dists.view(b, hw)

    cyclical_dists_norm = cyclical_dists - cyclical_dists.min(1, keepdim=True)[0]        # Normalize to [0, 1]
    cyclical_dists_norm /= cyclical_dists_norm.max(1, keepdim=True)[0]

    # -----------------
    # Further mask pixel locations in Image1 which are not highlighted by FG mask
    # -----------------
    cyclical_dists_norm *= fg_mask1.float()

    # -----------------
    # Find the TopK points in Image1 and their correspondences in Image2
    # -----------------
    sorted_vals, topk_candidate_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)
    topk_candidate_points_image_1 = topk_candidate_points_image_1[:, :num_pairs * 2]

    # -----------------
    # Now do K-Means clustering on the descriptors in image 1 to choose well distributed features
    # -----------------
    selected_points_image_1 = []
    for b in range(B):

        idxs_b = topk_candidate_points_image_1[b]
        feats_b = descriptors1[b][0, :, :][idxs_b] # num_pairs_for_topk x D * H
        feats_b = F.normalize(feats_b, dim=-1).cpu().numpy()
        salience_b = saliency_map1[b][idxs_b] # num_pairs_for_topk

        kmeans = KMeans(n_clusters=num_pairs, random_state=0).fit(feats_b)
        kmeans_labels = torch.as_tensor(kmeans.labels_).to(device)

        final_idxs_chosen_from_image_1_b = []
        for k in range(num_pairs):

            locations_in_cluster_k = torch.where(kmeans_labels == k)[0]
            saliencies_at_k = salience_b[locations_in_cluster_k]
            point_chosen_from_cluster_k = saliencies_at_k.argmax()
            final_idxs_chosen_from_image_1_b.append(idxs_b[locations_in_cluster_k][point_chosen_from_cluster_k])

        final_idxs_chosen_from_image_1_b = torch.stack(final_idxs_chosen_from_image_1_b)
        selected_points_image_1.append(final_idxs_chosen_from_image_1_b)

    selected_points_image_1 = torch.stack(selected_points_image_1)

    # Get corresponding points in image 2
    selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)

    # -----------------
    # Compute the distances of the selected points
    # -----------------
    sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=selected_points_image_1.to(device))

    # Convert to cartesian coordinates
    selected_points_image_1, selected_points_image_2 = (_to_cartesian(inds, shape=num_patches1) for inds in
                                                        (selected_points_image_1, selected_points_image_2))

    cyclical_dists = cyclical_dists.reshape(-1, num_patches1[0], num_patches1[1])

    # Remove batch dim
    points1 = selected_points_image_1[0]
    points2 = selected_points_image_2[0]
    return points1, points2
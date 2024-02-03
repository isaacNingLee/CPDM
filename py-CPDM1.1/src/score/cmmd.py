import os
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from embedding import CLIPPytorch
import io_util


_SIGMA = 1.0
_SCALE = 1.0

def mmd(x, y):
    """
    Memory-efficient MMD implementation in PyTorch.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)

    # Calculate the squared norms of each row in x and y
    x_sqnorms = torch.sum(x**2, dim=1)
    y_sqnorms = torch.sum(y**2, dim=1)

    gamma = 1 / (2 * _SIGMA**2)

    # Compute the kernel matrices
    k_xx = torch.mean(torch.exp(-gamma * (-2 * torch.mm(x, x.t()) + x_sqnorms.view(-1, 1) + x_sqnorms.view(1, -1))))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * torch.mm(x, y.t()) + x_sqnorms.view(-1, 1) + y_sqnorms.view(1, -1))))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * torch.mm(y, y.t()) + y_sqnorms.view(-1, 1) + y_sqnorms.view(1, -1))))

    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def compute_cmmd(
    ref_dir, eval_dir, batch_size = 32, max_count = -1
):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
        ref_dir: Path to the directory containing reference images.
        eval_dir: Path to the directory containing images to be evaluated.
        batch_size: Batch size used in the CLIP embedding calculation.
        max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
        The CMMD value between the image sets.
    """
    embedding_model = CLIPPytorch()
    ref_embs = io_util.compute_embeddings_for_dir(
        ref_dir, embedding_model, batch_size, max_count
    )
    eval_embs = io_util.compute_embeddings_for_dir(
        eval_dir, embedding_model, batch_size, max_count
    )
    val = mmd(ref_embs, eval_embs)
    return np.asarray(val)
# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def save_image_grid(images, path, ncols=8):
    """
    images: numpy array, shape (N, H, W, C), values in [0,1] or [0,255]
    saves a grid image at path.
    """
    if images.dtype != np.uint8:
        images = (images * 255).astype(np.uint8)

    N, H, W, C = images.shape
    ncols = min(ncols, N)
    nrows = int(np.ceil(N / ncols))
    grid = np.ones((nrows * H + (nrows - 1), ncols * W + (ncols - 1), C), dtype=np.uint8) * 255

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= N:
                break
            y0 = r * (H + 1)
            x0 = c * (W + 1)
            grid[y0:y0+H, x0:x0+W] = images[idx]
            idx += 1

    # If grayscale single channel, squeeze
    if C == 1:
        plt.imsave(path, grid.squeeze(), cmap="gray")
    else:
        plt.imsave(path, grid)


def make_checkpoint_manager(generator, discriminator, g_opt, d_opt, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(generator=generator,
                               discriminator=discriminator,
                               generator_optimizer=g_opt,
                               discriminator_optimizer=d_opt)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    return ckpt, manager

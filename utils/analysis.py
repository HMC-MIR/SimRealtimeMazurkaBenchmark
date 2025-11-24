import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

def plot_alignment_slice(
    D, path, gt,
    frame_start=0, frame_end=None,
    figsize=(10, 8),
    title="NOA Alignment (Query-Sliced) with Normalized Accumulated Cost"
):
    """Plot alignment and ground-truth paths over a sliced cost matrix."""
    
    if frame_end is None:
        frame_end = D.shape[1]

    # Slice D on the query axis
    D_slice = D[frame_start:frame_end]

    # Filter alignment path
    mask = (path[0] >= frame_start) & (path[0] < frame_end)
    p_y, p_x = path[0][mask], path[1][mask]

    # Filter ground truth
    gt_mask = (gt[0] >= frame_start) & (gt[0] < frame_end)
    gt_y, gt_x = gt[0][gt_mask], gt[1][gt_mask]

    # Determine visible x-range
    all_x = np.concatenate([p_x, gt_x]) if gt_x.size else p_x
    if all_x.size:
        x_min, x_max = np.min(all_x), np.max(all_x)
        pad = max(10, 0.05 * (x_max - x_min))
        x_lo = max(0, x_min - pad)
        x_hi = min(D.shape[1], x_max + pad)
    else:
        x_lo, x_hi = 0, D.shape[1]

    col_start, col_end = int(np.floor(x_lo)), int(np.ceil(x_hi))
    D_view = D_slice[:, col_start:col_end]

    # Colormap only depends on values in the slice
    finite_vals = D_view[np.isfinite(D_view)]
    vmin = finite_vals[finite_vals != -np.inf].min() if np.any((finite_vals != -np.inf)) else D_view.min()
    vmax = finite_vals[finite_vals != np.inf].max() if np.any((finite_vals != np.inf)) else D_view.max()

    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "lightblue", "blue", "grey"])

    im = ax.imshow(
        D_view,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=[col_start, col_end, frame_start, frame_end],
        vmin=vmin,
        vmax=vmax
    )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Cost")

    # Plot paths
    ax.plot(p_x, p_y, "r-", lw=2, label="Alignment Path")
    ax.plot(gt_x, gt_y, "x", color="darkorange", lw=2, label="Ground Truth")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(frame_start, frame_end)
    ax.set_aspect("equal")

    ax.set_xlabel("Reference (frame)")
    ax.set_ylabel("Query (frame)")
    ax.legend()

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

from numba import njit
@njit
def normalize_D(D):
    """Normalize the cost matrix D by dividing each element by (i + j + 2)"""
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i, j] = D[i, j] / (i + j + 2)
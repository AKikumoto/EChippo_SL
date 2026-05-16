"""visualize.py — single-axes plot helpers for EChipp_SL notebooks.

Each function fills one matplotlib Axes.
Figure layout (GridSpec, figsize, suptitle, savefig) is the caller's responsibility.

Functions
---------
plot_rsa_heatmap : RSA similarity matrix heatmap with group boundary lines
plot_sim_bars    : pattern similarity bars per layer × category (mean ± SE)
plot_output_prob : output probability learning curve(s) with optional chance line
"""

import numpy as np


def plot_rsa_heatmap(ax, mat, n_groups, group_size, title=None, ylabel=None,
                     vmin=-0.5, vmax=1.0, cmap='RdYlBu_r'):
    """RSA heatmap with group boundary lines.

    Parameters
    ----------
    ax         : matplotlib Axes
    mat        : ndarray (n_items, n_items) — mean similarity matrix
    n_groups   : int — number of groups/communities/pairs
    group_size : int — items per group
    title      : str or None
    ylabel     : str or None

    Returns
    -------
    im : AxesImage (pass to plt.colorbar if needed)
    """
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    for k in range(1, n_groups):
        ax.axhline(k * group_size - 0.5, color='k', lw=0.8)
        ax.axvline(k * group_size - 0.5, color='k', lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8)
    return im


def plot_sim_bars(ax, sim_dict, layers, categories, colors=None,
                  title='Pattern similarity', ylabel='Pattern similarity (r)'):
    """Pattern similarity bars per layer per category (mean ± SE).

    Parameters
    ----------
    ax         : matplotlib Axes
    sim_dict   : dict[layer][category] → (mean: float, se: float)
    layers     : list[str] — layer names (x-axis groups)
    categories : list[str] — category names (bars within each group)
    colors     : list[str] or None — one color per category (default: C0, C1, ...)
    title      : str or None
    ylabel     : str
    """
    if colors is None:
        colors = [f'C{i}' for i in range(len(categories))]
    n_cats  = len(categories)
    width   = 0.65 / n_cats
    offsets = [(j - (n_cats - 1) / 2) * width for j in range(n_cats)]
    x       = np.arange(len(layers))

    for i, layer in enumerate(layers):
        for j, (cat, col) in enumerate(zip(categories, colors)):
            m, se = sim_dict[layer][cat]
            ax.bar(x[i] + offsets[j], m, width=width, color=col, alpha=0.8,
                   label=cat if i == 0 else '')
            ax.errorbar(x[i] + offsets[j], m, yerr=se, fmt='none', color='k', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in layers], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=7)
    if title is not None:
        ax.set_title(title, fontsize=9)


def plot_output_prob(ax, epochs, probs, chance=None, title=None,
                     ylabel='Output probability'):
    """Output probability learning curve(s) over epochs.

    Parameters
    ----------
    ax      : matplotlib Axes
    epochs  : array-like — epoch indices (x-axis)
    probs   : dict[label → ndarray] or ndarray — curves to plot.
              dict keys become legend labels; bare ndarray is plotted unlabelled.
    chance  : float or None — horizontal chance line
    title   : str or None
    ylabel  : str
    """
    markers = ('o', 's', '^', 'D')
    items   = list(probs.items()) if isinstance(probs, dict) else [(None, probs)]

    for k, (lbl, p) in enumerate(items):
        ax.plot(epochs, p, marker=markers[k % len(markers)], markersize=4,
                color=f'C{k}', label=lbl)

    if chance is not None:
        ax.axhline(chance, color='gray', lw=0.8, ls='--', label='chance')

    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    epochs_list = list(epochs)
    ax.set_xticks(epochs_list[::2] if len(epochs_list) > 5 else epochs_list)
    if any(lbl is not None for lbl, _ in items) or chance is not None:
        ax.legend(fontsize=7)
    if title is not None:
        ax.set_title(title, fontsize=9)

"""visualize.py — single-axes plot helpers for EChipp_SL notebooks.

Each function fills one matplotlib Axes.
Figure layout (GridSpec, figsize, suptitle, savefig) is the caller's responsibility.

Functions
---------
plot_community_graph : community graph with bottleneck topology (networkx-based)
plot_rsa_heatmap     : RSA similarity matrix heatmap with group boundary lines
plot_sim_bars        : pattern similarity bars per layer × category (mean ± SE)
plot_output_prob     : output probability learning curve(s) with optional chance line
"""

import networkx as nx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np


def plot_community_graph(
    ax,
    adjacency,
    n_communities,
    items_per_community,
    *,
    palette=None,
    title=None,
    monochrome=False,
    R_comm=2.0,
    r_item=0.75,
):
    """Community graph: nodes colored by community, bottleneck nodes as squares.

    Parameters
    ----------
    ax                  : matplotlib Axes
    adjacency           : ndarray (n_items, n_items) — binary adjacency matrix
    n_communities       : int
    items_per_community : int
    palette             : list[str] or None — one color per community
    title               : str or None
    R_comm              : float — radius of the community center ring
    r_item              : float — radius of item cluster within each community

    Notes
    -----
    Layout: community centers at equal angular intervals; the last item of each
    community (IPC-1) faces the next community center so bottleneck edges read cleanly.
    Bottleneck nodes (items with any between-community edge) are drawn as squares.
    """
    nc  = n_communities
    ipc = items_per_community

    if palette is None:
        palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
                   '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
    palette = palette[:nc]

    G = nx.from_numpy_array(adjacency)

    bottlenecks = {
        i for i in range(nc * ipc)
        for j in range(nc * ipc)
        if adjacency[i, j] > 0 and i // ipc != j // ipc
    }

    pos = {}
    for c in range(nc):
        alpha = np.pi / 2 + 2 * np.pi * c / nc
        cx, cy = R_comm * np.cos(alpha), R_comm * np.sin(alpha)
        alpha_next = np.pi / 2 + 2 * np.pi * ((c + 1) % nc) / nc
        phi = np.arctan2(R_comm * np.sin(alpha_next) - cy,
                         R_comm * np.cos(alpha_next) - cx)
        for i in range(ipc):
            angle = phi + 2 * np.pi * (i - (ipc - 1)) / ipc
            pos[c * ipc + i] = (cx + r_item * np.cos(angle),
                                  cy + r_item * np.sin(angle))

    edge_within  = [(u, v) for u, v in G.edges() if u // ipc == v // ipc]
    edge_between = [(u, v) for u, v in G.edges() if u // ipc != v // ipc]

    if monochrome:
        # Paper Fig 3(b): black=boundary, gray=internal, no community circles or legend
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#333333', width=1.2, alpha=0.8)
        interior = [n for n in G.nodes() if n not in bottlenecks]
        bns      = list(bottlenecks)
        nx.draw_networkx_nodes(G, pos, nodelist=interior, ax=ax,
                               node_color='#888888', node_size=320, node_shape='o')
        nx.draw_networkx_nodes(G, pos, nodelist=bns, ax=ax,
                               node_color='black', node_size=420, node_shape='o',
                               edgecolors='black', linewidths=1.5)
        span = R_comm + r_item + 0.5
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect('equal')
        ax.axis('off')
        if title is not None:
            ax.set_title(title, fontsize=11)
        return

    for c in range(nc):
        alpha = np.pi / 2 + 2 * np.pi * c / nc
        cx, cy = R_comm * np.cos(alpha), R_comm * np.sin(alpha)
        ax.add_patch(mpatches.Circle((cx, cy), r_item + 0.25,
                                      color=palette[c], alpha=0.08))
        ax.add_patch(mpatches.Circle((cx, cy), r_item + 0.25,
                                      color=palette[c], alpha=0.4,
                                      fill=False, linewidth=2, linestyle='--'))

    for c in range(nc):
        edges_c = [(u, v) for u, v in edge_within if u // ipc == c]
        nx.draw_networkx_edges(G, pos, edgelist=edges_c, ax=ax,
                               edge_color=palette[c], width=1.5, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=edge_between, ax=ax,
                           edge_color='#222', width=2.5, alpha=0.9, style='dashed')

    interior = [n for n in G.nodes() if n not in bottlenecks]
    nx.draw_networkx_nodes(G, pos, nodelist=interior, ax=ax,
                           node_color=[palette[n // ipc] for n in interior],
                           node_size=350, node_shape='o')
    bns = list(bottlenecks)
    nx.draw_networkx_nodes(G, pos, nodelist=bns, ax=ax,
                           node_color=[palette[n // ipc] for n in bns],
                           node_size=450, node_shape='s',
                           edgecolors='#222', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax,
                             font_size=8, font_color='white', font_weight='bold')

    legend_handles = [
        mpatches.Patch(color=palette[c],
                       label=f'Community {c}  (items {c*ipc}–{c*ipc+ipc-1})')
        for c in range(nc)
    ] + [
        mpatches.Patch(facecolor='white', edgecolor='#222', linewidth=1.5,
                       label='Bottleneck node (square)'),
        mlines.Line2D([0], [0], color='#222', linewidth=2.5, linestyle='--',
                      label='Between-community edge'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    span = R_comm + r_item + 0.5
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=11)


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

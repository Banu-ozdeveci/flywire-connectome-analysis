from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


@dataclass(frozen=True)
class BlockFlow:
    flow: np.ndarray
    expected: np.ndarray
    send_levels: np.ndarray
    recv_levels: np.ndarray
    kout_by_send: np.ndarray


def _as_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _set_heatmap_ticks(ax, xlabels, ylabels, *, x_offset: float = 0.0, y_offset: float = 0.0):
    xlabels = [str(x) for x in xlabels]
    ylabels = [str(y) for y in ylabels]

    n_x = len(xlabels)
    n_y = len(ylabels)

    # Prefer extracting cell centers from the rendered QuadMesh (seaborn heatmap),
    # which is robust across seaborn/matplotlib versions and avoids off-by-0.5 ticks.
    x_centers = None
    y_centers = None
    if ax.collections and hasattr(ax.collections[0], "get_coordinates"):
        coords = ax.collections[0].get_coordinates()
        if coords is not None and coords.ndim == 3 and coords.shape[-1] == 2:
            x_edges = coords[0, :, 0].astype(float, copy=False)
            y_edges = coords[:, 0, 1].astype(float, copy=False)
            if len(x_edges) == n_x + 1 and len(y_edges) == n_y + 1:
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    # Fallback: infer centers from axis limits.
    if x_centers is None or y_centers is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        x_left = min(x0, x1)
        y_bottom = min(y0, y1)

        if n_x:
            x_centers = np.linspace(x_left + 0.5, max(x0, x1) - 0.5, n_x)
        else:
            x_centers = []

        if n_y:
            y_centers = np.linspace(y_bottom + 0.5, max(y0, y1) - 0.5, n_y)
        else:
            y_centers = []

    if len(x_centers):
        x_centers = np.asarray(x_centers, dtype=float) + float(x_offset)
    if len(y_centers):
        y_centers = np.asarray(y_centers, dtype=float) + float(y_offset)

    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels, rotation=0)


def block_flow_from_edges(
    df: pd.DataFrame,
    send_labels: Sequence[int],
    recv_labels: Sequence[int],
    *,
    pre_col: str = "pre_idx",
    post_col: str = "post_idx",
    weight_col: Optional[str] = "syn_count",
    drop_labels: Optional[Collection[int]] = None,
) -> BlockFlow:
    send_labels = _as_array(send_labels)
    recv_labels = _as_array(recv_labels)

    pre_idx = df[pre_col].to_numpy(dtype=int, copy=False)
    post_idx = df[post_col].to_numpy(dtype=int, copy=False)

    send_edge = send_labels[pre_idx]
    recv_edge = recv_labels[post_idx]

    if drop_labels:
        drop = set(drop_labels)
        keep = ~np.isin(send_edge, list(drop)) & ~np.isin(recv_edge, list(drop))
        pre_idx = pre_idx[keep]
        post_idx = post_idx[keep]
        send_edge = send_edge[keep]
        recv_edge = recv_edge[keep]
        weights = df.loc[keep, weight_col].to_numpy(float, copy=False) if weight_col else None
    else:
        weights = df[weight_col].to_numpy(float, copy=False) if weight_col else None

    send_levels, send_mapped = np.unique(send_edge, return_inverse=True)
    recv_levels, recv_mapped = np.unique(recv_edge, return_inverse=True)

    n_send = len(send_levels)
    n_recv = len(recv_levels)

    flow = np.zeros((n_send, n_recv), dtype=float)
    if weights is None:
        np.add.at(flow, (send_mapped, recv_mapped), 1.0)
    else:
        np.add.at(flow, (send_mapped, recv_mapped), weights)

    # Out–in null expectation at the block level: E_ab = kout_a * kin_b / m
    if weights is None:
        edge_w = np.ones_like(send_mapped, dtype=float)
    else:
        edge_w = weights

    kout_by_send = np.bincount(send_mapped, weights=edge_w, minlength=n_send)
    kin_by_recv = np.bincount(recv_mapped, weights=edge_w, minlength=n_recv)
    m = float(edge_w.sum())
    expected = np.outer(kout_by_send, kin_by_recv) / m if m > 0 else np.zeros_like(flow)

    return BlockFlow(
        flow=flow,
        expected=expected,
        send_levels=send_levels,
        recv_levels=recv_levels,
        kout_by_send=kout_by_send,
    )


def plot_block_flow_panels(
    block: BlockFlow,
    *,
    labels_map: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    eps: float = 1.0,
    figsize: Tuple[int, int] = (18, 5),
    tick_offset_x: float = 0.0,
    tick_offset_y: float = 0.0,
):
    import seaborn as sns

    send_ticks = [labels_map.get(int(x), str(int(x))) for x in block.send_levels] if labels_map else block.send_levels
    recv_ticks = [labels_map.get(int(x), str(int(x))) for x in block.recv_levels] if labels_map else block.recv_levels

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sns.heatmap(
        np.log1p(block.flow),
        cmap="magma",
        ax=axes[0],
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "log(1 + weight)"},
    )
    _set_heatmap_ticks(axes[0], recv_ticks, send_ticks, x_offset=tick_offset_x, y_offset=tick_offset_y)
    axes[0].set_title("Block flow (log scale)")
    axes[0].set_xlabel("Receiver label")
    axes[0].set_ylabel("Sender label")

    row_frac = np.divide(
        block.flow,
        block.kout_by_send[:, None],
        out=np.zeros_like(block.flow),
        where=block.kout_by_send[:, None] > 0,
    )
    vmax = float(np.nanpercentile(row_frac, 99.5)) if row_frac.size else 1.0
    vmax = max(vmax, 1e-12)
    sns.heatmap(
        row_frac,
        cmap="viridis",
        ax=axes[1],
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Fraction of outgoing weight"},
        vmin=0,
        vmax=vmax,
    )
    _set_heatmap_ticks(axes[1], recv_ticks, send_ticks, x_offset=tick_offset_x, y_offset=tick_offset_y)
    axes[1].set_title("Row-normalized flow")
    axes[1].set_xlabel("Receiver label")
    axes[1].set_ylabel("Sender label")

    log_enrich = np.log10((block.flow + eps) / (block.expected + eps))
    vmax = float(np.nanpercentile(np.abs(log_enrich), 99.0)) if log_enrich.size else 1.0
    vmax = max(vmax, 1e-12)
    sns.heatmap(
        log_enrich,
        cmap="coolwarm",
        center=0,
        ax=axes[2],
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "log10((obs+eps)/(exp+eps))"},
        vmin=-vmax,
        vmax=vmax,
    )
    _set_heatmap_ticks(axes[2], recv_ticks, send_ticks, x_offset=tick_offset_x, y_offset=tick_offset_y)
    axes[2].set_title("Enrichment vs out-in null")
    axes[2].set_xlabel("Receiver label")
    axes[2].set_ylabel("Sender label")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_block_flow_single(
    block: BlockFlow,
    *,
    kind: str,
    labels_map: Optional[Dict[int, str]] = None,
    eps: float = 1.0,
    figsize: Tuple[int, int] = (7, 6),
    title: Optional[str] = None,
    tick_offset_x: float = 0.0,
    tick_offset_y: float = 0.0,
):
    import seaborn as sns

    kind = kind.lower().strip()
    send_ticks = [labels_map.get(int(x), str(int(x))) for x in block.send_levels] if labels_map else block.send_levels
    recv_ticks = [labels_map.get(int(x), str(int(x))) for x in block.recv_levels] if labels_map else block.recv_levels

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    heatmap_kwargs = {}
    if kind in {"log", "log1p"}:
        data = np.log1p(block.flow)
        cmap = "magma"
        cbar_label = "log(1 + weight)"
        panel_title = "Block flow (log scale)"
    elif kind in {"row", "row_norm", "rownorm"}:
        data = np.divide(
            block.flow,
            block.kout_by_send[:, None],
            out=np.zeros_like(block.flow),
            where=block.kout_by_send[:, None] > 0,
        )
        vmax = float(np.nanpercentile(data, 99.5)) if data.size else 1.0
        vmax = max(vmax, 1e-12)
        cmap = "viridis"
        cbar_label = "Fraction of outgoing weight"
        panel_title = "Row-normalized flow"
        heatmap_kwargs.update(vmin=0, vmax=vmax)
    elif kind in {"enrich", "enrichment", "null"}:
        data = np.log10((block.flow + eps) / (block.expected + eps))
        vmax = float(np.nanpercentile(np.abs(data), 99.0)) if data.size else 1.0
        vmax = max(vmax, 1e-12)
        cmap = "coolwarm"
        cbar_label = "log10((obs+eps)/(exp+eps))"
        panel_title = "Enrichment vs out-in null"
        heatmap_kwargs.update(center=0, vmin=-vmax, vmax=vmax)
    else:
        raise ValueError("kind must be one of: 'log', 'row', 'enrich'")

    sns.heatmap(
        data,
        cmap=cmap,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": cbar_label},
        **heatmap_kwargs,
    )
    _set_heatmap_ticks(ax, recv_ticks, send_ticks, x_offset=tick_offset_x, y_offset=tick_offset_y)
    ax.set_title(title or panel_title)
    ax.set_xlabel("Receiver label")
    ax.set_ylabel("Sender label")
    fig.tight_layout()
    return fig


def plot_block_flow_panels_from_df(
    *,
    df: pd.DataFrame,
    send_labels: Sequence[int],
    recv_labels: Sequence[int],
    pre_col: str = "pre_idx",
    post_col: str = "post_idx",
    weight_col: Optional[str] = "syn_count",
    labels_map: Optional[Dict[int, str]] = None,
    drop_labels: Optional[Collection[int]] = None,
    eps: float = 1.0,
    figsize: Tuple[int, int] = (18, 5),
):
    block = block_flow_from_edges(
        df,
        send_labels,
        recv_labels,
        pre_col=pre_col,
        post_col=post_col,
        weight_col=weight_col,
        drop_labels=drop_labels,
    )
    return plot_block_flow_panels(block, labels_map=labels_map, eps=eps, figsize=figsize)


def plot_edge_cluster_grid(
    *,
    df: pd.DataFrame,
    U: np.ndarray,
    V: np.ndarray,
    background_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "edge_cluster",
    pre_col: str = "pre_idx",
    post_col: str = "post_idx",
    weight_col: str = "syn_count",
    vector_id: int = 0,
    max_clusters: Optional[int] = 9,
    ncols: int = 3,
    background_edges: int = 6000,
    max_edges_per_cluster: int = 2500,
    random_state: int = 0,
    edge_alpha: float = 0.08,
):
    pre = df[pre_col].to_numpy(dtype=int, copy=False)
    post = df[post_col].to_numpy(dtype=int, copy=False)
    edge_w = df[weight_col].to_numpy(dtype=float, copy=False)
    clusters = df[cluster_col].to_numpy(dtype=int, copy=False)

    n_nodes = U.shape[0]
    x = U[:, vector_id].astype(float, copy=False)
    y = V[:, vector_id].astype(float, copy=False)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    pos = np.column_stack([x, y])

    n_clusters = int(clusters.max()) + 1 if clusters.size else 0
    if n_clusters == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No clusters to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    cluster_weight = np.bincount(clusters, weights=edge_w, minlength=n_clusters)
    cluster_order = np.argsort(cluster_weight)[::-1]
    if max_clusters is not None:
        cluster_order = cluster_order[:max_clusters]

    n_panels = len(cluster_order)
    ncols = max(1, min(ncols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    rng = np.random.default_rng(random_state)
    if background_df is None:
        bg_pre = pre
        bg_post = post
    else:
        bg_pre = background_df[pre_col].to_numpy(dtype=int, copy=False)
        bg_post = background_df[post_col].to_numpy(dtype=int, copy=False)

    bg_n = min(int(background_edges), bg_pre.size)
    if bg_n > 0:
        bg_sel = rng.choice(bg_pre.size, size=bg_n, replace=False)
        bg_segments = np.stack([pos[bg_pre[bg_sel]], pos[bg_post[bg_sel]]], axis=1)
    else:
        bg_segments = None

    x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
    y_min, y_max = float(pos[:, 1].min()), float(pos[:, 1].max())
    x_pad = 0.05 * (x_max - x_min + 1e-12)
    y_pad = 0.05 * (y_max - y_min + 1e-12)

    send_color = "#d62728"  # red
    recv_color = "#1f77b4"  # blue
    both_color = "#9467bd"  # purple

    for panel_idx, cluster_id in enumerate(cluster_order):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        ax.set_axis_off()
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if bg_segments is not None:
            ax.add_collection(LineCollection(bg_segments, colors="k", linewidths=0.2, alpha=0.03))

        mask = clusters == cluster_id
        if not np.any(mask):
            ax.set_title(f"Cluster {cluster_id}")
            continue

        cu = pre[mask]
        cv = post[mask]
        cw = edge_w[mask]

        send_nodes = np.unique(cu)
        recv_nodes = np.unique(cv)
        is_send = np.zeros(n_nodes, dtype=bool)
        is_recv = np.zeros(n_nodes, dtype=bool)
        is_send[send_nodes] = True
        is_recv[recv_nodes] = True
        is_both = is_send & is_recv
        send_only = is_send & ~is_both
        recv_only = is_recv & ~is_both
        neither = ~(is_send | is_recv)

        ax.scatter(pos[neither, 0], pos[neither, 1], s=10, c="tab:gray", alpha=0.35, linewidths=0)
        ax.scatter(pos[send_only, 0], pos[send_only, 1], s=30, c=send_color, marker="s", edgecolors="k", linewidths=0.3)
        ax.scatter(pos[recv_only, 0], pos[recv_only, 1], s=30, c=recv_color, marker="D", edgecolors="k", linewidths=0.3)
        ax.scatter(pos[is_both, 0], pos[is_both, 1], s=30, c=both_color, marker="o", edgecolors="k", linewidths=0.3)

        if cu.size > max_edges_per_cluster:
            probs = cw / cw.sum() if cw.sum() > 0 else None
            sel = rng.choice(cu.size, size=max_edges_per_cluster, replace=False, p=probs)
            draw_u = cu[sel]
            draw_v = cv[sel]
        else:
            draw_u = cu
            draw_v = cv

        segments = np.stack([pos[draw_u], pos[draw_v]], axis=1)
        ax.add_collection(LineCollection(segments, colors="k", linewidths=0.35, alpha=edge_alpha))

        ax.set_title(f"Cluster {cluster_id} | edges={int(mask.sum()):,} | weight={cluster_weight[cluster_id]:.0f}")

    for panel_idx in range(n_panels, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].set_axis_off()

    fig.tight_layout()
    return fig


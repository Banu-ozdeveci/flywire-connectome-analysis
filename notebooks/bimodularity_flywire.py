from __future__ import annotations

"""
Bimodularity community detection on a directed FlyWire subgraph.

Implements the "bimodularity" idea from Cionca et al. by treating directed
communities as pairs of sender/receiver sets and scoring them against an
out-in configuration null model:

    B_ij = A_ij - gamma * (k_out(i) * k_in(j)) / m
    Q_bi = (1/m) * sum_{i,j} B_ij * 1[send(i) == recv(j)]

We obtain a sender/receiver partition by:
  1) computing the SVD of B (asymmetric modularity matrix),
  2) embedding edges (i->j) with features ~ s_l * u_il * v_jl,
  3) clustering edges with K-means,
  4) assigning each node to the edge-cluster it most strongly participates in
     as a sender (outgoing) and receiver (incoming).

The script also runs conventional modularity on the symmetrized graph (Louvain)
and saves comparison plots to `output/bimodularity/`.
"""

import argparse
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity


class SliceInfo(NamedTuple):
    nodes: list[int]
    reason: str


@dataclass(frozen=True)
class BimodularityResult:
    send_label: np.ndarray  # shape (n,)
    recv_label: np.ndarray  # shape (n,)
    edge_cluster: np.ndarray  # shape (m_edges,)
    n_clusters: int
    q_bimod: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_flywire_subgraph_pickle(path: Path) -> nx.DiGraph:
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.DiGraph):
        raise TypeError(f"Expected nx.DiGraph in {path}, got {type(graph)}")
    return graph


def symmetrize_to_undirected(graph: nx.DiGraph, weight: str = "weight") -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        w = float(data.get(weight, 1.0))
        if undirected.has_edge(u, v):
            undirected[u][v][weight] += w
        else:
            undirected.add_edge(u, v, **{weight: w})
    return undirected


def adjacency_from_digraph(graph: nx.DiGraph, nodes: list[int], weight: str = "weight") -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)), dtype=float)
    for u, v, data in graph.edges(data=True):
        if u in idx and v in idx:
            a[idx[u], idx[v]] += float(data.get(weight, 1.0))
    return a


def directed_modularity_matrix_outin(a: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square adjacency, got shape {a.shape}")
    m = float(a.sum())
    if m <= 0:
        return np.zeros_like(a)
    kout = a.sum(axis=1, keepdims=True)
    kin = a.sum(axis=0, keepdims=True)
    expected = gamma * (kout @ kin) / m
    return a - expected


def directed_modularity_score(a: np.ndarray, labels: np.ndarray, gamma: float = 1.0) -> float:
    b = directed_modularity_matrix_outin(a, gamma=gamma)
    m = float(a.sum())
    if m <= 0:
        return 0.0
    same = labels[:, None] == labels[None, :]
    return float(b[same].sum() / m)


def bimodularity_score(a: np.ndarray, send_label: np.ndarray, recv_label: np.ndarray, gamma: float = 1.0) -> float:
    b = directed_modularity_matrix_outin(a, gamma=gamma)
    m = float(a.sum())
    if m <= 0:
        return 0.0
    same = send_label[:, None] == recv_label[None, :]
    return float(b[same].sum() / m)


def _kmeanspp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=float)
    first = rng.integers(0, n)
    centroids[0] = x[first]
    d2 = np.full(n, np.inf, dtype=float)
    for ci in range(1, k):
        d2 = np.minimum(d2, ((x - centroids[ci - 1]) ** 2).sum(axis=1))
        tot = float(d2.sum())
        if tot <= 0:
            centroids[ci] = x[rng.integers(0, n)]
            continue
        probs = d2 / tot
        centroids[ci] = x[int(rng.choice(n, p=probs))]
    return centroids


def kmeans(
    x: np.ndarray,
    k: int,
    *,
    sample_weight: np.ndarray | None = None,
    n_init: int = 8,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if k < 2:
        raise ValueError("k must be >= 2")
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    n = x.shape[0]
    if n < k:
        raise ValueError("Need at least k samples")
    w = np.ones(n, dtype=float) if sample_weight is None else sample_weight.astype(float, copy=False)
    w = np.clip(w, 0.0, np.inf)
    if w.sum() <= 0:
        w = np.ones(n, dtype=float)

    best_labels = None
    best_centroids = None
    best_inertia = np.inf

    rng_master = np.random.default_rng(seed)
    for _ in range(n_init):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        centroids = _kmeanspp_init(x, k, rng)
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1).astype(int)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            new_centroids = centroids.copy()
            for ci in range(k):
                mask = labels == ci
                if not np.any(mask):
                    new_centroids[ci] = x[rng.integers(0, n)]
                    continue
                ww = w[mask][:, None]
                new_centroids[ci] = (x[mask] * ww).sum(axis=0) / ww.sum()

            shift = float(np.max(np.abs(new_centroids - centroids)))
            centroids = new_centroids
            if shift < tol:
                break

        inertia = float((((x - centroids[labels]) ** 2).sum(axis=1) * w).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, best_inertia


def _community_quality_directed(graph: nx.DiGraph, nodes: set[int], weight: str = "weight") -> dict[str, float]:
    n = len(nodes)
    if n <= 1:
        return {"n": float(n), "internal_weight": 0.0, "cut_out_weight": 0.0, "density": 0.0, "conductance_out": 1.0}
    internal = 0.0
    out_total = 0.0
    for u in nodes:
        for _, v, data in graph.out_edges(u, data=True):
            w = float(data.get(weight, 1.0))
            out_total += w
            if v in nodes:
                internal += w
    cut_out = max(0.0, out_total - internal)
    density = internal / (n * (n - 1))
    conductance_out = cut_out / out_total if out_total > 0 else 1.0
    return {
        "n": float(n),
        "internal_weight": internal,
        "cut_out_weight": cut_out,
        "density": density,
        "conductance_out": conductance_out,
    }


def pick_slice_via_louvain(
    graph_full: nx.DiGraph,
    *,
    target_min: int = 100,
    target_max: int = 200,
    seed: int = 0,
    weight: str = "weight",
) -> SliceInfo:
    und_full = symmetrize_to_undirected(graph_full, weight=weight)
    communities = louvain_communities(und_full, weight=weight, seed=seed)

    scored: list[tuple[float, set[int], dict[str, float]]] = []
    for comm in communities:
        nodes = set(int(n) for n in comm)
        metrics = _community_quality_directed(graph_full, nodes, weight=weight)
        n = int(metrics["n"])
        if target_min <= n <= target_max:
            score = float(metrics["density"]) * (1.0 - float(metrics["conductance_out"]))
            scored.append((score, nodes, metrics))

    if scored:
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_nodes, metrics = scored[0]
        reason = (
            f"louvain community (n={int(metrics['n'])}, density={metrics['density']:.4g}, "
            f"conductance_out={metrics['conductance_out']:.3f}, score={best_score:.4g})"
        )
        return SliceInfo(nodes=sorted(best_nodes), reason=reason)

    # Fallback: pick the closest-sized Louvain community and trim/expand to target range.
    comms_sorted = sorted((set(int(n) for n in c) for c in communities), key=len, reverse=True)
    if not comms_sorted:
        raise RuntimeError("No communities found by Louvain")

    target_mid = (target_min + target_max) / 2.0
    base = min(comms_sorted, key=lambda c: abs(len(c) - target_mid))
    nodes = set(base)

    def internal_strength(nid: int) -> float:
        s = 0.0
        for u, v, data in graph_full.in_edges(nid, data=True):
            if u in nodes:
                s += float(data.get(weight, 1.0))
        for u, v, data in graph_full.out_edges(nid, data=True):
            if v in nodes:
                s += float(data.get(weight, 1.0))
        return s

    if len(nodes) > target_max:
        ranked = sorted(nodes, key=internal_strength, reverse=True)
        nodes = set(ranked[:target_max])
        reason = f"trimmed louvain community to n={len(nodes)} by internal strength"
        return SliceInfo(nodes=sorted(nodes), reason=reason)

    while len(nodes) < target_min:
        candidate_strength: dict[int, float] = defaultdict(float)
        for u in list(nodes):
            for _, v, data in graph_full.out_edges(u, data=True):
                if v not in nodes:
                    candidate_strength[int(v)] += float(data.get(weight, 1.0))
            for v, _, data in graph_full.in_edges(u, data=True):
                if v not in nodes:
                    candidate_strength[int(v)] += float(data.get(weight, 1.0))
        if not candidate_strength:
            break
        best = max(candidate_strength.items(), key=lambda t: t[1])[0]
        nodes.add(best)

    reason = f"expanded louvain community to n={len(nodes)} by neighbor strength"
    return SliceInfo(nodes=sorted(nodes), reason=reason)


def edge_features_from_svd(
    u: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    rank: int,
) -> np.ndarray:
    k = min(rank, u.shape[1], v.shape[1], s.shape[0])
    u_k = u[:, :k]
    v_k = v[:, :k]
    s_k = s[:k]
    features = np.zeros((len(edges), k), dtype=float)
    for ei, (i, j) in enumerate(edges):
        features[ei] = s_k * u_k[i] * v_k[j]
    return features


def bimodularity_partition_via_edge_kmeans(
    a: np.ndarray,
    *,
    svd_rank: int = 6,
    n_clusters: int = 6,
    min_edge_weight: float = 1.0,
    max_edges: int | None = 8000,
    n_init: int = 10,
    seed: int = 0,
    gamma: float = 1.0,
) -> BimodularityResult:
    n = a.shape[0]
    edges: list[tuple[int, int]] = []
    edge_w: list[float] = []
    for i in range(n):
        js = np.where(a[i] >= min_edge_weight)[0]
        for j in js:
            edges.append((i, int(j)))
            edge_w.append(float(a[i, j]))

    if not edges:
        raise ValueError("No edges matched min_edge_weight; lower threshold or choose a denser slice.")

    if max_edges is not None and len(edges) > max_edges:
        idx = np.argsort(edge_w)[::-1][:max_edges]
        edges = [edges[i] for i in idx]
        edge_w = [edge_w[i] for i in idx]

    b = directed_modularity_matrix_outin(a, gamma=gamma)
    u, s, vh = np.linalg.svd(b, full_matrices=False)
    v = vh.T

    x = edge_features_from_svd(u, s, v, edges, rank=svd_rank)
    labels, _, _ = kmeans(
        x,
        n_clusters,
        sample_weight=np.asarray(edge_w, dtype=float),
        n_init=n_init,
        seed=seed,
    )

    out_weight_by_cluster = np.zeros((n, n_clusters), dtype=float)
    in_weight_by_cluster = np.zeros((n, n_clusters), dtype=float)
    for (i, j), c, w in zip(edges, labels, edge_w, strict=True):
        out_weight_by_cluster[i, c] += w
        in_weight_by_cluster[j, c] += w

    send_label = out_weight_by_cluster.argmax(axis=1).astype(int)
    recv_label = in_weight_by_cluster.argmax(axis=1).astype(int)

    q_bimod = bimodularity_score(a, send_label, recv_label, gamma=gamma)
    return BimodularityResult(
        send_label=send_label,
        recv_label=recv_label,
        edge_cluster=labels,
        n_clusters=n_clusters,
        q_bimod=q_bimod,
    )


def _labels_from_communities(nodes: list[int], communities: Iterable[set[int]]) -> np.ndarray:
    node_to_label: dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_to_label[int(n)] = int(cid)
    return np.asarray([node_to_label[int(n)] for n in nodes], dtype=int)


def _coarsened_weight_matrix(a: np.ndarray, row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    r = int(row_labels.max()) + 1
    c = int(col_labels.max()) + 1
    m = np.zeros((r, c), dtype=float)
    col_labels_int = col_labels.astype(int, copy=False)
    for i in range(a.shape[0]):
        ri = int(row_labels[i])
        m[ri] += np.bincount(col_labels_int, weights=a[i], minlength=c)
    return m


def _order_by_label_then_strength(a: np.ndarray, labels: np.ndarray) -> np.ndarray:
    strength = a.sum(axis=0) + a.sum(axis=1)
    return np.lexsort((-strength, labels))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Community detection on a FlyWire subgraph using bimodularity (directed) vs conventional modularity."
    )
    parser.add_argument("--graph-pkl", type=str, default="data/preprocessed/subgraph.pkl")
    parser.add_argument("--nodes-min", type=int, default=100)
    parser.add_argument("--nodes-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--svd-rank", type=int, default=6)
    parser.add_argument("--min-edge-weight", type=float, default=2.0)
    parser.add_argument("--max-edges", type=int, default=8000)
    parser.add_argument("--clusters-min", type=int, default=3)
    parser.add_argument("--clusters-max", type=int, default=9)
    parser.add_argument("--clusters", type=int, default=0, help="Fix number of bimodularity edge-clusters (0=auto).")
    args = parser.parse_args()

    root = _repo_root()
    graph_path = (root / args.graph_pkl).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    graph_full = load_flywire_subgraph_pickle(graph_path)
    slice_info = pick_slice_via_louvain(
        graph_full, target_min=args.nodes_min, target_max=args.nodes_max, seed=args.seed, weight="weight"
    )

    graph_slice = graph_full.subgraph(slice_info.nodes).copy()
    nodes = list(graph_slice.nodes())
    a = adjacency_from_digraph(graph_slice, nodes, weight="weight")

    und = symmetrize_to_undirected(graph_slice, weight="weight")
    mod_comms = louvain_communities(und, weight="weight", seed=args.seed)
    mod_q_und = modularity(und, mod_comms, weight="weight")
    mod_labels = _labels_from_communities(nodes, mod_comms)
    mod_q_dir = directed_modularity_score(a, mod_labels, gamma=args.gamma)
    mod_within_frac = float(a[mod_labels[:, None] == mod_labels[None, :]].sum() / max(1.0, a.sum()))

    if args.clusters and args.clusters >= 2:
        bimod_best = bimodularity_partition_via_edge_kmeans(
            a,
            svd_rank=args.svd_rank,
            n_clusters=args.clusters,
            min_edge_weight=args.min_edge_weight,
            max_edges=args.max_edges,
            seed=args.seed,
            gamma=args.gamma,
        )
    else:
        candidates: list[BimodularityResult] = []
        for k in range(args.clusters_min, args.clusters_max + 1):
            candidates.append(
                bimodularity_partition_via_edge_kmeans(
                    a,
                    svd_rank=args.svd_rank,
                    n_clusters=k,
                    min_edge_weight=args.min_edge_weight,
                    max_edges=args.max_edges,
                    seed=args.seed,
                    gamma=args.gamma,
                )
            )
        bimod_best = max(candidates, key=lambda r: r.q_bimod)

    bimod_within_frac = float(
        a[bimod_best.send_label[:, None] == bimod_best.recv_label[None, :]].sum() / max(1.0, a.sum())
    )

    nodes_df_path = root / "data/preprocessed/processed_nodes.csv"
    node_meta = None
    if nodes_df_path.exists():
        nodes_df = pd.read_csv(nodes_df_path)
        node_meta = nodes_df.set_index("id").reindex(nodes)

    out_dir = root / "output" / "bimodularity"
    out_dir.mkdir(parents=True, exist_ok=True)

    title = f"slice n={len(nodes)} ({slice_info.reason})"
    print("Selected slice:", title)
    print(f"Edges: {graph_slice.number_of_edges()}  Total weight: {a.sum():.0f}")
    print(f"Modularity (undirected Louvain on sym): Q_und={mod_q_und:.4f}")
    print(f"Directed modularity (same partition):   Q_dir={mod_q_dir:.4f}")
    print(f"Weight within communities (directed):   frac={mod_within_frac:.3f}")
    print(
        f"Bimodularity (edge-kmeans, K={bimod_best.n_clusters}): Q_bi={bimod_best.q_bimod:.4f}  "
        f"within(frac)={bimod_within_frac:.3f}"
    )
    print(f"Within-weight gain (bimod - modularity): {bimod_within_frac - mod_within_frac:+.3f}")

    if node_meta is not None and "community_label" in node_meta.columns:
        top_types = node_meta["community_label"].fillna("NA").value_counts().head(12)
        print("Top community_label in slice:")
        for label, cnt in top_types.items():
            label_str = str(label).encode("ascii", "backslashreplace").decode("ascii")
            print(f"  {cnt:>3}  {label_str}")

    # Plot adjacency re-ordered by undirected communities
    order_mod = _order_by_label_then_strength(a, mod_labels)
    a_mod = a[np.ix_(order_mod, order_mod)]

    # Plot adjacency re-ordered by sending/receiving roles (rows by send, cols by recv)
    row_order_bi = _order_by_label_then_strength(a, bimod_best.send_label)
    col_order_bi = _order_by_label_then_strength(a.T, bimod_best.recv_label)  # strength based on columns
    a_bi = a[np.ix_(row_order_bi, col_order_bi)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = np.percentile(a[a > 0], 99) if np.any(a > 0) else 1.0
    sns.heatmap(np.log1p(a_mod), ax=axes[0], cmap="mako", cbar=False, vmin=0, vmax=math.log1p(vmax))
    axes[0].set_title("Conventional modularity (sym Louvain)\nlog(1 + A) reordered")
    axes[0].set_xlabel("nodes (by community)")
    axes[0].set_ylabel("nodes (by community)")

    sns.heatmap(np.log1p(a_bi), ax=axes[1], cmap="mako", cbar=False, vmin=0, vmax=math.log1p(vmax))
    axes[1].set_title("Bimodularity (sending vs receiving)\nlog(1 + A) reordered")
    axes[1].set_xlabel("receivers (by community)")
    axes[1].set_ylabel("senders (by community)")

    fig_adj_path = out_dir / "adjacency_reordered.png"
    fig.savefig(fig_adj_path, dpi=200)
    plt.close(fig)

    # Coarsened block matrices
    block_mod = _coarsened_weight_matrix(a, mod_labels, mod_labels)
    block_bi = _coarsened_weight_matrix(a, bimod_best.send_label, bimod_best.recv_label)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.heatmap(block_mod, ax=axes[0], cmap="rocket_r", square=True)
    axes[0].set_title("Directed weight between Louvain communities")
    axes[0].set_xlabel("to community")
    axes[0].set_ylabel("from community")

    sns.heatmap(block_bi, ax=axes[1], cmap="rocket_r", square=True)
    axes[1].set_title("Directed weight between bimodularity roles\n(send cluster -> recv cluster)")
    axes[1].set_xlabel("to (recv cluster)")
    axes[1].set_ylabel("from (send cluster)")

    fig_block_path = out_dir / "block_matrices.png"
    fig.savefig(fig_block_path, dpi=200)
    plt.close(fig)

    # Optional: summarize role split
    role_mismatch = float(np.mean(bimod_best.send_label != bimod_best.recv_label))
    summary = pd.DataFrame(
        {
            "node_id": nodes,
            "send_cluster": bimod_best.send_label,
            "recv_cluster": bimod_best.recv_label,
            "mod_cluster": mod_labels,
        }
    )
    if node_meta is not None and "name" in node_meta.columns:
        summary["name"] = node_meta["name"].astype(str).values
    if node_meta is not None and "community_label" in node_meta.columns:
        summary["community_label"] = node_meta["community_label"].astype(str).values
    summary_path = out_dir / "slice_node_assignments.csv"
    summary.to_csv(summary_path, index=False)

    # Per-bicommunity contribution
    b = directed_modularity_matrix_outin(a, gamma=args.gamma)
    per_cluster_rows = []
    m_total = float(a.sum())
    for c in range(bimod_best.n_clusters):
        send_mask = bimod_best.send_label == c
        recv_mask = bimod_best.recv_label == c
        contrib = float(b[np.ix_(send_mask, recv_mask)].sum() / m_total) if m_total > 0 else 0.0
        per_cluster_rows.append(
            {
                "cluster": c,
                "n_senders": int(send_mask.sum()),
                "n_receivers": int(recv_mask.sum()),
                "q_contrib": contrib,
            }
        )
    per_cluster_df = pd.DataFrame(per_cluster_rows).sort_values("q_contrib", ascending=False)
    per_cluster_path = out_dir / "bimodularity_cluster_contributions.csv"
    per_cluster_df.to_csv(per_cluster_path, index=False)

    print(f"Role mismatch (send!=recv): {role_mismatch:.3f}")
    print("Wrote:", str(fig_adj_path))
    print("Wrote:", str(fig_block_path))
    print("Wrote:", str(summary_path))
    print("Wrote:", str(per_cluster_path))


if __name__ == "__main__":
    main()

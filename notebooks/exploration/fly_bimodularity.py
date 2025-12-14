import argparse
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


def modularity_matrix(A: np.ndarray, null_model: str = "outin") -> np.ndarray:
    kout = A.sum(axis=1, keepdims=True)
    kin = A.sum(axis=0, keepdims=True)
    m = A.sum()
    if m == 0:
        return np.zeros_like(A)
    if null_model != "outin":
        raise ValueError("Only 'outin' null model supported here")
    P = kout @ kin / m
    return A - P


def sorted_svd(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    order = np.argsort(S)[::-1]
    return U[:, order], S[order], Vt[order].T


def edge_bicommunities(
    B: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    n_vectors: int,
    n_clusters: int,
    random_state: int,
):
    u_emb = U[:, :n_vectors]
    v_emb = V[:, :n_vectors]
    send_idx, recv_idx = np.nonzero(B)
    feats = np.hstack([u_emb[send_idx], v_emb[recv_idx]])
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(feats)
    return send_idx, recv_idx, labels, km


def get_node_communities(edge_u, edge_v, edge_labels, n_nodes: int):
    send_comm = np.zeros(n_nodes, dtype=int)
    recv_comm = np.zeros(n_nodes, dtype=int)
    for node in range(n_nodes):
        outgoing = edge_labels[edge_u == node]
        incoming = edge_labels[edge_v == node]
        if len(outgoing):
            send_comm[node] = np.bincount(outgoing).argmax()
        if len(incoming):
            recv_comm[node] = np.bincount(incoming).argmax()
    return send_comm, recv_comm


def bimodularity_index(A: np.ndarray, send_comm: np.ndarray, recv_comm: np.ndarray) -> np.ndarray:
    m = A.sum()
    scores = np.zeros(len(send_comm))
    for i in range(len(send_comm)):
        mask_out = send_comm == send_comm[i]
        mask_in = recv_comm == recv_comm[i]
        block_weight = A[np.ix_(mask_out, mask_in)].sum()
        if m > 0:
            scores[i] = block_weight / m
    return scores


def build_adjacency(connections: pd.DataFrame, top_n_nodes: int, weight_col: str):
    out_strength = connections.groupby("pre_root_id")[weight_col].sum()
    in_strength = connections.groupby("post_root_id")[weight_col].sum()
    strength = out_strength.add(in_strength, fill_value=0)

    if top_n_nodes is not None:
        keep_nodes = strength.sort_values(ascending=False).head(top_n_nodes).index
        filtered = connections[
            connections["pre_root_id"].isin(keep_nodes)
            & connections["post_root_id"].isin(keep_nodes)
        ].copy()
    else:
        keep_nodes = strength.index
        filtered = connections.copy()

    node_list = pd.Index(sorted(keep_nodes))
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    filtered["pre_idx"] = filtered["pre_root_id"].map(node_to_idx)
    filtered["post_idx"] = filtered["post_root_id"].map(node_to_idx)

    n = len(node_list)
    adj = np.zeros((n, n), dtype=float)
    for _, row in filtered.iterrows():
        adj[int(row["pre_idx"]), int(row["post_idx"])] += row[weight_col]

    return adj, node_list, strength.loc[node_list], filtered


def plot_spectrum(S, U, V, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(S[:60], marker="o")
    axes[0].set_title("Singular values (top 60)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("sigma")

    axes[1].plot(U[:, 0], label="u1")
    axes[1].plot(V[:, 0], label="v1", alpha=0.7)
    axes[1].set_title("Leading sending/receiving components")
    axes[1].legend()

    axes[2].scatter(U[:, 0], V[:, 0], s=10, alpha=0.5)
    axes[2].set_title("u1 vs v1 (nodes)")
    axes[2].set_xlabel("u1")
    axes[2].set_ylabel("v1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_edge_clusters(edge_feats, edge_labels, n_clusters: int, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter = axes[0].scatter(
        edge_feats[:, 0],
        edge_feats[:, 1],
        c=edge_labels,
        cmap="tab10",
        alpha=0.5,
        s=10,
    )
    axes[0].set_title("Edge embedding (first two components)")
    axes[0].set_xlabel("u1")
    axes[0].set_ylabel("v1")
    axes[0].legend(*scatter.legend_elements(), title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    sns.histplot(edge_labels, ax=axes[1], bins=n_clusters, discrete=True)
    axes[1].set_title("Edge cluster sizes")
    axes[1].set_xlabel("Cluster id")
    axes[1].set_ylabel("Edges")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bimodularity(bimod_scores, score_df, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(bimod_scores, bins=40, ax=axes[0])
    axes[0].set_title("Bimodularity score distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Nodes")

    axes[1].scatter(
        score_df["send_comm"],
        score_df["recv_comm"],
        c=score_df["bimod_score"],
        cmap="viridis",
        s=12,
        alpha=0.6,
    )
    axes[1].set_title("Send vs receive communities (colored by score)")
    axes[1].set_xlabel("Sending community")
    axes[1].set_ylabel("Receiving community")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sorted_adjacency(adj_sorted, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.log1p(adj_sorted), cmap="magma", cbar_kws={"label": "log(1+synapses)"}, ax=ax)
    ax.set_title("Adjacency sorted by sending/receiving communities")
    ax.set_xlabel("Post neuron (sorted)")
    ax.set_ylabel("Pre neuron (sorted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_subgraph(adj, send_comm, recv_comm, node_list, random_state: int, out_path: Path):
    largest_send = np.bincount(send_comm).argmax()
    mask_send = send_comm == largest_send
    sub_idx = np.where(mask_send)[0]
    sub_nodes = node_list[sub_idx]
    if len(sub_idx) == 0:
        return

    G_sub = nx.from_numpy_array(adj[np.ix_(sub_idx, sub_idx)], create_using=nx.DiGraph)
    pos = nx.spring_layout(G_sub, weight="weight", seed=random_state, k=0.6)
    node_colors = recv_comm[sub_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_size=80, node_color=node_colors, cmap="tab10", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.25, arrows=False, ax=ax)
    ax.set_title("Largest sending community; colored by receiving community")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def configure_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def main():
    parser = argparse.ArgumentParser(description="Fly connectome bimodularity analysis")
    parser.add_argument("--data-path", type=Path, default=Path("../data/raw/connections_princeton.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("../output/fly_bimodularity"))
    parser.add_argument("--top-n-nodes", type=int, default=1000, help="Number of strongest nodes to keep (None for all)")
    parser.add_argument("--n-clusters", type=int, default=6, help="Edge bicommunity clusters")
    parser.add_argument("--n-vectors", type=int, default=4, help="Singular vectors to embed edges")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--edge-weight-col", type=str, default="syn_count")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "fly_bimodularity.log"
    configure_logging(log_path)

    logging.info("Loading connections from %s", args.data_path)
    connections = pd.read_csv(args.data_path)
    cols = ["pre_root_id", "post_root_id", args.edge_weight_col]
    extra_cols = [c for c in ["neuropil", "nt_type"] if c in connections.columns]
    connections = connections[cols + extra_cols]

    logging.info("Building adjacency with top_n_nodes=%s", args.top_n_nodes)
    adj, node_list, strength, filtered = build_adjacency(
        connections, args.top_n_nodes, args.edge_weight_col
    )
    logging.info("Nodes kept: %d | edges kept: %d | total synapses: %.0f", len(node_list), len(filtered), filtered[args.edge_weight_col].sum())
    logging.info("Top 10 nodes by strength:\n%s", strength.sort_values(ascending=False).head(10))

    logging.info("Computing directed modularity and SVD")
    B = modularity_matrix(adj, null_model="outin")
    U, S, V = sorted_svd(B)

    logging.info("Clustering edges: n_clusters=%d, n_vectors=%d", args.n_clusters, args.n_vectors)
    edge_u, edge_v, edge_labels, _ = edge_bicommunities(
        B, U, V, n_vectors=args.n_vectors, n_clusters=args.n_clusters, random_state=args.random_seed
    )
    send_comm, recv_comm = get_node_communities(edge_u, edge_v, edge_labels, n_nodes=len(node_list))
    logging.info("Edge cluster sizes: %s", np.bincount(edge_labels))
    logging.info("Sending communities: %s", np.bincount(send_comm))
    logging.info("Receiving communities: %s", np.bincount(recv_comm))

    logging.info("Computing bimodularity scores")
    bimod_scores = bimodularity_index(adj, send_comm, recv_comm)
    score_df = pd.DataFrame(
        {
            "node": node_list,
            "send_comm": send_comm,
            "recv_comm": recv_comm,
            "bimod_score": bimod_scores,
        }
    ).sort_values("bimod_score", ascending=False)
    score_df_path = args.output_dir / "bimodularity_scores.csv"
    score_df.to_csv(score_df_path, index=False)
    logging.info("Saved bimodularity scores to %s", score_df_path)

    logging.info("Saving figures")
    plot_spectrum(S, U, V, args.output_dir / "spectrum.png")

    edge_feats = np.hstack([U[edge_u, : args.n_vectors], V[edge_v, : args.n_vectors]])
    plot_edge_clusters(edge_feats, edge_labels, args.n_clusters, args.output_dir / "edge_clusters.png")

    plot_bimodularity(bimod_scores, score_df, args.output_dir / "bimodularity_scores.png")

    order = np.argsort(send_comm * 100 + recv_comm)
    adj_sorted = adj[np.ix_(order, order)]
    plot_sorted_adjacency(adj_sorted, args.output_dir / "adjacency_sorted.png")

    plot_subgraph(adj, send_comm, recv_comm, node_list, args.random_seed, args.output_dir / "largest_sending_subgraph.png")

    logging.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    sns.set_context("talk")
    sns.set_style("whitegrid")
    main()

"""
Spatial MRF post-processing for cluster label refinement.
Uses Iterated Conditional Modes (ICM) to encourage spatially
neighbouring segments (at nearby Locs with overlapping depth)
to share cluster labels, weighted by horizontal distance and
depth overlap.

When use_spatial_mrf is False (default), this module is never imported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .config import Config


def load_locations(config: Config) -> dict[str, tuple[float, float, float]]:
    """Load {Target: (X, Y, Z)} from the location CSV."""
    path = config.project_dir / "data" / config.location_file
    df = pd.read_csv(path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    out: dict[str, tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        target = str(row["Target"]).strip()
        x = float(row["X"])
        y = float(row["Y"])
        z = float(row["Z"])
        if target not in out:
            out[target] = (x, y, z)
    return out


def _build_adjacency(
    features: pd.DataFrame,
    locations: dict[str, tuple[float, float, float]],
    config: Config,
) -> list[tuple[int, int, float]]:
    """
    Build weighted edge list between segments at different Locs.
    An edge (i, j, w) exists when:
      - segments i and j belong to different Locs
      - horizontal distance < spatial_max_distance_m
      - their depth intervals overlap
    Weight combines distance decay and depth overlap fraction.
    """
    targets = features["target"].values
    tops = features["top"].values.astype(float)
    bottoms = features["bottom"].values.astype(float)
    n = len(features)

    # Map each target to its XY position
    unique_targets = list(set(targets))
    target_xy = {}
    for t in unique_targets:
        loc = locations.get(t)
        if loc is not None:
            target_xy[t] = np.array([loc[0], loc[1]])

    # Build KD-tree over unique target XY positions for fast neighbor lookup
    indexed_targets = [t for t in unique_targets if t in target_xy]
    if len(indexed_targets) < 2:
        return []

    xy_array = np.array([target_xy[t] for t in indexed_targets])
    tree = KDTree(xy_array)
    target_to_idx = {t: i for i, t in enumerate(indexed_targets)}

    # Find which Loc pairs are within max distance
    neighbor_pairs: set[tuple[str, str]] = set()
    pairs = tree.query_pairs(r=config.spatial_max_distance_m)
    for i, j in pairs:
        neighbor_pairs.add((indexed_targets[i], indexed_targets[j]))
        neighbor_pairs.add((indexed_targets[j], indexed_targets[i]))

    # Build segment-level edges for neighboring Loc pairs
    # Index segments by target for fast lookup
    target_segments: dict[str, list[int]] = {}
    for idx in range(n):
        t = targets[idx]
        target_segments.setdefault(t, []).append(idx)

    sigma2 = 2.0 * config.spatial_sigma_m ** 2
    edges: list[tuple[int, int, float]] = []

    visited: set[tuple[str, str]] = set()
    for t_a, t_b in neighbor_pairs:
        if (t_a, t_b) in visited:
            continue
        visited.add((t_a, t_b))
        visited.add((t_b, t_a))

        if t_a not in target_xy or t_b not in target_xy:
            continue
        dist = float(np.linalg.norm(target_xy[t_a] - target_xy[t_b]))
        if dist > config.spatial_max_distance_m:
            continue
        dist_weight = float(np.exp(-dist ** 2 / sigma2))

        for i in target_segments.get(t_a, []):
            for j in target_segments.get(t_b, []):
                # Compute depth overlap
                overlap = min(bottoms[i], bottoms[j]) - max(tops[i], tops[j])
                if overlap <= 0:
                    continue
                max_thick = max(bottoms[i] - tops[i], bottoms[j] - tops[j])
                overlap_frac = overlap / max_thick if max_thick > 0 else 0.0
                w = dist_weight * overlap_frac
                if w > 1e-6:
                    edges.append((i, j, w))

    return edges


def _get_unary_costs(
    estimator: object,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Compute unary cost matrix: -log P(k | x_i) for each segment i and cluster k.
    Returns shape (n_segments, n_clusters).
    """
    proba = estimator.predict_proba(X_scaled)  # (n, k)
    # Clip to avoid log(0)
    proba = np.clip(proba, 1e-10, 1.0)
    return -np.log(proba)


def _icm_solve(
    unary: np.ndarray,
    edges: list[tuple[int, int, float]],
    lam: float,
    max_iter: int = 20,
) -> np.ndarray:
    """
    Iterated Conditional Modes (ICM) to minimize:
      E(L) = sum_i unary[i, L_i] + lambda * sum_{(i,j,w)} w * 1[L_i != L_j]

    Initializes from the MAP (argmin unary) labels.
    """
    n, k = unary.shape
    labels = np.argmin(unary, axis=1).copy()

    # Pre-build neighbor lists for speed
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, w in edges:
        neighbors[i].append((j, w))
        neighbors[j].append((i, w))

    for _iteration in range(max_iter):
        changed = 0
        for i in range(n):
            # Cost of each label for node i
            costs = unary[i].copy()  # shape (k,)
            for j, w in neighbors[i]:
                # Add spatial penalty for disagreeing with neighbor j
                for c in range(k):
                    if c != labels[j]:
                        costs[c] += lam * w
            best = int(np.argmin(costs))
            if best != labels[i]:
                labels[i] = best
                changed += 1
        if changed == 0:
            break

    return labels


def refine_labels_mrf(
    features: pd.DataFrame,
    cluster_result: "ClusterResult",  # noqa: F821
    feature_matrix: pd.DataFrame,
    config: Config,
) -> np.ndarray:
    """
    Main entry point. Takes the fitted cluster result and the feature
    DataFrame (with target/top/bottom columns), returns refined labels.
    """
    locations = load_locations(config)

    # Build spatial adjacency graph
    edges = _build_adjacency(features, locations, config)
    if not edges:
        # No spatial neighbors found — fall back to original predictions
        from .model import predict
        return predict(cluster_result, feature_matrix)

    # Get scaled data for probability computation
    X_scaled = cluster_result.pipeline.transform(feature_matrix)

    # Unary costs from GMM posterior
    unary = _get_unary_costs(cluster_result.estimator, X_scaled)

    # Solve MRF via ICM
    labels = _icm_solve(unary, edges, config.spatial_lambda)

    return labels

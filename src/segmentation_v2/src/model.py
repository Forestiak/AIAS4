"""
Unsupervised clustering.
Supports GMM (BIC-based), KMeans, and Agglomerative Clustering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .config import Config


@dataclass
class ClusterResult:
    pipeline: Pipeline
    estimator: object
    n_clusters: int
    model_type: str
    bic: float | None = None


def fit(feature_matrix: pd.DataFrame, config: Config) -> ClusterResult:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
    pca_components = _effective_pca_components(config.pca_n_components, feature_matrix.shape)
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=config.random_state)))
    pipe = Pipeline(steps)
    X = pipe.fit_transform(feature_matrix)

    builders = {
        "gmm": _fit_gmm,
        "kmeans": _fit_kmeans,
        "agglomerative": _fit_agglomerative,
    }
    builder = builders.get(config.model_type)
    if builder is None:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    return builder(X, pipe, config)


def predict(result: ClusterResult, feature_matrix: pd.DataFrame) -> np.ndarray:
    X = result.pipeline.transform(feature_matrix)
    if result.model_type == "agglomerative":
        return result.estimator.fit_predict(X)
    return result.estimator.predict(X)


def predict_proba(result: ClusterResult, feature_matrix: pd.DataFrame) -> np.ndarray:
    """Return soft cluster probabilities (n_samples, n_clusters). GMM only."""
    X = result.pipeline.transform(feature_matrix)
    if not hasattr(result.estimator, "predict_proba"):
        raise TypeError(f"Spatial MRF requires a model with predict_proba (got {result.model_type})")
    return result.estimator.predict_proba(X)


# ── GMM (select k by BIC) ────────────────────────────────────────────

def _fit_gmm(X: np.ndarray, pipe: Pipeline, config: Config) -> ClusterResult:
    n_samples = X.shape[0]
    if config.cluster_selection == "fixed":
        if config.n_clusters is None or config.n_clusters < 1:
            raise ValueError("Fixed GMM cluster selection requires parameters.n_clusters >= 1.")
        if config.n_clusters > n_samples:
            raise ValueError(
                f"Fixed GMM cluster selection requires n_clusters <= n_samples (got {config.n_clusters} > {n_samples})."
            )
        gmm = _build_gmm(config.n_clusters, config)
        gmm.fit(X)
        return ClusterResult(
            pipeline=pipe,
            estimator=gmm,
            n_clusters=config.n_clusters,
            model_type="gmm",
            bic=float(gmm.bic(X)),
        )

    if config.cluster_selection != "bic":
        raise ValueError(f"Unknown cluster_selection for GMM: {config.cluster_selection}")

    min_clusters = max(1, config.min_clusters)
    max_clusters = min(config.max_clusters, n_samples)
    if min_clusters > max_clusters:
        raise ValueError(
            f"GMM BIC search requires min_clusters <= min(max_clusters, n_samples) "
            f"(got min_clusters={config.min_clusters}, max_clusters={config.max_clusters}, n_samples={n_samples})."
        )

    best: ClusterResult | None = None
    for k in range(min_clusters, max_clusters + 1):
        gmm = _build_gmm(k, config)
        gmm.fit(X)
        bic = float(gmm.bic(X))
        if best is None or bic < best.bic:
            best = ClusterResult(pipeline=pipe, estimator=gmm, n_clusters=k, model_type="gmm", bic=bic)
    return best


# ── KMeans ────────────────────────────────────────────────────────────

def _fit_kmeans(X: np.ndarray, pipe: Pipeline, config: Config) -> ClusterResult:
    best: ClusterResult | None = None
    for k in range(config.min_clusters, config.max_clusters + 1):
        km = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
        km.fit(X)
        inertia = float(km.inertia_)
        if best is None or inertia < best.bic:
            best = ClusterResult(pipeline=pipe, estimator=km, n_clusters=k, model_type="kmeans", bic=inertia)
    return best


# ── Agglomerative ─────────────────────────────────────────────────────

def _fit_agglomerative(X: np.ndarray, pipe: Pipeline, config: Config) -> ClusterResult:
    k = config.max_clusters
    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(X)
    return ClusterResult(pipeline=pipe, estimator=agg, n_clusters=k, model_type="agglomerative")


def _build_gmm(k: int, config: Config) -> GaussianMixture:
    return GaussianMixture(
        n_components=k,
        covariance_type=config.covariance_type,
        random_state=config.random_state,
        reg_covar=config.reg_covar,
    )


def _effective_pca_components(requested: int | None, shape: tuple[int, int]) -> int | None:
    if requested is None:
        return None
    n_samples, n_features = shape
    max_components = min(n_features, n_samples - 1)
    if max_components < 1:
        return None
    return min(requested, max_components)

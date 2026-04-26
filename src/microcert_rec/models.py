"""Two-tower fit + persistence.

Collaborative tower: TruncatedSVD on the implicit interaction matrix.
Content tower: TF-IDF over `skills_taught`. Both saved to ``models/two_tower.joblib``.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .data import PROCESSED, load_all
from .features import build_interaction_matrix, fit_cert_tfidf

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def fit_collaborative_tower(R, k: int = 32, seed: int = 42):
    """TruncatedSVD on R. Returns (svd, U_factors, V_factors)."""
    svd = TruncatedSVD(n_components=k, random_state=seed)
    # U_k * sqrt(Sigma_k)  (we recompute via fit_transform on R, then factor V from components_)
    U_sigma = svd.fit_transform(R)
    sigma = svd.singular_values_
    sqrt_sigma = np.sqrt(np.maximum(sigma, 1e-9))
    # Normalise to balance the factor scale on both sides
    U = U_sigma / sqrt_sigma
    V = svd.components_.T * sqrt_sigma
    return svd, U.astype(np.float32), V.astype(np.float32)


def fit(
    learners: pd.DataFrame, certs: pd.DataFrame, interactions: pd.DataFrame, k: int = 32
) -> dict:
    R, learner_ids, cert_ids = build_interaction_matrix(learners, certs, interactions)
    svd, U, V = fit_collaborative_tower(R, k=k)
    vec, X_certs = fit_cert_tfidf(certs)

    return dict(
        svd=svd,
        U=U,
        V=V,
        tfidf=vec,
        X_certs=X_certs,
        learner_ids=learner_ids,
        cert_ids=cert_ids,
    )


def save(obj, name: str = "two_tower.joblib") -> Path:
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return path


def load(name: str = "two_tower.joblib"):
    return joblib.load(MODEL_DIR / name)


def main():
    learners, certs, interactions = load_all()
    artefacts = fit(learners, certs, interactions, k=32)
    path = save(artefacts)
    print(f"Saved two-tower artefacts -> {path}")
    print(f"  CF: {artefacts['U'].shape[0]} learners x k={artefacts['U'].shape[1]} factors")
    print(f"  CF: {artefacts['V'].shape[0]} certs    x k={artefacts['V'].shape[1]} factors")
    print(f"  Content: TF-IDF vocab={len(artefacts['tfidf'].vocabulary_)}, X_certs={artefacts['X_certs'].shape}")


if __name__ == "__main__":
    main()

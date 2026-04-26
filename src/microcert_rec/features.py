"""Feature builders: implicit interaction matrix + TF-IDF over `skills_taught`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Per-event weight in the implicit matrix
EVENT_WEIGHT = {"enrolled": 1.0, "completed": 2.0, "rated": 0.0}  # ratings handled separately


def build_interaction_matrix(
    learners: pd.DataFrame, certs: pd.DataFrame, interactions: pd.DataFrame
) -> tuple[csr_matrix, list[str], list[str]]:
    """Aggregate (learner, cert) events into a positive-only implicit matrix."""
    learner_ids = learners["learner_id"].tolist()
    cert_ids = certs["cert_id"].tolist()
    learner_idx = {l: i for i, l in enumerate(learner_ids)}
    cert_idx = {c: i for i, c in enumerate(cert_ids)}

    rows, cols, vals = [], [], []
    for r in interactions.itertuples(index=False):
        l = learner_idx.get(r.learner_id)
        c = cert_idx.get(r.cert_id)
        if l is None or c is None:
            continue
        if r.event_type == "rated":
            v = float(r.rating)  # 1..5
        else:
            v = float(EVENT_WEIGHT[r.event_type])
        rows.append(l)
        cols.append(c)
        vals.append(v)
    R = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(learner_ids), len(cert_ids)),
        dtype=np.float32,
    )
    return R, learner_ids, cert_ids


def fit_cert_tfidf(certs: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray]:
    """TF-IDF over the `skills_taught` text. Returns the fitted vectoriser + cert matrix."""
    docs = certs["skills_taught"].fillna("").tolist()
    vec = TfidfVectorizer(
        token_pattern=r"[A-Za-z][A-Za-z\-+#./ ]+",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=512,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs).astype(np.float32)
    return vec, X.toarray()


def learner_text(skill_set: list[str] | set[str]) -> str:
    """Render a learner's skill set into the same text shape as `skills_taught`."""
    return ", ".join(sorted(skill_set))

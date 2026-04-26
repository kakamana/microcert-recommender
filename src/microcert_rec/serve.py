"""Inference: combine collaborative + content towers, return top-K with reasons."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import models
from .features import learner_text

DATA_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


@lru_cache(maxsize=1)
def _load() -> dict:
    art = models.load()
    art["certs"] = pd.read_parquet(DATA_PROC / "certs.parquet")
    art["learner_index"] = {l: i for i, l in enumerate(art["learner_ids"])}
    return art


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    a = a.reshape(1, -1)
    a = a / (np.linalg.norm(a) + eps)
    bn = np.linalg.norm(b, axis=1, keepdims=True) + eps
    bnorm = b / bn
    return (a @ bnorm.T).ravel()


def recommend(
    learner_skills: list[str],
    learner_id: str | None = None,
    k: int = 10,
    beta: float = 0.6,
) -> list[dict]:
    """Top-K cert recommendations.

    - If `learner_id` exists in the trained matrix, use the collaborative factor.
    - Otherwise fall back to the content tower only (cold start).

    Returns list of dicts: {cert_id, title, issuer, hours, cost, score, reason}.
    """
    art = _load()
    certs: pd.DataFrame = art["certs"]
    V = art["V"]
    X_certs = art["X_certs"]
    vec = art["tfidf"]

    # Content side: embed learner text in cert TF-IDF space
    skill_text = learner_text(learner_skills)
    learner_v = vec.transform([skill_text]).toarray()[0]
    cs_content = _cosine(learner_v, X_certs)

    # Collaborative side
    li = art["learner_index"].get(learner_id) if learner_id else None
    if li is not None:
        u_factor = art["U"][li]
        cs_cf = _cosine(u_factor, V)
        score = beta * cs_cf + (1.0 - beta) * cs_content
        cf_used = True
    else:
        cs_cf = np.zeros_like(cs_content)
        score = cs_content
        cf_used = False

    # Top-K
    order = np.argsort(-score)[:k]
    cert_rows = certs.iloc[order].copy()

    out = []
    for rank, idx in enumerate(order):
        row = certs.iloc[int(idx)]
        cf_s = float(cs_cf[idx]) if cf_used else 0.0
        co_s = float(cs_content[idx])
        # Reason heuristic
        if cf_used and cf_s > co_s and cf_s > 0.05:
            reason = "popular among learners with similar enrolment history"
        else:
            taught = set(map(str.strip, str(row.get("skills_taught", "")).split(",")))
            overlap = sorted(taught & set(learner_skills))
            shown = ", ".join(overlap[:3]) if overlap else "your skill profile"
            reason = f"matches your {shown}"
        out.append(dict(
            cert_id=str(row["cert_id"]),
            title=str(row["title"]),
            issuer=str(row["issuer"]),
            hours=float(row["hours"]),
            cost=float(row["cost"]),
            relevance=float(score[idx]),
            cf_score=cf_s,
            content_score=co_s,
            reason=reason,
            rank=rank + 1,
        ))
    return out

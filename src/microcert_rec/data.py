"""Synthetic data generator for the micro-cert recommender.

Writes three parquet files to ``data/processed/``::

    learners.parquet       - 2,000 rows, learner_id + 40 binary skill columns
    certs.parquet          - 500 rows, cert_id + metadata + skills_taught text
    interactions.parquet   - ~25,000 rows of (learner_id, cert_id, event_type, rating, ts)

Run with::

    python -m microcert_rec.data
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED = DATA_DIR / "processed"

# 40-skill controlled vocabulary
SKILL_VOCAB: tuple[str, ...] = (
    "Python", "SQL", "Pandas", "Spark", "dbt", "Airflow", "Snowflake",
    "BigQuery", "Power BI", "Tableau", "AWS", "Azure", "GCP", "Terraform",
    "Kubernetes", "Docker", "CI-CD", "Linux", "Git", "REST APIs",
    "Scikit-learn", "PyTorch", "Deep Learning", "NLP", "Computer Vision",
    "Prompt Engineering", "MLOps", "Model Evaluation", "Data Modelling",
    "ETL Design", "Data Quality", "Communication", "Stakeholder Management",
    "Project Management", "Agile Delivery", "Change Management", "Mentoring",
    "Negotiation", "Decision Making", "Workshop Facilitation",
)

ISSUERS: tuple[str, ...] = ("Coursera", "edX", "LinkedIn Learning", "AWS Skill Builder",
                            "Google Cloud Skills Boost", "Microsoft Learn", "DataCamp",
                            "In-house Academy")

ISSUER_WEIGHTS = np.array([0.18, 0.10, 0.20, 0.12, 0.10, 0.10, 0.10, 0.10])

EVENT_TYPES = ("enrolled", "completed", "rated")

# Latent skill themes drive both learner skill clustering AND cert content,
# so the recommender has a real signal to recover.
THEMES: dict[str, list[str]] = {
    "Data": ["Python", "SQL", "Pandas", "Spark", "dbt", "Airflow", "Snowflake",
             "BigQuery", "Data Modelling", "ETL Design", "Data Quality"],
    "BI":   ["Power BI", "Tableau", "SQL", "Communication", "Stakeholder Management"],
    "Cloud": ["AWS", "Azure", "GCP", "Terraform", "Kubernetes", "Docker",
              "CI-CD", "Linux"],
    "AI-ML": ["Scikit-learn", "PyTorch", "Deep Learning", "NLP", "Computer Vision",
              "Prompt Engineering", "MLOps", "Model Evaluation", "Python"],
    "Soft":  ["Communication", "Stakeholder Management", "Project Management",
              "Agile Delivery", "Change Management", "Mentoring", "Negotiation",
              "Decision Making", "Workshop Facilitation"],
    "Eng":   ["Python", "Git", "REST APIs", "CI-CD", "Linux", "Docker"],
}
THEME_LIST = list(THEMES.keys())


def make_learners(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # Each learner mixes 1-3 themes
        chosen = rng.choice(THEME_LIST, size=int(rng.integers(1, 4)), replace=False)
        skill_pool: set[str] = set()
        for theme in chosen:
            pool = THEMES[theme]
            k = int(rng.integers(2, min(8, len(pool) + 1)))
            skill_pool.update(rng.choice(pool, size=k, replace=False).tolist())
        # Sprinkle long-tail noise
        for s in SKILL_VOCAB:
            if s not in skill_pool and rng.random() < 0.04:
                skill_pool.add(s)

        row = {f"skill__{s}": int(s in skill_pool) for s in SKILL_VOCAB}
        row["learner_id"] = f"L-{i:05d}"
        row["primary_theme"] = chosen[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["learner_id", "primary_theme"] + [c for c in df.columns if c.startswith("skill__")]
    return df[cols]


def make_certs(n: int = 500, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        theme = rng.choice(THEME_LIST)
        pool = THEMES[theme]
        k = int(rng.integers(3, min(8, len(pool) + 1)))
        skills = rng.choice(pool, size=k, replace=False).tolist()
        # Sometimes add 1-2 cross-theme skills (overlap)
        if rng.random() < 0.4:
            other = rng.choice([t for t in THEME_LIST if t != theme])
            extras = rng.choice(THEMES[other], size=int(rng.integers(1, 3)), replace=False).tolist()
            skills = list(set(skills) | set(extras))

        issuer = str(rng.choice(ISSUERS, p=ISSUER_WEIGHTS))
        hours = float(np.clip(rng.normal({"Soft": 6, "BI": 10, "Eng": 15, "Data": 20, "Cloud": 25, "AI-ML": 30}[theme], 5), 2, 80))
        cost = float(np.clip(rng.normal(40 + hours * 6, 30), 0, 800))
        title = f"{theme} Track {i:03d}: {skills[0]} + {skills[-1]}"
        rows.append(dict(
            cert_id=f"C-{i:05d}",
            title=title,
            issuer=issuer,
            theme=theme,
            skills_taught=", ".join(skills),
            hours=round(hours, 1),
            cost=round(cost, 2),
        ))
    return pd.DataFrame(rows)


def make_interactions(
    learners: pd.DataFrame, certs: pd.DataFrame, n_events: int = 25000, seed: int = 44
) -> pd.DataFrame:
    """Generate plausible interactions: learners gravitate to certs in their theme,
    with a popularity head over a few "hot" certs.
    """
    rng = np.random.default_rng(seed)
    cert_theme = certs.set_index("cert_id")["theme"].to_dict()

    # Build a popularity prior — a small set of certs gets disproportionately many enrolments
    pop_p = np.ones(len(certs))
    hot_idx = rng.choice(len(certs), size=int(0.05 * len(certs)), replace=False)
    pop_p[hot_idx] += 8.0
    pop_p = pop_p / pop_p.sum()

    cert_ids = certs["cert_id"].to_numpy()

    # Per-theme cert pool for affinity sampling
    theme_to_certs: dict[str, np.ndarray] = {
        t: certs.loc[certs["theme"] == t, "cert_id"].to_numpy()
        for t in THEME_LIST
    }

    learner_themes = learners.set_index("learner_id")["primary_theme"].to_dict()
    learner_ids = learners["learner_id"].to_numpy()

    rows = []
    base_ts = datetime(2024, 1, 1)
    for _ in range(n_events):
        learner = str(rng.choice(learner_ids))
        # 70% pick by learner's theme; 20% by global popularity; 10% uniform
        roll = rng.random()
        if roll < 0.70:
            pool = theme_to_certs[learner_themes[learner]]
            cert = str(rng.choice(pool))
        elif roll < 0.90:
            cert = str(rng.choice(cert_ids, p=pop_p))
        else:
            cert = str(rng.choice(cert_ids))

        # Event type: most are enrol; some complete; a few rated
        e_roll = rng.random()
        if e_roll < 0.55:
            event = "enrolled"
            rating = 0.0
        elif e_roll < 0.85:
            event = "completed"
            rating = 0.0
        else:
            event = "rated"
            # Ratings biased high if cert theme matches learner theme
            base = 4.2 if cert_theme[cert] == learner_themes[learner] else 3.4
            rating = float(np.clip(rng.normal(base, 0.7), 1, 5))

        ts = base_ts + timedelta(minutes=int(rng.integers(0, 60 * 24 * 365)))
        rows.append(dict(
            learner_id=learner,
            cert_id=cert,
            event_type=event,
            rating=round(rating, 2),
            ts=ts,
        ))

    df = pd.DataFrame(rows)
    df.sort_values("ts", inplace=True, ignore_index=True)
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learners = pd.read_parquet(PROCESSED / "learners.parquet")
    certs = pd.read_parquet(PROCESSED / "certs.parquet")
    interactions = pd.read_parquet(PROCESSED / "interactions.parquet")
    return learners, certs, interactions


def make_training_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    learners = make_learners()
    certs = make_certs()
    interactions = make_interactions(learners, certs)
    learners.to_parquet(PROCESSED / "learners.parquet", index=False)
    certs.to_parquet(PROCESSED / "certs.parquet", index=False)
    interactions.to_parquet(PROCESSED / "interactions.parquet", index=False)
    return learners, certs, interactions


if __name__ == "__main__":
    learners, certs, interactions = make_training_artifacts()
    print(f"learners: {len(learners):,} rows -> data/processed/learners.parquet")
    print(f"certs: {len(certs):,} rows -> data/processed/certs.parquet")
    print(f"interactions: {len(interactions):,} rows -> data/processed/interactions.parquet")

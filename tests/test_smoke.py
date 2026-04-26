from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_recommend_cold():
    r = client.post(
        "/recommend",
        json={
            "learner_id": None,
            "learner_skills": ["Python", "SQL", "Pandas"],
            "k": 5,
            "beta": 0.6,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["used_collaborative_tower"] is False
    assert len(body["items"]) == 5
    for it in body["items"]:
        assert "cert_id" in it
        assert "relevance" in it


def test_data_shapes_deterministic():
    from microcert_rec.data import make_certs, make_interactions, make_learners

    learners = make_learners(n=200, seed=42)
    certs = make_certs(n=80, seed=43)
    inter = make_interactions(learners, certs, n_events=2000, seed=44)

    assert len(learners) == 200
    assert "skill__Python" in learners.columns
    assert {"cert_id", "title", "issuer", "skills_taught", "hours", "cost"} <= set(certs.columns)
    assert {"learner_id", "cert_id", "event_type", "rating", "ts"} <= set(inter.columns)
    assert len(inter) == 2000

    # determinism
    learners2 = make_learners(n=200, seed=42)
    assert (learners == learners2).all().all()

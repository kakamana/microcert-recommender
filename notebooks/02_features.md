# Notebook 02 — Featurisation

## 1. Implicit interaction matrix
>>> `R = build_interaction_matrix(inter)` — sparse CSR.

## 2. TF-IDF over skills_taught
>>> `vec, X_certs = fit_cert_tfidf(certs)` — store the fitted vectoriser.

## 3. Learner skill text
>>> `learner_text(skills_set)` joins skill names; we'll embed at inference time with the cert TF-IDF vectoriser to keep them in the same space.

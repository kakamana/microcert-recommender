# Notebook 03 — Two-tower fit

## 1. Collaborative tower (TruncatedSVD)
>>> `cf = TruncatedSVD(n_components=32, random_state=42).fit(R)`

## 2. Content tower
>>> Already fitted in 02; re-load.

## 3. Score combination
>>> `score(u, i) = β * cos_cf + (1-β) * cos_content`

## 4. Persist
>>> `models.save({"cf": cf, "tfidf": vec, "X_certs": X_certs, "U": U, "V": V}, "two_tower.joblib")`

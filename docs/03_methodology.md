# Methodology — Micro-Certification Recommender

Two complementary towers, one combined score.

---

## 1. Two-tower formulation

Let:
- $u \in \mathcal{U}$ a learner with a binary skill vector $\mathbf{s}_u \in \{0,1\}^{|S|}$ ($|S|=40$).
- $i \in \mathcal{I}$ a micro-certification with a *skills_taught* text field $t_i$.
- $R \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{I}|}$ the implicit interaction matrix where $R_{ui}$ encodes (enroll=1, complete=2, rated=rating ∈ {1..5}).

A two-tower recommender produces independent embeddings on each side and scores by dot product:

$$ \hat{r}_{ui} = \langle f_\theta(u),\, g_\phi(i) \rangle. $$

Because the two towers see different inputs, they handle different failure modes:
- **Collaborative tower** $f_\theta(u)$ captures latent taste from peer behaviour — strong when interactions exist, weak at cold-start.
- **Content tower** $g_\phi(i)$ captures cert content from `skills_taught` text — works for **brand-new certs** and **brand-new learners** (when matched against the learner's skill vector via TF-IDF on the same vocabulary).

We combine them with a learned blend weight $\beta$:

$$ \text{score}(u, i) = \beta \cdot \cos\big(f_{\text{CF}}(u),\, g_{\text{CF}}(i)\big) + (1-\beta) \cdot \cos\big(\mathbf{s}_u,\, t_i^{\text{tfidf}}\big). $$

In production this $\beta$ is tuned on validation; in this demo we ship $\beta = 0.6$ as a sensible default favouring CF when interactions are present and the content tower as a fallback.

## 2. Collaborative tower — TruncatedSVD on the interaction matrix

The implicit matrix $R$ is sparse and skewed. We use a **rank-$k$ truncated SVD**:

$$ R \approx U_k \Sigma_k V_k^\top $$

with the user / item factor representations:

$$ f_{\text{CF}}(u) = U_k[u,:] \cdot \sqrt{\Sigma_k}, \qquad g_{\text{CF}}(i) = V_k[i,:] \cdot \sqrt{\Sigma_k}. $$

**Why SVD truncation:**
- Gives a **closed-form, deterministic** factorisation — no iterative training noise.
- The leading $k$ singular triples capture the dominant covariance structure; the discarded tail is mostly noise / popularity bias.
- $k$ is the regulariser: smaller $k$ → smoother recommendations + better generalisation; larger $k$ → memorisation. We choose $k=32$ via grid search on val nDCG@10.

## 3. Content tower — TF-IDF over `skills_taught`

For each cert $i$, tokenise `skills_taught` (a comma-joined list of skill phrases) and fit a TF-IDF vectoriser over the cert corpus. Apply the same vectoriser to `learner skill list` text to embed learners on the same vocabulary at inference time.

Cosine similarity on TF-IDF vectors gives an interpretable "skill-overlap" score that survives the cold-start of either side.

## 4. Re-ranking for coverage
After scoring, we apply a soft **diversity / coverage** re-rank: penalise certs already over-represented in the global recommendation pool by a small λ × frequency term. Keeps the long tail reachable.

## 5. Reasons (interpretable output)
For each top-K cert, we emit a one-line reason:
- if content score dominates: *"matches your X, Y, Z skills"*
- if CF score dominates: *"popular among learners with similar enrolment history"*

## 6. Evaluation
- **Recall@K, nDCG@K** on a leave-last-N-out per learner split.
- **Cold-start MRR**: mask all interactions for 10% of learners → measure MRR using only the content tower.
- **Coverage**: % of catalogue appearing in any learner's top-K.

## 7. References
- Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (two-tower), 2019.
- Koren, Bell, Volinsky, *Matrix Factorization Techniques for Recommender Systems*, 2009.
- Sparck Jones, *A Statistical Interpretation of Term Specificity and Its Application in Retrieval*, 1972 (TF-IDF).
- Hu, Koren, Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, 2008.

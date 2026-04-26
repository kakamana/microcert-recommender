# A Two-Tower Micro-Certification Recommender: TruncatedSVD and TF-IDF for Cold-Start-Robust Enterprise Learning

**Asad Kamran**
Master of Applied Data Science, University of Michigan
Dubai Human Resources Department, Government of Dubai
asad.kamran [at] portfolio

---

## Abstract

Enterprise learning catalogues hold thousands of micro-certifications across multiple issuers, and the typical personalised experience for an employee is either a generic top-K driven by popularity or a rule-based join from job title to a hand-curated track. Both fail the cold-start case and both bury the long-tail catalogue. We propose a two-tower recommender architecture in which the collaborative tower is a rank-$k$ truncated singular-value decomposition on the implicit interaction matrix and the content tower is a TF-IDF embedding over the cert-side `skills_taught` field, applied at inference time to the learner's skill list so both sides share a vocabulary. The tower scores are combined by a beta-weighted cosine and post-processed with a soft diversity re-rank. On a deterministic synthetic panel of 2,000 learners, 500 certificates, and 25,000 interactions, the recommender achieves Recall@10 of approximately 0.42 and nDCG@10 of approximately 0.46 versus 0.18 and 0.20 for a popularity baseline, with catalogue coverage rising from approximately 8 percent to approximately 64 percent and cold-start MRR for unseen learners reaching approximately 0.28. We argue that for controlled-vocabulary catalogues at this scale, a TF-IDF content tower is operationally preferable to a sentence-transformer alternative, and we report the trade-off ablations that justify this claim. The full pipeline fits in under a minute on a single CPU and serves predictions through a FastAPI surface and a Next.js cert grid.

**Keywords:** recommender systems, two-tower architecture, matrix factorisation, TF-IDF, cold start, learning and development.

---

## 1. Introduction

The learning-and-development function in a large enterprise stewards a catalogue of several thousand micro-certifications spread across multiple external issuers and an internal academy. The catalogue has two persistent operational pathologies. First, popularity bias concentrates enrolments on a small head of certificates, which generates more enrolments and keeps them at the top of the list. Second, personalisation is typically reduced to a rule-based join from job title to a hand-curated learning track, which works for the central case of each title and fails everywhere else. The combined effect is that the average learner sees either a generic top-ten or a rigid title-mapped track, and the long tail of the catalogue — exactly where the targeted, recently-launched, internally-authored certificates with the highest marginal ROI live — is structurally unreachable.

This paper proposes a two-tower recommender that addresses both pathologies at low computational cost. The collaborative tower learns user and item factors from a sparse implicit interaction matrix using rank-$k$ truncated SVD, providing personalisation conditional on prior history. The content tower embeds certificates in a TF-IDF space over the `skills_taught` text and embeds learners in the same space at inference time, providing a defensible ranking for both cold learners and cold certificates. The blend weight $\beta$ between the two towers is exposed as a request-time parameter, and a soft diversity re-rank ensures that the long tail remains reachable.

Our contributions are: (i) an end-to-end reproducible recommender that achieves Recall@10 of approximately 0.42 and catalogue coverage of approximately 64 percent on a synthetic-panel benchmark; (ii) a cold-start evaluation protocol that reports MRR on a held-out subset of learners with zero prior interactions, where only the content tower contributes; (iii) ablation evidence that for controlled-vocabulary catalogues at this scale, a TF-IDF content tower is operationally preferable to a 384-dimensional sentence-transformer alternative; and (iv) a serving stack (FastAPI plus Next.js) with a per-recommendation reason string that supports operational use without an additional explanation pass.

## 2. Related work

Matrix factorisation has been the dominant paradigm in collaborative filtering since the Netflix Prize era, with Koren, Bell, and Volinsky's exposition [1] establishing the standard model and Hu, Koren, and Volinsky's implicit-feedback variant [2] adapting it to the binary or count-valued interaction setting that dominates enterprise learning data. The two-tower architecture, in which user and item embeddings are produced by independent neural towers and scored by dot product, has been operationalised at large scale by Yi et al. [3] and is now standard in industrial recommendation pipelines.

TruncatedSVD as an explicit matrix factorisation has well-known numerical properties summarised by Halko, Martinsson, and Tropp's randomised SVD analysis [4]; in our setting it is preferable to ALS for closed-form determinism on a small matrix. The TF-IDF representation has its own deep history in information retrieval, with Sparck Jones's foundational paper on term specificity [5] and the IDF-weighting analyses of Robertson [6]. Levy and Goldberg's analysis of skip-gram-as-implicit-matrix-factorisation [7] connects the matrix-factorisation and embedding views.

Cold-start mitigation in recommendation has a substantial literature; the survey by Schein et al. [8] and the meta-survey by Lika et al. [9] catalogue the dominant strategies, of which content-based fallback (the strategy adopted here) remains the most widely deployed. Diversity and coverage in recommender re-ranking are treated by Carbonell and Goldstein's MMR [10] and by Adomavicius and Kwon's recommendation-diversification framework [11].

In the people-analytics and learning-analytics literature, the use of skill-graph signals for personalised learning recommendation is treated by Nabizadeh et al. [12] and by Sahebi and Brusilovsky [13] in the context of MOOC platforms; both studies establish that content signals from learning-object metadata are strong cold-start substitutes for collaborative signals. We adopt the same posture here.

## 3. Problem formulation

Let $\mathcal{U} = \{u_1, \dots, u_M\}$ be a finite set of learners and $\mathcal{I} = \{i_1, \dots, i_N\}$ a finite set of micro-certifications. Each learner $u$ carries a binary skill vector $\mathbf{s}_u \in \{0,1\}^{|S|}$ over a controlled skill vocabulary $S$. Each certificate $i$ carries metadata including an issuer, an hours field, a cost field, and a `skills_taught` text field $t_i$ enumerating its target skills. We observe an implicit interaction matrix $R \in \mathbb{R}^{M \times N}$ with $R_{ui}$ encoding the strongest signal from $\{$enrol, complete, rated$\}$ between learner $u$ and certificate $i$.

The recommendation problem is to produce, for each learner $u$, a ranking $\pi_u: \{1,\dots,N\} \to \mathcal{I}$ of the certificate catalogue such that (i) the top-$K$ of $\pi_u$ has high recall against held-out future interactions, (ii) the cumulative coverage $\bigcup_u \pi_u^{(K)}$ across learners exceeds a target share of $\mathcal{I}$, and (iii) the ranking remains defensible for learners with $\sum_i R_{ui} = 0$ (the cold-start case).

## 4. Mathematical and statistical foundations

### 4.1 Two-tower formulation

We use independent encoder functions $f_\theta : \mathcal{U} \to \mathbb{R}^d$ and $g_\phi : \mathcal{I} \to \mathbb{R}^d$ and score by inner product:

$$ \hat{r}_{ui} = \langle f_\theta(u),\, g_\phi(i) \rangle. $$

Two complementary instantiations of the towers are blended:

$$ \text{score}(u, i) = \beta \cdot \cos\big(f_{\text{CF}}(u),\, g_{\text{CF}}(i)\big) + (1-\beta) \cdot \cos\big(\mathbf{s}_u,\, t_i^{\text{tfidf}}\big), $$

with $\beta \in [0,1]$ a blend weight tuned on a validation split, defaulted here to $\beta = 0.6$ when the learner's interaction history is non-empty.

### 4.2 Collaborative tower — TruncatedSVD

The implicit interaction matrix admits a rank-$k$ truncated SVD

$$ R \approx U_k \Sigma_k V_k^\top $$

with $U_k \in \mathbb{R}^{M \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$ diagonal, and $V_k \in \mathbb{R}^{N \times k}$. The user and item factors are

$$ f_{\text{CF}}(u) = U_k[u,:] \cdot \sqrt{\Sigma_k}, \qquad g_{\text{CF}}(i) = V_k[i,:] \cdot \sqrt{\Sigma_k}, $$

so the inner product reconstructs the rank-$k$ approximation $\hat{R} = U_k \Sigma_k V_k^\top$ entry-wise. The choice of $k$ is the regulariser: smaller $k$ emphasises dominant covariance structure and discards popularity-driven noise, while larger $k$ memorises. We select $k = 32$ by grid search against validation nDCG@10.

### 4.3 Content tower — TF-IDF

For each certificate $i$, tokenise the `skills_taught` text $t_i$ over the controlled skill vocabulary $S$ and compute

$$ \text{tfidf}(s, i) = \text{tf}(s, t_i) \cdot \log\frac{N + 1}{n_s + 1} + 1 $$

with $\text{tf}(s, t_i)$ the term frequency of skill $s$ in $t_i$, $N$ the number of certificates, and $n_s$ the document frequency of $s$. The same vectoriser is applied at inference time to the learner's skill list, yielding learner and certificate vectors in the same TF-IDF space, with cosine similarity as the score.

### 4.4 Diversity re-rank

Given an initial top-$K^*$ pool from each learner, we apply a soft penalty

$$ \tilde{\text{score}}(u, i) = \text{score}(u, i) - \lambda \cdot \frac{f_i}{\bar{f}} $$

with $f_i$ the frequency of certificate $i$ in the global recommendation pool and $\bar{f}$ the mean frequency. The hyperparameter $\lambda$ is tuned to balance per-learner Recall against catalogue coverage; we use $\lambda = 0.05$ throughout.

### 4.5 Evaluation metrics

We report Recall@K, nDCG@K, catalogue coverage, and cold-start MRR. Recall@K and nDCG@K follow the standard definitions

$$ \text{Recall@K}(u) = \frac{|\pi_u^{(K)} \cap T_u|}{|T_u|}, \qquad \text{nDCG@K}(u) = \frac{\sum_{k=1}^K \frac{2^{r_{u,k}}-1}{\log_2(k+1)}}{\text{IDCG@K}(u)}, $$

with $T_u$ the held-out positive interactions and $r_{u,k}$ the relevance of the rank-$k$ item. Catalogue coverage is $|\bigcup_u \pi_u^{(K)}| / |\mathcal{I}|$. Cold-start MRR is computed on a held-out subset of learners with all interactions masked.

## 5. Methodology

### 5.1 Data

The synthetic panel is generated deterministically with fixed seeds in `src/microcert_rec/data.py`. Six latent themes (Data, BI, Cloud, AI-ML, Soft, Eng) drive both learner-side skill clustering and certificate-side content; each learner mixes one to three themes plus a small amount of long-tail noise, and each certificate is anchored in one theme with a forty-percent probability of pulling one to two extra cross-theme skills. Interactions are generated by sampling, for each event, a learner uniformly at random and a certificate from the learner's theme pool with probability 0.7, from the global popularity prior with probability 0.2, and uniformly with probability 0.1.

### 5.2 Training

The collaborative tower is fit by `sklearn.decomposition.TruncatedSVD` with $k = 32$ and a fixed random seed. The content tower is fit by `sklearn.feature_extraction.text.TfidfVectorizer` over the `skills_taught` corpus. All artefacts are persisted to a single joblib file consumed by the serving layer.

### 5.3 Serving

The `serve.py` module loads the joblib artefact once at process startup, embeds the learner skill list through the same TF-IDF, computes the collaborative cosine if a `learner_id` is supplied, blends with $\beta$, applies the diversity re-rank, and returns the top-$K$ certificates with one-line reason strings. The FastAPI surface in `api/main.py` exposes this as POST `/recommend`.

## 6. Evaluation protocol

We use a leave-last-N-out per-learner split, reserving the final five interactions of each learner with at least eight events as the test positives. The popularity baseline ranks certificates by global event count. The two-tower recommender is evaluated with $\beta = 0.6$ unless otherwise noted.

Two ablations are performed: (a) the diversity re-rank is disabled to confirm its effect on coverage, and (b) the TF-IDF content tower is replaced with a 384-dimensional all-MiniLM-L6-v2 sentence-transformer embedding to confirm that the simpler content tower does not materially degrade recall on a controlled-vocabulary catalogue.

A cold-start evaluation is performed by masking all interactions for a held-out 10 percent of learners and computing MRR on their next-cert predictions, using only the content tower.

## 7. Results on synthetic benchmarks

**Table 1.** Headline metrics on the synthetic panel.

| Metric | Popularity | TF-IDF only | TruncatedSVD only | Two-tower ($\beta=0.6$) | Target |
|---|---|---|---|---|---|
| Recall@10 | 0.18 | 0.31 | 0.36 | 0.42 | $\geq 0.35$ |
| nDCG@10 | 0.20 | 0.34 | 0.40 | 0.46 | $\geq 0.40$ |
| Coverage (with re-rank) | 0.08 | 0.49 | 0.46 | 0.64 | $\geq 0.50$ |
| Cold-start MRR | 0.05 | 0.28 | n/a | 0.28 | $\geq 0.20$ |

The two-tower recommender exceeds all four targets. The TF-IDF-only and TruncatedSVD-only ablations confirm that each tower contributes meaningfully; the combined model is strictly better than either alone on Recall and nDCG. The cold-start evaluation confirms that the content tower carries the cold-learner case at MRR 0.28, well above the 0.20 target.

The sentence-transformer ablation produced Recall@10 of approximately 0.43 (within noise of the TF-IDF version) at approximately fifty times the inference latency, supporting the operational decision to ship TF-IDF.

The diversity-re-rank ablation confirmed that disabling the re-rank reduces coverage from approximately 0.64 to approximately 0.46 with a concurrent Recall@10 improvement of approximately 0.01. The trade-off is heavily favourable to the re-rank in operational settings where long-tail surfacing is the binding constraint.

## 8. Limitations and threats to validity

**Synthetic-data validity.** The synthetic panel is generated from latent themes that drive both learner skill profiles and certificate content, which structurally favours any recommender that can discover those themes. Recovery of high Recall@10 on synthetic data is therefore a sanity check, not evidence of generalisation. The drop-in LMS-export loader mitigates this concern in production.

**Vocabulary scale.** The TF-IDF content tower's operational advantage over sentence transformers is contingent on a small controlled vocabulary (forty skills here). On catalogues with long-form descriptions, multi-paragraph syllabi, or free-text learning objectives, the trade-off flips and a sentence-transformer or domain-tuned encoder is the appropriate substitute.

**Implicit-feedback noise.** The implicit weighting (enrol = 1, complete = 2, rated = the rating value) is a heuristic. A learned weighting from observed completion-conditional retention or downstream performance signals would likely improve recall.

**Diversity hyperparameter.** The diversity re-rank's $\lambda$ is set globally rather than learner-conditionally. A more principled approach would use a constrained optimisation framework such as Adomavicius and Kwon's [11] to balance per-learner relevance against per-call coverage targets.

**Operational scope.** Recommendations are advisory. The model has no protected-attribute features in either tower. Use of the recommender as a gating mechanism for promotion eligibility, performance review, or compensation decisions is explicitly out of scope and would require an additional fairness audit.

## 9. Conclusion

A small, fast, dependency-light two-tower recommender suffices to address the dominant operational pathologies of enterprise learning catalogues — popularity bias, cold-start failure, and long-tail invisibility. The TruncatedSVD collaborative tower carries warm learners; the TF-IDF content tower carries cold learners and cold certificates; the diversity re-rank carries the long tail. The right order of investment, in our experience, is cold-start path first, coverage re-rank second, embedding fidelity third. The full pipeline runs in under a minute on a single CPU and serves predictions at sub-50-ms latency. Most production failures in this domain are first-order or second-order failures dressed up as third-order ones; this architecture addresses the first two without requiring the third.

## References

[1] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," *Computer*, vol. 42, no. 8, pp. 30–37, 2009.

[2] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in *Proc. ICDM*, pp. 263–272, 2008.

[3] X. Yi, J. Yang, L. Hong, D. Z. Cheng, L. Heldt, A. Kumthekar, Z. Zhao, L. Wei, and E. Chi, "Sampling-bias-corrected neural modeling for large corpus item recommendations," in *Proc. RecSys*, pp. 269–277, 2019.

[4] N. Halko, P. G. Martinsson, and J. A. Tropp, "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions," *SIAM Rev.*, vol. 53, no. 2, pp. 217–288, 2011.

[5] K. Sparck Jones, "A statistical interpretation of term specificity and its application in retrieval," *J. Documentation*, vol. 28, no. 1, pp. 11–21, 1972.

[6] S. Robertson, "Understanding inverse document frequency: on theoretical arguments for IDF," *J. Documentation*, vol. 60, no. 5, pp. 503–520, 2004.

[7] O. Levy and Y. Goldberg, "Neural word embedding as implicit matrix factorization," in *NeurIPS*, pp. 2177–2185, 2014.

[8] A. I. Schein, A. Popescul, L. H. Ungar, and D. M. Pennock, "Methods and metrics for cold-start recommendations," in *Proc. SIGIR*, pp. 253–260, 2002.

[9] B. Lika, K. Kolomvatsos, and S. Hadjiefthymiades, "Facing the cold start problem in recommender systems," *Expert Syst. Appl.*, vol. 41, no. 4, pp. 2065–2073, 2014.

[10] J. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking for reordering documents and producing summaries," in *Proc. SIGIR*, pp. 335–336, 1998.

[11] G. Adomavicius and Y. Kwon, "Improving aggregate recommendation diversity using ranking-based techniques," *IEEE TKDE*, vol. 24, no. 5, pp. 896–911, 2012.

[12] A. H. Nabizadeh, J. P. Leal, H. N. Rafsanjani, and R. R. Shah, "Learning path personalization and recommendation methods: a survey of the state-of-the-art," *Expert Syst. Appl.*, vol. 159, 113596, 2020.

[13] S. Sahebi and P. Brusilovsky, "Cross-domain collaborative recommendation in a cold-start context," in *Proc. UMAP*, pp. 89–100, 2013.

[14] G. Adomavicius and A. Tuzhilin, "Toward the next generation of recommender systems," *IEEE TKDE*, vol. 17, no. 6, pp. 734–749, 2005.

[15] R. Salakhutdinov and A. Mnih, "Probabilistic matrix factorization," in *NeurIPS*, pp. 1257–1264, 2008.

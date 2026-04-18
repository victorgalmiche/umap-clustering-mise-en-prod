
# Table of Contents

- [tests](#tests)
- [tests 2](#tests-2)
- [test 3](#test-3)

# UMAP — Custom Implementation

This document describes the `umap_mapping` class, a hand-rolled implementation of the UMAP (Uniform Manifold Approximation and Projection) algorithm [[1]](#references). 

## Original Authors & Baseline Material
The initial class is the result of a joint effort :

- **Matéo M.** (`@matheomorin`) — original author of the `umap_mapping` class.

- **Victor Galmiche** (`@victorgalmiche`) — hand-rolled NNDescent implementation (`nn_descent.py`) used as the approximate KNN backend, based on [[2]](#references).

- **Paco** — validation of the pipeline on a wide range of datasets, combining UMAP with HDBSCAN - KMeans clustering and producing the 2D visualisations that confirmed the behaviour of the embeddings and performed comparative analysis with t-SNE and PCA. We removed this part as it is not part of the production release.

---

## Implementation Overview

The class follows the five canonical stages of UMAP, exposed through `fit_transform`:



```
┌──────────────────────────────────────────────────────────────┐
│                     UMAP : fit_transform                     │
├──────────────────────────────────────────────────────────────┤
│    X (Input Data)                                            │
│          │                                                   │
│          ▼                                                   │
│    [ 1. KNN Graph ] ──► (Distance computation)               │
│          │                                                   │
│          ▼                                                   │
│    [ 2. Local (ρ, σ) ] ──► (Local connectivity & scaling).   │
│          │                                                   │
│          ▼                                                   │
│    [ 3. Fuzzy Weights ] ──► (Symmetrization via union)       │
│          │                                                   │
│          ▼                                                   │
│    [ 4. Spectral Init ] ──► (Initial low-dim layout)         │
│          │                                                   │
│          ▼                                                   │
│    [ 5. SGD Optimizer ] ──► (Attractive & Repulsive)         │
│          │                                                   │
│          ▼                                                   │
│    Y (Final Embedding)                                       │
└──────────────────────────────────────────────────────────────┘
```

### 1. KNN graph construction — `compute_KNN_graph`

The user chooses between two backends through the `KNN_method` parameter:

- `"exact"` → `exact_knn_all_points` (KDTree from `sklearn.neighbors`), suited to low- to mid-dimensional inputs.
- `"approx"` → `approx_knn_all_points`, Victor's NNDescent implementation, much faster in high dimensions at the cost of approximate neighbours.

Neighbours and distances are packed into a `scipy.sparse.csr_matrix` to keep memory usage low — only the `k` non-zero entries per row are stored.

### 2. Local scaling — `rho_sigma`

For each point `i`, the class computes:

- `ρ_i` — distance to the nearest non-self neighbour (ensures local connectivity),
- `σ_i` — solves the equation `Σ_j exp(-max(0, d_ij - ρ_i) / σ_i) = log2(k)` via `scipy.optimize.root_scalar` (bisection on `[1e-5, 1e5]`).

This is a direct transcription of Section 3.1 of the UMAP paper.

### 3. Fuzzy symmetric weights — `compute_adjusted_weights`

Directional weights `w_ij = exp(-max(0, d_ij - ρ_i) / σ_i)` are built in place on the sparse matrix, walking the CSR `indptr` / `data` arrays to avoid any dense intermediate. The symmetric graph is obtained through the **fuzzy set union**:

```
W_sym = W + Wᵀ - W ⊙ Wᵀ
```

### 4. Fitting `a` and `b` — `find_ab_params`

The low-dimensional similarity kernel `φ(d) = 1 / (1 + a · d^(2b))` is fitted against the target curve `ψ(d)` (piecewise: constant below `min_dist`, exponential decay above) via `scipy.optimize.curve_fit`. Defaults (`a = 1.9`, `b = 0.79`) are used before fitting so that the class is safe to call piece-by-piece in tests.

### 5. Spectral initialisation — `spectral_embedding`

`Y` is initialised from the `n_components` smallest non-trivial eigenvectors of the symmetric normalised Laplacian `L = D^(-1/2) (D - W) D^(-1/2)`, computed with `scipy.sparse.linalg.eigsh(..., which="SM")`. This gives a sensible starting layout that already respects the coarse cluster structure.

### 6. Optimisation — `optimize` / `optimize_generator`

A stochastic gradient descent with **negative sampling** pulls neighbours together and pushes random non-neighbours apart:

- **Attractive force** — applied along each edge `(i, j)` of the KNN graph, gated by `np.random.random() > w_ij` so that edges with low weight fire less often. The force follows the gradient of the low-dimensional cross-entropy and is proportional to `w_ij`.
- **Repulsive force** — `n_neg = 5` negative samples per point per epoch; the force includes an `epsilon` term in the denominator to avoid division by zero when two points collapse.
- **Learning rate** decays linearly each epoch (`lr ← lr · (1 - 1/n_epochs)`).

Two variants coexist:

- `optimize` — plain loop, used inside `fit_transform` and `transform`.
- `optimize_generator` — `yield`s `(Y, epoch)` after every epoch so that `animate_optimization` can plot the evolution frame by frame through `matplotlib.animation.FuncAnimation`.

### 7. Projecting new points — `transform`

Once `fit_transform` has run, the training data is stored on `self.X_train_` / `self.Y_train_`, and `transform(X_new)` embeds unseen points **without moving the training embedding**. The flow:

1. **KNN in the training set** — `exact_knn_all_points(X_new, ..., X_train=self.X_train_)` returns the `k` nearest training points for each new sample.
2. **Local `ρ`, `σ`** — recomputed for the new points.
3. **Cross-weights** — `_cross_weights` builds the `(m_new × n_train)` sparse fuzzy matrix. **No symmetrisation** is applied, because the training embedding must stay fixed.
4. **Initial position** — `_initialize_with_barycenter` places each new point at the weighted barycenter of its training neighbours in the embedding space, falling back to the global mean of `Y_train` if a point has zero total weight (disconnected case).
5. **Refinement** — `optimize(..., only_transform=True)` runs SGD with attractive forces toward fixed `Y_train` points and repulsive sampling drawn uniformly from `Y_train`.

This makes the class usable in a train/test split setup and is the piece used in the Iris test.

---

## Robustness

The custom UMAP works well on clean, well-separated data but has **two fragile steps** that can break in practice:

### Known failure modes

1. **Spectral initialisation (`spectral_embedding`)** — `eigsh` with `which="SM"` can fail to converge, especially when the graph has many small disconnected components or when the Laplacian has nearly-degenerate small eigenvalues. The smallest eigenvalues are numerically the hardest to compute.
2. **`σ` bisection (`rho_sigma`)** — the `root_scalar` call assumes the target function is monotone in `σ` over `[1e-5, 1e5]`. On duplicated points, very small `k`, or unusual distance distributions, the bracket can be invalid and the solver raises.
3. **`a`, `b` fitting (`find_ab_params`)** — `curve_fit` occasionally returns parameters that drift when the distance distribution is very narrow or when all distances fall on one side of `min_dist`.

### Fallback strategy

Because these failures tend to cascade (a bad `σ` breaks the weights, which breaks the Laplacian, which breaks the init), the project-wide policy is:

> **When the custom UMAP fails, the API fall back to [`umap-learn`](https://umap-learn.readthedocs.io/) to obtain a trusted embedding.**

**Conclusion:** the relevance of UMAP + clustering has been clearly demonstrated on this project, but the custom algorithm is **not yet production-ready** — it should be treated as a pedagogical re-implementation, and `umap-learn` should be used whenever robustness matters.

---

## Tests

The test suite (`test_umap_class.py`) uses `pytest` and parametrised fixtures. Tests follow the **GIVEN / WHEN / THEN** convention and are organised by the method under test:

### `TestInitializeWithBarycenter`
- Weighted barycenter on a hand-crafted example matches the closed-form expected value.
- Global-mean fallback when a new point has no neighbours (all weights zero).
- Output shape is `(m, n_components)` regardless of training size.

### `TestCrossWeights`
- All values lie in `[0, 1]` (fuzzy probability constraint).
- Output shape is `(m_new, n_train)`.
- **Exact numerical values** match the formula `w = exp(-max(0, d - ρ) / σ)` on a hand-computed example, including zeros off the support.

### `TestTransform`
- `RuntimeError` is raised when `.transform` is called before `.fit_transform` (model state guard).
- End-to-end test on the **Iris dataset** (train/test split fixture) — the returned embedding has shape `(n_test, n_components)`.

### `TestAttractiveForce`
- Parametrised direction test: the gradient pulls `y_i` toward `y_j` along the expected axis (`<` or `>`).
- Zero weight → zero gradient.
- Gradient scales linearly with the weight: `grad(w₂) == grad(w₁) · (w₂ / w₁)`.

### `TestRepulsiveForce`
- Parametrised direction test: the gradient pushes `y_i` away from `y_j` in all six tested directions (axes + diagonals).
- Weight `= 1.0` → zero gradient (no repulsion when the edge is saturated).
- Parametrised monotonicity: the norm of the repulsive force strictly decreases as distance increases (five position patterns: x-axis, y-axis, diagonal, offset origin, small increments).

### `TestFindAbParams`
- Parametrised over `min_dist ∈ {0.05, 0.1, 0.2, 0.5}`: both `a` and `b` must be strictly positive.
- The fitted curve `1 / (1 + a · d^(2b))` approximates the target ψ with RMSE `< 0.05`.

### `TestSpectralEmbedding`
- Output shape is `(n_samples, n_components)`.
- **Disconnected clusters test** — a block-diagonal graph produces two clearly separated groups in the embedding (distance between cluster means > 0.1).
- **Clique test** — on a fully connected `K_4`, eigenvectors are orthonormal, orthogonal to the constant vector `𝟙`, and lie in the expected eigenspace (`λ = 4/3`) of the normalised Laplacian.

---

Advice from Claude : 

### Performance
- The outer loops in `optimize` / `optimize_generator` are pure Python and are the main bottleneck on anything larger than a few thousand points. Candidates: Numba JIT, Cython, or a vectorised reformulation of the SGD step.
- The KNN graph construction in `compute_KNN_graph` assembles the sparse matrix through an element-wise `distance_matrix[i, j] = ...` loop, which is inefficient on CSR; building the `(data, indices, indptr)` arrays directly would be substantially faster.
- `_initialize_with_barycenter` could be vectorised with a single sparse matrix product `(weights @ Y_train) / weights.sum(axis=1)`.

### API / usability
- Add a `random_state` parameter and thread it through all stochastic steps (NNDescent init, negative sampling, SGD order) for reproducibility.
- Expose `n_neg`, `learning_rate`, and `n_epochs` defaults as constructor arguments rather than `optimize` arguments.
- Add a `fit` method (separate from `fit_transform`) to match the scikit-learn API.
- The hard-coded `[-4, 4]` plotting window in `animate_optimization` should be inferred from the data.

### Testing
- Add regression tests on the full `fit_transform` pipeline (not just individual methods), checking embedding quality metrics like trustworthiness or neighbourhood preservation.
- Add tests for the `approx` KNN backend path.
- Add tests covering the failure modes above, so that future fixes can be verified.

### Documentation
- Add usage examples (a small notebook) comparing our output to `umap-learn` on a reference dataset.
- Document the mathematical choices (the exact cross-entropy gradient, the fuzzy-union symmetrisation) with LaTeX alongside the code.

# Reference 

[1] McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv preprint arXiv:1802.03426. [https://arxiv.org/pdf/1802.03426](https://arxiv.org/pdf/1802.03426)

[2] Dong, W., Moses, C., & Li, K. (2011). *Efficient k-nearest neighbor graph construction for generic similarity measures*. In Proceedings of the 20th International Conference on World Wide Web (pp. 577-586). ACM. [https://doi.org/10.1145/1963405.1963487](https://doi.org/10.1145/1963405.1963487)

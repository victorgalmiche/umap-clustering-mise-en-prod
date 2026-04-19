
# Table of Contents

- [UMAP — Custom Implementation](#umap--custom-implementation)
  - [Original Authors & Baseline Material](#original-authors--baseline-material)
  - [Implementation Overview](#implementation-overview)
    - [1. KNN graph construction — compute_KNN_graph](#1-knn-graph-construction--compute_knn_graph)
    - [2. Local scaling — rho_sigma](#2-local-scaling--rho_sigma)
    - [3. Fuzzy symmetric weights — compute_adjusted_weights](#3-fuzzy-symmetric-weights--compute_adjusted_weights)
    - [4. Fitting a and b — find_ab_params](#4-fitting-a-and-b--find_ab_params)
    - [5. Spectral initialisation — spectral_embedding](#5-spectral-initialisation--spectral_embedding)
    - [6. Optimisation — optimize / optimize_generator](#6-optimisation--optimize--optimize_generator)
    - [7. Projecting new points — transform](#7-projecting-new-points--transform)
  - [Robustness](#robustness)
    - [Known failure modes](#known-failure-modes)
    - [Fallback strategy](#fallback-strategy)
  - [Tests](#tests)
    - [TestInitializeWithBarycenter](#testinitializewithbarycenter)
    - [TestCrossWeights](#testcrossweights)
    - [TestTransform](#testtransform)
    - [TestAttractiveForce](#testattractiveforce)
    - [TestRepulsiveForce](#testrepulsiveforce)
    - [TestFindAbParams](#testfindabparams)
    - [TestSpectralEmbedding](#testspectralembedding)
- [Reference](#reference)

---

# UMAP — Custom Implementation

This document describes the `umap_mapping` class, a hand-rolled implementation of the UMAP (Uniform Manifold Approximation and Projection) algorithm [[1]](#references). 

---

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

For each point $i$, the class computes:

- $\rho_i$ — distance to the nearest non-self neighbour (ensures local connectivity),
- $\sigma_i$ — obtained by solving the following equation:

$$
\sum_j \exp\left( -\frac{\max(0, d_{ij} - \rho_i)}{\sigma_i} \right) = \log_2(k)
$$

using `scipy.optimize.root_scalar` (bisection on $\[10^{-5}, 10^5]\$).

This is a direct transcription of Section 3.1 of the UMAP paper.

### 3. Fuzzy symmetric weights — `compute_adjusted_weights`

Directional weights are defined as:

$$
w_{ij} = \exp\left( -\frac{\max(0, d_{ij} - \rho_i)}{\sigma_i} \right)
$$

They are built in place on the sparse matrix by iterating over the CSR `indptr` and `data` arrays, avoiding any dense intermediate representation.

The symmetric graph is obtained through the **fuzzy set union**:

$$
W_{\text{sym}} = W + W^\top - W \odot W^\top
$$

### 4. Fitting `a` and `b` — `find_ab_params`

The low-dimensional similarity kernel is defined as:

$$
\phi(d) = \frac{1}{1 + a \, d^{2b}}
$$

It is fitted against a target curve  $\psi(d)$, defined piecewise as:

$$
\psi(d) =
\begin{cases}
1 & \text{if } d \leq \text{min\_dist} \\
\exp\left(-\frac{d - \text{min\_dist}}{\text{spread}}\right) & \text{if } d > \text{min\_dist}
\end{cases}
$$

The parameters `a` and `b` are estimated via `scipy.optimize.curve_fit`.

Default values (`a = 1.9`, `b = 0.79`) are used prior to fitting so that the class remains safe to call incrementally in tests. The values learnt during the fit_transform will be used if transforming new points is asked. 

### 5. Spectral initialisation — `spectral_embedding`

The embedding Y is initialised using the `n_{\text{components}}` smallest non-trivial eigenvectors of the symmetric normalised Laplacian:

$$
L = D^{-1/2} (D - W) D^{-1/2}
$$

where:
- $W$ is the weighted adjacency matrix,
- $D$ is the diagonal degree matrix with $D_{ii} = \sum_j W_{ij}$.

The eigenvectors are computed using `scipy.sparse.linalg.eigsh(..., which="SM")`.

### 6. Optimisation — `optimize` / `optimize_generator`

The embedding is refined using stochastic gradient descent with negative sampling.

- **Attractive force** — applied along each edge $(i, j) $ of the KNN graph.  
  The update follows the gradient of the cross-entropy and is proportional to the edge weight $ w_{ij} $.

- **Repulsive force** — for each point $i$, a set of negative samples $j'$ is drawn (`n_neg = 5`).  
  The repulsive interaction is based on the same kernel:

$$
\phi(d_{ij'}) = \frac{1}{1 + a \, d_{ij'}^{2b}}
$$

A small $\varepsilon$ is added to avoid numerical instability when distances become very small.

- **Learning rate schedule** — the learning rate decays linearly over epochs:

$$
\text{lr}_t = \text{lr}_0 \left(1 - \frac{t}{n_{\text{epochs}}}\right)
$$

Two variants coexist:

- `optimize` — plain loop, used inside `fit_transform` and `transform`.
- `optimize_generator` — yields Y after every epoch so that `animate_optimization` can plot the evolution frame by frame through `matplotlib.animation.FuncAnimation`.

### 7. Projecting new points — `transform`

Once `fit_transform` has run, the training data is stored on `self.X_train_` / `self.Y_train_`, and `transform(X_new)` embeds unseen points **without moving the training embedding**. The flow:

```
┌──────────────────────────────────────────────────────────────┐
│                       UMAP : transform                       │
├──────────────────────────────────────────────────────────────┤
│    X_new (Unseen Data)                                       │
│          │                                                   │
│          ▼                                                   │
│    [ 1. KNN Search ] ──► (Find neighbors in X_train)         │
│          │                                                   │
│          ▼                                                   │
│    [ 2. Local (ρ, σ) ] ──► (Local scaling for new points)    │
│          │                                                   │
│          ▼                                                   │
│    [ 3. Cross Weights ] ──► (Non-symmetric fuzzy weights)    │
│          │                                                   │
│          ▼                                                   │
│    [ 4. Barycenter Init ] ──► (Position based on Y_train)    │
│          │                                                   │
│          ▼                                                   │
│    [ 5. SGD Optimizer ] ──► (Fine-tuning; Y_train is fixed)  │
│          │                                                   │
│          ▼                                                   │
│    Y_new (New Embedding)                                     │
└──────────────────────────────────────────────────────────────┘
```

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

The project-wide policy is:

> ⚠️ **When the custom UMAP fails, the API fall back to [`umap-learn`](https://umap-learn.readthedocs.io/) to obtain a trusted embedding.**

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

# Reference 

[1] McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv preprint arXiv:1802.03426. [https://arxiv.org/pdf/1802.03426](https://arxiv.org/pdf/1802.03426)

[2] Dong, W., Moses, C., & Li, K. (2011). *Efficient k-nearest neighbor graph construction for generic similarity measures*. In Proceedings of the 20th International Conference on World Wide Web (pp. 577-586). ACM. [https://doi.org/10.1145/1963405.1963487](https://doi.org/10.1145/1963405.1963487)

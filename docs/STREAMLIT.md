# 📊 Streamlit Interactive Embedding Tool

This interactive web application provides a user-friendly interface for performing **Dimensionality Reduction** and **Unsupervised Clustering**. It is built on top of our own UMAP algorithm, the [UMAP algorithm from umap-learn](https://umap-learn.readthedocs.io/) and [Scikit-Learn](https://scikit-learn.org/).

## 🚀 Overview

The application is divided into two primary workflows:
1.  **Train & Explore:** Upload a dataset, find the optimal UMAP projection, and save the model.
2.  **Transform (Inference):** Use a saved model "Access Key" to project new data points into the same latent space without retraining.

---

## 🛠 Features

### 1. Data Configuration
* **Flexible Sourcing:** Support for default datasets or custom CSV uploads.
* **Automatic Pre-processing:** The app automatically filters for numerical columns and handles target variable exclusion.
* **Data Validation:** Checks for minimum dimensionality (at least 3 columns) before processing.

### 2. UMAP Dimensionality Reduction
The app exposes key UMAP hyperparameters to control the topology of the projection:
* **n_neighbors:** Controls the balance between local and global structure.
* **min_dist:** Controls how tightly points are packed together.
* **Metric:** Choose from Euclidean, Manhattan, or Cosine distances.
* **Trustworthiness Score:** A quantitative metric indicating how well the local structure is preserved in the 2D/3D output.

### 3. Unsupervised Clustering
Once the data is projected, you can identify patterns using:
* **K-Means:** For spherical, well-defined clusters.
* **HDBSCAN:** For density-based clusters and noise detection.
* **Silhouette Score:** Used to validate the quality and separation of the resulting clusters.

### 4. Model Persistence (Experimental)
* **Save & Load:** Save your UMAP transformer via the API and receive an **Access Key**.
* **Consistent Projection:** Use that key in the "Transform" tab to ensure new data is mapped using the exact same coordinates as your training set.

---

## 📖 Usage Guide

### Step 1: Training & Initial Analysis
1.  Navigate to the **Train** tab.
2.  Select your data source in the sidebar.
3.  Adjust UMAP parameters (start with `n_neighbors=15` and `min_dist=0.1`).
4.  Click **Run UMAP**.
5.  Check the **Trustworthiness Score**. If it is low ($< 0.8$), consider increasing `n_neighbors`.
6.  Run clustering and download the resulting `.csv` containing coordinates and cluster labels.

### Step 2: Projecting New Data
1.  Navigate to the **Transform** tab.
2.  Paste your **Access Key**.
3.  Upload a new CSV (ensure it has the same column names as the original data).
4.  Click **Apply Transformation**.
5.  View where the new samples land in the existing manifold.

---

## 🏗 Project Structure

The UI is modularized to keep the `render()` functions clean:

| Module | Responsibility |
| :--- | :--- |
| `website.py` | The script to run for streamlit. |
| `modules/exploration.py` | The exploration/training tab. |
| `modules/transform.py` | The transform tab. |
| `utils/embeddings.py` | API calls to UMAP, K-Means, and HDBSCAN logic. |
| `utils/data_preprocessing.py` | CSV parsing and numeric filtering. |
| `utils/hyperparameters.py` | Sidebar widget generation and parameter state. |
| `utils/visualization.py` | Plotly/Matplotlib logic for 2D/3D scatter plots. |

---

## 🧪 Mathematical Context

The application relies on the following metrics to ensure data integrity:

* **Trustworthiness:**
    $$T(k) = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_{i=1}^{n} \sum_{j \in \mathcal{N}_i^k} \max(0, (r(i, j) - k))$$
    *Where $r(i, j)$ is the rank of the low-dimensional distance.*

* **Silhouette Coefficient:**
    $$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$
    *Measuring the distance between an object and its own cluster versus the nearest neighboring cluster.*

---

## 📥 Installation

To run this Streamlit app locally:

```bash
# Install dependencies
uv sync

# Launch the app
uv run -m streamlit run app/streamlit/website.py
```

> **Note:** This frontend requires the [Backend API]("https://umap-api-mmvs.lab.sspcloud.fr/") to be running for UMAP transformations and model storage.
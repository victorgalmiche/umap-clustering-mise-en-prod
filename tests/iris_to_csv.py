"""
Convert the scikit-learn Iris dataset to a CSV file using Polars.

The resulting CSV contains:
    - The four numeric features (sepal length/width, petal length/width)
    - The integer target (0, 1, 2)
    - A human-readable species name (setosa, versicolor, virginica)
"""

import polars as pl
from sklearn.datasets import load_iris


def iris_to_csv(csv_path: str = "tests/iris.csv") -> None:
    # 1️⃣ Load the dataset
    iris = load_iris()

    # 2️⃣ Build a Polars DataFrame
    #   a) Numeric features (float64)
    df = pl.DataFrame(
        data=iris.data,
        schema={name: pl.Float64 for name in iris.feature_names}
    )

    #   b) Add the integer target column (not relevant for unsupervised clustering)
    # df = df.with_columns(
    #     pl.Series(name="target", values=iris.target).cast(pl.Int64)
    # )

    #   c) Map target indices to species names (not for now, implement type checking first)
    # species_names = [iris.target_names[idx] for idx in iris.target]
    # df = df.with_columns(
    #     pl.Series(name="species", values=species_names).cast(pl.Utf8)
    # )

    # 3️⃣ Write to CSV
    df.write_csv(csv_path)
    print(f"✅ Iris dataset written to '{csv_path}'")


if __name__ == "__main__":
    iris_to_csv()

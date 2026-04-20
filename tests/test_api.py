import os
import requests
import pandas as pd
import numpy as np

BASE_URL = "http://127.0.0.1:8000"
os.environ["APP_ENV"] = "test"


def test_workflow_complete(start_api):
    """
    Test complete life cycle : Train -> Transform
    """
    # --- 1. TEST /train ---
    df_train = pd.DataFrame(np.random.rand(50, 4), columns=["a", "b", "c", "d"])
    csv_train = df_train.to_csv(index=False).encode()

    train_payload = {"n_neighbors": 10, "n_components": 2, "min_dist": 0.1, "n_epochs": 20}

    files = {"file": ("train.csv", csv_train, "text/csv")}

    res_train = requests.post(f"{BASE_URL}/train", files=files, data=train_payload)

    assert res_train.status_code == 200, f"Erreur Train: {res_train.text}"
    data_train = res_train.json()

    access_key = data_train["access_key"]

    # --- 2. TEST /transform ---
    df_new = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
    csv_new = df_new.to_csv(index=False).encode()

    transform_payload = {"access_key": access_key, "n_epochs": 10}

    files_new = {"file": ("new.csv", csv_new, "text/csv")}

    res_transform = requests.post(f"{BASE_URL}/transform", files=files_new, data=transform_payload)

    assert res_transform.status_code == 200, f"Erreur Transform: {res_transform.text}"
    data_transform = res_transform.json()

    embedding = np.array(data_transform["embedding"])
    assert embedding.shape == (10, 2)


def test_legacy_umap(start_api):
    """
    Test the endpoint legacy /umap (all-in-one)
    """
    df = pd.DataFrame(np.random.rand(20, 4))
    csv_buffer = df.to_csv(index=False).encode()

    files = {"file": ("legacy.csv", csv_buffer, "text/csv")}
    data = {"n_neighbors": 5}

    response = requests.post(f"{BASE_URL}/umap", files=files, data=data)

    assert response.status_code == 200
    embedding = np.array(response.json()["embedding"])
    assert embedding.shape == (20, 2)


def test_error_handling(start_api):
    """
    Vérifie que l'API renvoie bien une erreur 403 pour une mauvaise clé
    """
    files = {"file": ("test.csv", b"a,b\n1,2", "text/csv")}
    data = {"access_key": "cle_inexistante"}

    res = requests.post(f"{BASE_URL}/transform", files=files, data=data)
    assert res.status_code == 403

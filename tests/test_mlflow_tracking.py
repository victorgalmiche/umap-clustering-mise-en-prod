# import mlflow
# import numpy as np
# import os
# from umap_algo.umap_class import umap_mapping
# from adapter.mlflow_tracker import ExperimentTracker, UmapStorage


#def test_umap_pyfunc_inference(iris_split, tmp_path):
    
#    X_train = iris_split["X_train"]
#    X_test = iris_split["X_test"]

#    umap_model = umap_mapping(n_neighbors=15, n_components=2)
#    umap_model.fit_transform(X_train)

    # pyfunc_model = UmapStorage(umap_model)

    # tracker = ExperimentTracker(
    #     experiment_name="test_umap_experiment", 
    #     run_name="pytest_run", 
    #     run_tags={"user": os.getenv("USER")}
    # )

    # with tracker.run():
    #     tracker.log_pyfunc_model(pyfunc_model, artifact_path="umap_model")
    #     run_id = mlflow.active_run().info.run_id

    # loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/umap_model")

    # Y_test_pred = loaded_model.predict(X_test)
    
    # assert isinstance(Y_test_pred, np.ndarray), "inference should return np.darray"
    # assert Y_test_pred.shape[0] == X_test.shape[0], "Number of lines does not align X_test shape"
    # assert Y_test_pred.shape[1] == 2, "Number of dimensions should be n_components"
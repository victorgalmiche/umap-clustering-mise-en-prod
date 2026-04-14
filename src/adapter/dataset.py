def register(X_train, Y_train, model_config):

    X_train.to_parquet(model_config["path_X_train"])
    Y_train.to_parquet(model_config["path_Y_train"])

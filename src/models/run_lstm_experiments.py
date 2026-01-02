from src.models.train_utils import *


def baseline_lstm_pipeline(device, config):
    df = load_data(config["data_path"])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    train_df, test_df = split_data(df, train_size=config["train_size"])
    train_df, test_df, scaler = scale_data(train_df, test_df, target_col=config["target_col"], scaler_path=config["scaler_path"])

    print("Train split size:", len(train_df))
    print("Test split size:", len(test_df))

    X_all_train, y_all_train = create_time_windows(train_df, n_steps=config["n_steps"], target_col=config["target_col"])
    X_train, y_train, X_val, y_val = split_windows_train_val(X_all_train, y_all_train, val_ratio=0.10)

    X_test, y_test = create_time_windows(test_df, n_steps=config["n_steps"], target_col=config["target_col"])

    model = load_model(input_shape=(X_train.shape[1], X_train.shape[2]), loss_fn=BCE_LOSS)

    with tf.device(device):
        history = train(model, X_train, y_train, X_val, y_val, 
                        epochs=config["epochs"], batch_size=config["batch_size"], model_path=config["model_path"])

        eval(model, X_test, y_test, save_path=config["results_path"])
    return history



def oversampled_lstm_pipeline(device, config):
    df_train = load_data(config['train_data_path'])
    df_test = load_data(config['test_data_path'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test = df_test.set_index('date')    

    target_col = config["target_col"]
    n_steps = config["n_steps"]

    X_all_train, y_all_train = reshape_flat_windows(df_train, target_col)
    X_train, y_train, X_vval, y_val = split_windows_train_val(X_all_train, y_all_train, val_ratio=0.10)
    X_test, y_test = create_time_windows(df_test, n_steps=n_steps, target_col=target_col)

    print(X_train.shape[1] == n_steps)
    print(X_train.shape[1], X_train.shape[2])

    model = load_model(input_shape=(X_train.shape[1], X_train.shape[2]), loss_fn=BCE_LOSS)

    with tf.device(device):
        history = train(model, X_train, y_train, X_test, y_test, 
              epochs=config["epochs"], batch_size=config["batch_size"], model_path=config["model_path"])
        
        eval(model, X_test, y_test, save_path=config["results_path"])
    return history



def focal_lstm_pipeline(device, config):
    df = load_data(config["data_path"])
    df['date'] = pd.to_datetime(df['date'])
    df= df.set_index('date') 

    train_df, test_df = split_data(df, train_size=config["train_size"])

    train_df, test_df, _ = scale_data(train_df, test_df, target_col=config["target_col"], scaler_path=config["scaler_path"])

    X_all_train, y_all_train = create_time_windows(train_df, n_steps=config["n_steps"], target_col=config["target_col"])
    X_train, y_train, X_val, y_val = split_windows_train_val(X_all_train, y_all_train, val_ratio=0.10)
    X_test, y_test = create_time_windows(test_df, n_steps=config["n_steps"], target_col=config["target_col"] )

    focal_loss_fn = focal_loss(alpha=config.get("alpha", 0.25), gamma=config.get("gamma", 2.0))

    model = load_model(input_shape=(X_train.shape[1], X_train.shape[2]),loss_fn=focal_loss_fn)

    with tf.device(device):
        history = train(model, X_train, y_train, X_val, y_val,
            model_path=config["model_path"], epochs=config["epochs"], batch_size=config["batch_size"])

        eval(model, X_test, y_test, save_path=config["results_path"])
    return history


def class_weights_lstm_pipeline(device, config):
    df = load_data(config["data_path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    train_df, test_df = split_data(df, train_size=config["train_size"])
    train_df, test_df, _ = scale_data(train_df, test_df,target_col=config["target_col"], scaler_path=config["scaler_path"])

    X_all_train, y_all_train = create_time_windows(train_df, n_steps=config["n_steps"], target_col=config["target_col"])
    X_train, y_train, X_val, y_val = split_windows_train_val(X_all_train, y_all_train, val_ratio=0.10)
    X_test, y_test = create_time_windows(test_df, n_steps=config["n_steps"], target_col=config["target_col"])

    ir = (y_train == 0).sum() / (y_train == 1).sum()

    loss_fn = class_weights_loss({0: ir, 1: 1.0})

    model = load_model(input_shape=(X_train.shape[1], X_train.shape[2]),loss_fn=loss_fn)

    with tf.device(device):
        history = train(model, X_train, y_train, X_val, y_val,
            model_path=config["model_path"], epochs=config["epochs"], batch_size=config["batch_size"])

        eval(model, X_test, y_test, save_path=config["results_path"])
    return history



def cost_sensitive_adaptive_lstm_pipeline(device, config):
    df = load_data(config["data_path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    train_df, test_df = split_data(df, train_size=config["train_size"])
    train_df, test_df, _ = scale_data(train_df, test_df, target_col=config["target_col"], scaler_path=config["scaler_path"] )

    X_all_train, y_all_train = create_time_windows(train_df, n_steps=config["n_steps"], target_col=config["target_col"])
    X_train, y_train, X_val, y_val = split_windows_train_val(X_all_train, y_all_train, val_ratio=0.10)
    X_test, y_test = create_time_windows(test_df, n_steps=config["n_steps"], target_col=config["target_col"])

    print("Validation positives:", (y_val == 1).sum())
    print("Validation negatives:", (y_val == 0).sum())


    # overall imbalance ratio
    ir = (y_train == 0).sum() / (y_train == 1).sum()

    lambda_cb = CostSensitiveLambda(ir)

    loss_fn = cost_sensitive_loss(lambda_cb.lambda_major)

    model = load_model(input_shape=(X_train.shape[1], X_train.shape[2]), loss_fn=loss_fn)

    with tf.device(device):
        history = train(model, X_train, y_train, X_val, y_val,
            model_path=config["model_path"], epochs=config["epochs"], batch_size=config["batch_size"], callbacks=[lambda_cb])

        eval(model, X_test, y_test, save_path=config["results_path"])
    return history



def main(config_path):
    device = get_device()
    print("Using device: ", device)

    config = load_config(config_path)

    for key in ["model_path", "results_path", "scaler_path"]:
        if key in config:
            config[key] = PROJECT_ROOT / config[key]
            config[key].parent.mkdir(parents=True, exist_ok=True)


    if config['method'] == 'baseline':
        history = baseline_lstm_pipeline(device, config)

    elif config["method"] == "window_oversampling":
        print("Training with Oversampled data")
        history = oversampled_lstm_pipeline(device, config)
    
    elif config["method"] == "focal_loss":
        print("Training with Focal Loss")
        history = focal_lstm_pipeline(device, config)

    elif config["method"] == "class_weights":
        print("Training with Class-Weights")
        history = class_weights_lstm_pipeline(device, config)

    elif config["method"] == "cost_sensitive_adaptive":
        print("Training with Adaptive Cost-Sensitive Loss")
        history = cost_sensitive_adaptive_lstm_pipeline(device, config)
    
    else:
        raise ValueError(f"Unknown method: {config['method']}")
    
    return history

    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the config path.\n")
    
    config_path = sys.argv[1]
    main(config_path)
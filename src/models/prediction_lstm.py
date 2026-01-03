import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
import joblib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))  


from src.models.train_utils import load_data, split_data, create_time_windows


MODEL_PATH = PROJECT_ROOT / "outputs" / "lstm_class_weights_model.h5"
SCALER_PATH = PROJECT_ROOT / "outputs" / "scaler_class_weights.save"
DATA_PATH = PROJECT_ROOT / "datasets" / "data_merged.csv"

print("MODEL_PATH:", MODEL_PATH)
print("SCALER_PATH:", SCALER_PATH)
print("DATA_PATH:", DATA_PATH)

df = load_data(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")


train_df, test_df = split_data(df, train_size=0.8)


scaler = joblib.load(SCALER_PATH)
feature_cols = [c for c in test_df.columns if c != "label"]
test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

n_steps = 4
X_test, y_test = create_time_windows(test_df, n_steps=n_steps, target_col="label")

model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded")

preds = model.predict(X_test)
print(f"Predictions: {preds.shape}")


np.save("../../outputs/X_test_class_weights.npy", X_test.astype(np.float32))
np.save("../../outputs/preds_class_weights.npy", preds.astype(np.float32))
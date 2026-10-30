import joblib
import pandas as pd

# Load trained model
model = joblib.load("rul_random_forest_model.pkl")

# Load dataset
data = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None)

columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

data.columns = columns

print("====================================")
print("      ENGINE RUL PREDICTION")
print("====================================")

# Show available engine IDs
engine_id = int(input("Enter Engine ID: "))

engine_data = data[data["engine_id"] == engine_id]

if engine_data.empty:
    print("Engine ID not found.")
else:
    print("\nAvailable cycles:", engine_data["cycle"].min(),
          "to", engine_data["cycle"].max())

    cycle = int(input("Enter Current Cycle: "))

    sample = engine_data[engine_data["cycle"] == cycle]

    if sample.empty:
        print("Cycle not found for this engine.")
    else:
        # Keep exactly the features used during training
        X_sample = sample[model.feature_names_in_]

        prediction = model.predict(X_sample)[0]

        if prediction > 100:
           status = "Healthy"
        elif prediction >= 50:
           status = "Warning"
        else:
           status = "Critical"

        print("\n------------------------------------")
        print("Predicted Remaining Useful Life:",
        round(prediction, 2), "cycles")
        print("Maintenance Status:", status)
        print("------------------------------------")
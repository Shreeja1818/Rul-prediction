import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Load dataset
file_path = "data/train_FD001.txt"
data = pd.read_csv(file_path, sep=r"\s+", header=None)

print("Dataset loaded successfully!")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# 2. Add column names
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

# 3. Calculate RUL
max_cycle = data.groupby("engine_id")["cycle"].transform("max")
data["RUL"] = max_cycle - data["cycle"]

# 4. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 5. Dataset summary
print("\nDataset summary:")
print(data.describe())

# 6. RUL graph
engine_1 = data[data["engine_id"] == 1]

plt.plot(engine_1["cycle"], engine_1["RUL"])
plt.xlabel("Cycle")
plt.ylabel("Remaining Useful Life (RUL)")
plt.title("Engine 1 - RUL vs Cycle")
plt.show()

# 7. Train-test split
X = data.drop(columns=["RUL"])
y = data["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel trained successfully!")
print("MAE:", mae)
print("RMSE:", rmse)


plt.figure(figsize=(10, 5))

plt.plot(y_test.values[:100], label="Actual RUL")
plt.plot(y_pred[:100], label="Predicted RUL")

plt.xlabel("Test Samples")
plt.ylabel("RUL")
plt.title("Actual vs Predicted RUL")
plt.legend()
plt.show()


# Improved model using sensor and operating-setting features

feature_columns = [
    "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

X = data[feature_columns]
y = data["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nImproved Model Results")
print("MAE:", mae)
print("RMSE:", rmse)


# Check which sensors are most related to RUL

correlation = data.corr(numeric_only=True)["RUL"].sort_values()

print("\nCorrelation with RUL:")
print(correlation)


# Select important features based on correlation

selected_features = [
    "cycle",
    "sensor_11",
    "sensor_4",
    "sensor_15",
    "sensor_2",
    "sensor_17",
    "sensor_3",
    "sensor_20",
    "sensor_21",
    "sensor_7",
    "sensor_12"
]

X_selected = data[selected_features]
y_selected = data["RUL"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_selected, y_selected, test_size=0.2, random_state=42
)

model_selected = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_selected.fit(X_train_s, y_train_s)

y_pred_s = model_selected.predict(X_test_s)

mae_s = mean_absolute_error(y_test_s, y_pred_s)
rmse_s = np.sqrt(mean_squared_error(y_test_s, y_pred_s))

print("\nSelected Features Model")
print("MAE:", mae_s)
print("RMSE:", rmse_s)


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))


comparison = pd.DataFrame({
    "Actual_RUL": y_test.values[:10],
    "Predicted_RUL": y_pred[:10]
})

print("\nActual vs Predicted RUL:")
print(comparison)


plt.figure(figsize=(10, 5))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")

plt.show()


import joblib

joblib.dump(model, "rul_random_forest_model.pkl")

print("\nModel saved successfully!")
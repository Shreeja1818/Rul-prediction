import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

# ==============================
# Load Model and Dataset
# ==============================

model = joblib.load("rul_random_forest_model.pkl")

data = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

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


# ==============================
# Prediction Function
# ==============================

def predict_rul():

    try:
        engine_id = int(engine_entry.get())
        cycle = int(cycle_entry.get())

        engine_data = data[data["engine_id"] == engine_id]
        sample = engine_data[engine_data["cycle"] == cycle]

        if sample.empty:
            messagebox.showerror(
                "Invalid Input",
                "Engine ID or Cycle not found."
            )
            return

        X_sample = sample[model.feature_names_in_]

        prediction = model.predict(X_sample)[0]

        if prediction > 100:
            status = "HEALTHY"
        elif prediction >= 50:
            status = "WARNING"
        else:
            status = "CRITICAL"

        result_value.config(
            text=f"{prediction:.2f} cycles"
        )

        status_value.config(
            text=status
        )

        if status == "HEALTHY":
            status_value.config(fg="#16803C")
        elif status == "WARNING":
            status_value.config(fg="#B77900")
        else:
            status_value.config(fg="#C62828")

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers."
        )


# ==============================
# Main Window
# ==============================

root = tk.Tk()

root.title("Predictive Maintenance System")
root.geometry("700x600")
root.resizable(False, False)

root.configure(bg="#F4F7FB")


# ==============================
# Header
# ==============================

header = tk.Frame(
    root,
    bg="#17365D",
    height=120
)

header.pack(fill="x")

title = tk.Label(
    header,
    text="PREDICTIVE MAINTENANCE",
    font=("Arial", 25, "bold"),
    bg="#17365D",
    fg="white"
)

title.pack(pady=(25, 3))

subtitle = tk.Label(
    header,
    text="Aircraft Engine • Remaining Useful Life Prediction",
    font=("Arial", 11),
    bg="#17365D",
    fg="#DCE6F2"
)

subtitle.pack()


# ==============================
# Input Card
# ==============================

input_card = tk.Frame(
    root,
    bg="white",
    padx=35,
    pady=25
)

input_card.pack(
    padx=50,
    pady=30,
    fill="x"
)

input_title = tk.Label(
    input_card,
    text="Engine Information",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#17365D"
)

input_title.grid(
    row=0,
    column=0,
    columnspan=2,
    pady=(0, 20)
)


# Engine ID

engine_label = tk.Label(
    input_card,
    text="Engine ID",
    font=("Arial", 11),
    bg="white",
    fg="#444444"
)

engine_label.grid(
    row=1,
    column=0,
    sticky="w",
    pady=8
)

engine_entry = tk.Entry(
    input_card,
    font=("Arial", 12),
    width=25,
    relief="solid",
    bd=1
)

engine_entry.grid(
    row=1,
    column=1,
    padx=20,
    pady=8
)


# Cycle

cycle_label = tk.Label(
    input_card,
    text="Current Cycle",
    font=("Arial", 11),
    bg="white",
    fg="#444444"
)

cycle_label.grid(
    row=2,
    column=0,
    sticky="w",
    pady=8
)

cycle_entry = tk.Entry(
    input_card,
    font=("Arial", 12),
    width=25,
    relief="solid",
    bd=1
)

cycle_entry.grid(
    row=2,
    column=1,
    padx=20,
    pady=8
)


# ==============================
# Predict Button
# ==============================

predict_button = tk.Button(
    input_card,
    text="  PREDICT RUL  ",
    command=predict_rul,
    font=("Arial", 12, "bold"),
    bg="#17365D",
    fg="white",
    activebackground="#24558C",
    activeforeground="white",
    relief="flat",
    padx=15,
    pady=10,
    cursor="hand2"
)

predict_button.grid(
    row=3,
    column=0,
    columnspan=2,
    pady=(20, 5)
)


# ==============================
# Result Card
# ==============================

result_card = tk.Frame(
    root,
    bg="white",
    padx=30,
    pady=20
)

result_card.pack(
    padx=50,
    fill="x"
)


result_title = tk.Label(
    result_card,
    text="Prediction Result",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#17365D"
)

result_title.pack()


result_value = tk.Label(
    result_card,
    text="-- cycles",
    font=("Arial", 27, "bold"),
    bg="white",
    fg="#17365D"
)

result_value.pack(pady=10)


status_value = tk.Label(
    result_card,
    text="Status: --",
    font=("Arial", 15, "bold"),
    bg="white",
    fg="#666666"
)

status_value.pack()


# ==============================
# Footer
# ==============================

footer = tk.Label(
    root,
    text="Machine Learning Model: Random Forest Regressor",
    font=("Arial", 9),
    bg="#F4F7FB",
    fg="#777777"
)

footer.pack(pady=20)


# ==============================
# Start Application
# ==============================

root.mainloop()
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD & TRAIN MODEL
# =========================

data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_diabetes():
    try:
        values = [
            float(preg_entry.get()),
            float(glucose_entry.get()),
            float(bp_entry.get()),
            float(skin_entry.get()),
            float(insulin_entry.get()),
            float(bmi_entry.get()),
            float(dpf_entry.get()),
            float(age_entry.get())
        ]

        scaled = scaler.transform([values])
        result = model.predict(scaled)

        if result[0] == 1:
            messagebox.showwarning("Result", "⚠ HIGH RISK of Diabetes")
        else:
            messagebox.showinfo("Result", "✅ LOW RISK of Diabetes")

    except:
        messagebox.showerror("Error", "Please enter valid numbers")

# =========================
# GUI DESIGN
# =========================

root = tk.Tk()
root.title("Diabetes Prediction System")
root.geometry("400x500")

title = tk.Label(root, text="Diabetes Prediction", font=("Arial", 16, "bold"))
title.pack(pady=10)

def create_field(label):
    tk.Label(root, text=label).pack()
    entry = tk.Entry(root)
    entry.pack()
    return entry

preg_entry = create_field("Pregnancies")
glucose_entry = create_field("Glucose")
bp_entry = create_field("Blood Pressure")
skin_entry = create_field("Skin Thickness")
insulin_entry = create_field("Insulin")
bmi_entry = create_field("BMI")
dpf_entry = create_field("Diabetes Pedigree Function")
age_entry = create_field("Age")

tk.Button(root, text="Predict", command=predict_diabetes, bg="green", fg="white").pack(pady=20)

root.mainloop()
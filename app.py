from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model and scaler
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Feature columns
feature_columns = ['Wall_Material_Type', 'Roof_Type', 'Building_Orientation',
                   'Avg_Daily_Energy_Consumption', 'Avg_Daily_Water_Usage']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {col: [float(request.form[col])] for col in feature_columns}
        df = pd.DataFrame.from_dict(data)

        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]

        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        if not file:
            return jsonify({"error": "No file uploaded!"})

        df = pd.read_csv(file)

        # Preprocess uploaded data
        df_scaled = scaler.transform(df[feature_columns])
        predictions = model.predict(df_scaled)

        # Save results
        df["Predicted_Energy_Efficiency"] = predictions
        output_file = "static/predicted_results.csv"
        df.to_csv(output_file, index=False)

        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(len(predictions))), y=df["Energy_Efficiency_Score"], name="Actual"))
        fig.add_trace(go.Bar(x=list(range(len(predictions))), y=predictions, name="Predicted"))

        fig.update_layout(title="Actual vs Predicted Energy Efficiency",
                          xaxis_title="Data Points",
                          yaxis_title="Energy Efficiency Score",
                          barmode="group")

        graph_json = fig.to_json()

        return jsonify({
            "message": "Predictions saved!",
            "file": output_file,
            "graph": graph_json
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/metrics")
def metrics():
    try:
        df = pd.read_csv("renewable_energy_rural_home_dataset.csv")

        X = scaler.transform(df[feature_columns])
        y_actual = df["Energy_Efficiency_Score"]
        y_pred = model.predict(X)

        mae = np.mean(np.abs(y_actual - y_pred))
        mse = np.mean((y_actual - y_pred) ** 2)
        r2 = 1 - (np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))

        return jsonify({
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "R² Score": round(r2, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

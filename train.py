from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- Load models once ---
required_files = ['scaler.pkl', 'getPCA_model.pkl', 'rf_model.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    raise FileNotFoundError(f"Missing model files: {missing_files}")

scaler = joblib.load('scaler.pkl')
pca = joblib.load('getPCA_model.pkl')
model = joblib.load('rf_model.pkl')

# --- HTML Template ---
html_template = """
<!doctype html>
<html>
<head>
    <title>Power Consumption Analysis</title>
    <style>
        body { background-color: #FFD700; font-family: Arial, sans-serif; }
        .container {
            margin: 100px auto; width: 400px; background: #fff;
            padding: 20px; border-radius: 15px; box-shadow: 0px 0px 10px gray;
        }
        h2 { text-align: center; }
        input { width: 100%; padding: 10px; margin: 8px 0; border-radius: 8px; border: 1px solid gray; }
        button {
            background: red; color: white; padding: 10px; width: 100%;
            border: none; border-radius: 8px; font-size: 16px;
        }
        .result { text-align: center; margin-top: 20px; font-size: 18px; color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Power Consumption Analysis</h2>
        <form method="post">
            <input type="number" step="any" name="grp" placeholder="Global reactive power (0-1)" required>
            <input type="number" step="any" name="gi" placeholder="Global Intensity" required>
            <input type="number" step="any" name="sm1" placeholder="Submeter Reading 1" required>
            <input type="number" step="any" name="sm2" placeholder="Submeter Reading 2" required>
            <input type="number" step="any" name="sm3" placeholder="Submeter Reading 3" required>
            <button type="submit">Predict</button>
        </form>
        {% if prediction is not none %}
            <div class="result">
                {% if prediction|string %}
                    <h3 style="color:red;">{{ prediction }}</h3>
                {% else %}
                    <h3>Predicted Global Active Power:</h3>
                    <p><b>{{ "%.3f"|format(prediction) }}</b> kW</p>
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            def safe_float(val):
                try:
                    return float(val.strip().replace(',', '.'))
                except:
                    return None

            # Get inputs from form
            grp = safe_float(request.form.get("grp", ""))
            gi  = safe_float(request.form.get("gi", ""))
            sm1 = safe_float(request.form.get("sm1", ""))
            sm2 = safe_float(request.form.get("sm2", ""))
            sm3 = safe_float(request.form.get("sm3", ""))

            # Validate inputs
            inputs = [grp, 0.0, gi, sm1, sm2, sm3]  # Voltage is set to 0.0 as placeholder
            if None in [grp, gi, sm1, sm2, sm3]:
                prediction = "Error: Please enter only numeric values!"
            elif not (0 <= grp <= 1):
                prediction = "Error: Global reactive power must be between 0 and 1."
            else:
                # Create DataFrame in correct order
                data = pd.DataFrame([inputs],
                                    columns=['Global_reactive_power', 'Voltage', 'Global_intensity',
                                             'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])

                # Scale, PCA, predict
                scaled = scaler.transform(data)
                pca_data = pca.transform(scaled)
                prediction = float(model.predict(pca_data)[0])

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(html_template, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.permanent_session_lifetime = timedelta(minutes=60)

# --- Load models ---
MODEL_FILES = {
    "scaler": "scaler.pkl",
    "pca": "getPCA_model.pkl",
    "rf_model": "rf_model.pkl"
}

missing = [f for f in MODEL_FILES.values() if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing model files: {missing}")

scaler = joblib.load(MODEL_FILES["scaler"])
pca = joblib.load(MODEL_FILES["pca"])
model = joblib.load(MODEL_FILES["rf_model"])

DEFAULT_VOLTAGE = 240.0

# --- In-memory user store ---
users = {"admin": "1234"}  # ⚠️ For demo only, use DB & hashed passwords in production
user_histories = {}  # store history per user

# --- Utility: Require login ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

# Welcome page
@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

# Root redirects to welcome
@app.route("/")
def root():
    return redirect(url_for("welcome"))

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username in users and users[username] == password:
            session.permanent = True
            session["user"] = username
            # initialize history for this user if not exists
            if username not in user_histories:
                user_histories[username] = []
            return redirect(url_for("home"))

        return render_template("login.html", error="Invalid credentials. Try again.")
    return render_template("login.html")

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("register.html", error="Please fill in all fields.")
        if username in users:
            return render_template("register.html", error="Username already exists.")

        users[username] = password
        user_histories[username] = []
        return render_template("register.html", success="Registration successful! You can now log in.")
    return render_template("register.html")

# Home
@app.route("/home")
@login_required
def home():
    return render_template("home.html")

# About
@app.route("/about")
@login_required
def about():
    return render_template("about.html")

# Predict
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction, top_submeter, recommendation = None, "", ""

    if request.method == "POST":
        try:
            def safe_float(val):
                try:
                    return float(val.strip().replace(",", "."))
                except:
                    return None

            grp = safe_float(request.form.get("grp", ""))
            gi = safe_float(request.form.get("gi", ""))
            sm1 = safe_float(request.form.get("sm1", ""))
            sm2 = safe_float(request.form.get("sm2", ""))
            sm3 = safe_float(request.form.get("sm3", ""))

            if None in [grp, gi, sm1, sm2, sm3]:
                prediction = "⚠️ Please enter only numeric values."
            elif not (0 <= grp <= 1):
                prediction = "⚠️ Global reactive power must be between 0 and 1."
            else:
                # Prepare input
                inputs = [grp, DEFAULT_VOLTAGE, gi, sm1, sm2, sm3]
                data = pd.DataFrame([inputs], columns=[
                    "Global_reactive_power", "Voltage", "Global_intensity",
                    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
                ])

                # Transform and predict
                scaled = scaler.transform(data)
                pca_data = pca.transform(scaled)
                prediction = round(float(model.predict(pca_data)[0]), 8)

                # Find highest submeter usage
                submeter_vals = {"Sub_metering_1": sm1, "Sub_metering_2": sm2, "Sub_metering_3": sm3}
                top_submeter = max(submeter_vals, key=submeter_vals.get)
                recommendations = {
                    "Sub_metering_1": "Check kitchen appliances; they consume the most energy.",
                    "Sub_metering_2": "Laundry or water heating appliances are consuming most energy.",
                    "Sub_metering_3": "Heating, AC, or other utilities are consuming most energy."
                }
                recommendation = recommendations.get(top_submeter, "")

                # Save to user history
                username = session["user"]
                record = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "grp": grp, "gi": gi, "sm1": sm1, "sm2": sm2, "sm3": sm3,
                    "prediction": prediction, "top_submeter": top_submeter
                }
                user_histories[username].append(record)

        except Exception as e:
            prediction = f"❌ Error: {e}"

    return render_template("predict.html",
                           prediction=prediction,
                           top_submeter=top_submeter,
                           recommendation=recommendation)

# History
@app.route("/history")
@login_required
def history():
    username = session["user"]
    history_data = user_histories.get(username, [])
    return render_template("history.html", history=history_data)

# Contact
@app.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        print(f"📩 Message from {name} ({email}): {message}")
        return render_template("contact.html", success=True, name=name)

    return render_template("contact.html", success=False)

# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import pickle
import pandas as pd


applicaton = Flask(__name__)
app = applicaton


scaler = pickle.load(open(r'E:\Prasoon\Coding\AI_ML\PROJECT_2\MODEL\standardScaler.pkl', "rb"))
model = pickle.load(open(r'E:\Prasoon\Coding\AI_ML\PROJECT_2\MODEL\modelForPrediction.pkl', "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract and validate inputs
            def safe_float(value, field_name):
                if value is None or value.strip() == "":
                    raise ValueError(f"{field_name} is missing or invalid.")
                try:
                    return float(value)
                except ValueError:
                    raise ValueError(f"{field_name} must be a valid number.")

            # Extract form data
            temperature = safe_float(request.form.get("temperature"), "Temperature")
            RH = safe_float(request.form.get("RH"), "Relative Humidity")
            Ws = safe_float(request.form.get("Ws"), "Wind Speed")
            Rain = safe_float(request.form.get("Rain"), "Rain")
            FFMC = safe_float(request.form.get("FFMC"), "FFMC")
            DMC = safe_float(request.form.get("DMC"), "DMC")

            # Create DataFrame with the correct feature names and order
            column_names = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC"]
            new_data_df = pd.DataFrame([[temperature, RH, Ws, Rain, FFMC, DMC]], columns=column_names)

            # Validate against scaler features
            missing_features = set(scaler.feature_names_in_) - set(new_data_df.columns)
            extra_features = set(new_data_df.columns) - set(scaler.feature_names_in_)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            if extra_features:
                raise ValueError(f"Extra features: {extra_features}")

            # Transform the input data and make predictions
            new_data_scaled = scaler.transform(new_data_df)
            prediction = model.predict(new_data_scaled)

            # Interpret the prediction result
            result = 'FIRE' if prediction[0] == 1 else 'NO FIRE'
            return render_template('single_prediction.html', result=result)

        except ValueError as ve:
            app.logger.error(f"Validation error: {ve}")
            return render_template('single_prediction.html', result=str(ve))
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return render_template('single_prediction.html', result="An unexpected error occurred. Please try again.")
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

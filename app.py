import pickle
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes',
    'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service',
    'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service',
    'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check']   

def buid_features(form):
    raw = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': float(form['tenure']),
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': form['InternetService'],
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': form['Contract'],
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': float(form['MonthlyCharges']),
        'TotalCharges': float(form['tenure']) * float(form['MonthlyCharges'])
    }

    df = pd.get_dummies(pd.DataFrame([raw]))
    return df.reindex(columns=FEATURE_COLUMNS, fill_value=0)


PAGE = """
<!doctype html>
<html>
<body>
<h2>Telco Churn Prediction</h2>
<form action="/predict" method="post">
  Tenure: (months) <input type="number" name="tenure" required><br><br>
  Monthly Charges: <input type="number" step="any" name="MonthlyCharges" required><br><br>
  Contract:
  <select name="Contract" required>
    <option value="">-- select --</option>
    <option value="Month-to-month">Month-to-month</option>
    <option value="One year">One year</option>
    <option value="Two year">Two year</option>
    </select><br><br>
   Internet Service:
  <select name="InternetService" required>
    <option value="">-- select --</option>
    <option value="DSL">DSL</option>
    <option value="Fiber optic">Fiber optic</option>
    <option value="No">No</option>
    </select><br><br>
    <input type="submit" value="Predict">
</form>

{% if prediction %}
<h3>{{ prediction }} ({{ probability }})</h3>
{% endif %}

</body>
</html>
"""
@app.route('/', methods=['GET'])
def home():
    return render_template_string(PAGE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    features = buid_features(request.form)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if pred == 1:
        prediction = "Customer Will Churn"
    else:
        prediction = "Customer Will Stay"

    return render_template_string(
        PAGE,
        prediction=prediction,
        probability=round(prob * 100, 1)
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
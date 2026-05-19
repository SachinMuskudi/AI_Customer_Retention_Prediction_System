import os
import traceback
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Load Model & Normalizer ───────────────────────────────────────────────────
model      = None
normalizer = None
LOAD_ERROR = None

# ── Exact 32 features from Normalizer.pkl (verified from pkl directly) ────────
# Source trace:
#   main.py       → Telecom_Partner randomly assigned (Airtel/BSNL/Jio/VI-!dea)
#   var_out.py    → yeojohnson + IQR trim → col+'_trim' suffix
#   handle_missing→ TotalCharges null → col+'_mode' suffix  → TotalCharges_mode_trim
#   Categorical_to_num.py → OHE(drop='first') on 15 cat cols + OrdinalEncoder on Contract
#   feature_scaling.py   → Normalizer().fit_transform → saved as Normalizer.pkl
#   main.py label map: {'Yes':0, 'No':1}  ← REVERSED (Yes churn = class 0)

MODEL_FEATURES = [
    'SeniorCitizen',                              # numeric, kept as-is (binary)
    'tenure_trim',                                # tenure after yeo+IQR
    'MonthlyCharges_trim',                        # MonthlyCharges after yeo+IQR
    'TotalCharges_mode_trim',                     # TotalCharges: null→mode, then yeo+IQR
    'gender_Male',                                # OHE drop='first' (Female=baseline)
    'Partner_Yes',                                # OHE drop='first' (No=baseline)
    'Dependents_Yes',                             # OHE drop='first' (No=baseline)
    'PhoneService_Yes',                           # OHE drop='first' (No=baseline)
    'MultipleLines_No phone service',             # OHE
    'MultipleLines_Yes',                          # OHE (No=baseline)
    'InternetService_Fiber optic',                # OHE
    'InternetService_No',                         # OHE (DSL=baseline)
    'OnlineSecurity_No internet service',         # OHE
    'OnlineSecurity_Yes',                         # OHE (No=baseline)
    'OnlineBackup_No internet service',           # OHE
    'OnlineBackup_Yes',                           # OHE (No=baseline)
    'DeviceProtection_No internet service',       # OHE
    'DeviceProtection_Yes',                       # OHE (No=baseline)
    'TechSupport_No internet service',            # OHE
    'TechSupport_Yes',                            # OHE (No=baseline)
    'StreamingTV_No internet service',            # OHE
    'StreamingTV_Yes',                            # OHE (No=baseline)
    'StreamingMovies_No internet service',        # OHE
    'StreamingMovies_Yes',                        # OHE (No=baseline)
    'PaperlessBilling_Yes',                       # OHE drop='first' (No=baseline)
    'PaymentMethod_Credit card (automatic)',       # OHE
    'PaymentMethod_Electronic check',             # OHE
    'PaymentMethod_Mailed check',                 # OHE (Bank transfer=baseline)
    'Telecom_Partner_BSNL',                       # OHE (Airtel=baseline, dropped)
    'Telecom_Partner_Jio',                        # OHE
    'Telecom_Partner_VI-!dea',                    # OHE (VI-!dea exact string)
    'Contract_od',                                # OrdinalEncoder: M-t-m=0, 1yr=1, 2yr=2
]

# ── Contract ordinal map (matches OrdinalEncoder fit on training data) ─────────
CONTRACT_MAP = {
    'Month-to-month': 0.0,
    'One year':       1.0,
    'Two year':       2.0
}

# ── Threshold calibration ──────────────────────────────────────────────────────
# Label encoding is REVERSED in main.py: Yes(churn)→0, No(churn)→1
# So: prob[0] = P(customer LEAVES), prob[1] = P(customer STAYS)
# prob[0] range: 0% – 27%  (Normalizer L2 compresses range)
# Threshold = 25.72% (aligned to actual dataset churn rate of 26.5%)
# prob[0] > 25.72% → LEAVE   |   prob[0] <= 25.72% → STAY
CHURN_THRESHOLD = 25.72

try:

    with open('Model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Normalizer.pkl', 'rb') as f:
        normalizer = pickle.load(f)

    print(f"✅ Model loaded — {len(MODEL_FEATURES)} features | threshold={CHURN_THRESHOLD}%")

except Exception as e:
    LOAD_ERROR = str(e)
    print(f"❌ Startup error: {LOAD_ERROR}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    from flask import jsonify
    if LOAD_ERROR:
        return jsonify({'status': 'error', 'reason': LOAD_ERROR}), 500
    return jsonify({
        'status':    'ok',
        'model':     type(model).__name__,
        'features':  len(MODEL_FEATURES),
        'threshold': CHURN_THRESHOLD
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or normalizer is None:
        return render_template('index.html',
            prediction=f"❌ Model not loaded: {LOAD_ERROR}",
            churn_class="error")
    try:
        data = request.form.to_dict()

        # ── Build feature row ─────────────────────────────────────────────────
        df = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)

        # Numerical (raw values fed in; Normalizer handles scaling)
        df['SeniorCitizen']          = int(data.get('SeniorCitizen', 0))
        df['tenure_trim']            = float(data.get('tenure', 0))
        df['MonthlyCharges_trim']    = float(data.get('MonthlyCharges', 0))

        tc = data.get('TotalCharges', '').strip()
        df['TotalCharges_mode_trim'] = (
            float(tc) if tc and float(tc) > 0
            else float(data.get('tenure', 0)) * float(data.get('MonthlyCharges', 0))
        )

        # Personal
        if data.get('gender')          == 'Male': df['gender_Male']          = 1
        if data.get('Partner')         == 'Yes':  df['Partner_Yes']          = 1
        if data.get('Dependents')      == 'Yes':  df['Dependents_Yes']       = 1
        if data.get('PhoneService')    == 'Yes':  df['PhoneService_Yes']     = 1
        if data.get('PaperlessBilling')== 'Yes':  df['PaperlessBilling_Yes'] = 1

        # MultipleLines
        ml = data.get('MultipleLines', '')
        if ml == 'No phone service': df['MultipleLines_No phone service'] = 1
        elif ml == 'Yes':            df['MultipleLines_Yes']              = 1

        # InternetService
        inet = data.get('InternetService', '')
        if inet == 'Fiber optic': df['InternetService_Fiber optic'] = 1
        elif inet == 'No':        df['InternetService_No']          = 1

        # Internet-dependent services
        for svc in ['OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']:
            val = data.get(svc, '')
            if val == 'Yes':                df[f'{svc}_Yes']               = 1
            elif val == 'No internet service': df[f'{svc}_No internet service'] = 1

        # PaymentMethod (Bank transfer = baseline → all zeros)
        pay = data.get('PaymentMethod', '')
        if pay == 'Credit card (automatic)': df['PaymentMethod_Credit card (automatic)'] = 1
        elif pay == 'Electronic check':      df['PaymentMethod_Electronic check']        = 1
        elif pay == 'Mailed check':          df['PaymentMethod_Mailed check']            = 1

        # Telecom Partner (Airtel = baseline → all zeros)
        net = data.get('Networks', '')
        if net == 'BSNL':   df['Telecom_Partner_BSNL']    = 1
        elif net == 'Jio':  df['Telecom_Partner_Jio']     = 1
        elif net == 'Idea': df['Telecom_Partner_VI-!dea'] = 1
        # Airtel → all zeros (correct OHE baseline)

        # Contract ordinal
        contract = data.get('Contract', 'Month-to-month')
        df['Contract_od'] = CONTRACT_MAP.get(contract, 0.0)

        # ── Predict ───────────────────────────────────────────────────────────
        scaled = normalizer.transform(df)
        probs  = model.predict_proba(scaled)[0]

        # Label encoding is REVERSED: Yes(churn)=0, No(churn)=1
        # prob[0] = P(customer will LEAVE/churn)
        # prob[1] = P(customer will STAY/not churn)
        p_leave = round(float(probs[0]) * 100, 2)
        p_stay  = round(float(probs[1]) * 100, 2)

        if p_leave > CHURN_THRESHOLD:
            result      = f"⚠️ Customer Will LEAVE"
            detail      = f"Churn Probability: {p_leave}%"
            churn_class = "leave"
        else:
            result      = f"✅ Customer Will STAY"
            detail      = f"Retention Probability: {p_stay}%"
            churn_class = "stay"

        print(f"Network={net} | Contract={contract} | "
              f"P(leave)={p_leave}% | {churn_class.upper()}")

        return render_template('index.html',
                               prediction=result,
                               detail=detail,
                               churn_class=churn_class)

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html',
                               prediction=f"❌ Error: {str(e)}",
                               churn_class="error")


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Telco Churn Predictor — Flask")
    print(f"  URL   : http://127.0.0.1:5000")
    print(f"  Health: http://127.0.0.1:5000/health")
    print("=" * 55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
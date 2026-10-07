import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")


model = load_model()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Page spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Metric cards */
    .metric-box {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        background: rgba(255,255,255,0.03);
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 13px;
        color: #9ca3af;
        margin-top: 5px;
    }

    /* Section title */
    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine learning-based customer churn risk prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">7,043</div>
            <div class="metric-label">Training Customers</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">80.38%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:
    st.markdown(
        """
        <div class="metric-box">
            <div class="metric-value">ML</div>
            <div class="metric-label">Logistic Regression</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")
st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


with col3:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


st.divider()


# =========================================================
# CUSTOMER SERVICES
# =========================================================

st.markdown(
    '<div class="section-title">🛠️ Customer Services</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.divider()


# =========================================================
# BILLING INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">💳 Billing Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col2:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=500.0,
        value=70.0,
        step=1.0
    )


with col3:

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )


st.write("")
st.divider()


# =========================================================
# ANALYZE CUSTOMER
# =========================================================

st.markdown(
    '<div class="section-title">🔮 Analyze Customer</div>',
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    churn_percentage = probability * 100


    st.write("")


    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.error(
            f"🔴 HIGH CHURN RISK — {churn_percentage:.1f}%"
        )

        st.progress(
            probability,
            text=f"Churn Probability: {churn_percentage:.1f}%"
        )

        st.warning(
            "⚠️ This customer may be at risk of leaving. "
            "Consider targeted retention offers or customer support."
        )

    else:

        st.success(
            f"🟢 LOW CHURN RISK — {churn_percentage:.1f}%"
        )

        st.progress(
            probability,
            text=f"Churn Probability: {churn_percentage:.1f}%"
        )

        st.info(
            "ℹ️ This customer is currently predicted to have "
            "a lower risk of leaving the service."
        )


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    st.write("")
    st.subheader("📋 Prediction Summary")

    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "Prediction",
            "Churn" if prediction == 1 else "No Churn"
        )


    with result2:

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.1f}%"
        )


    with result3:

        risk_level = "High" if prediction == 1 else "Low"

        st.metric(
            "Risk Level",
            risk_level
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Customer Churn Prediction • Machine Learning Project<br>
        Built with Python, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
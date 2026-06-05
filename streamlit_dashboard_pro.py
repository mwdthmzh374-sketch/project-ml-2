
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mobile Sales ML Dashboard", layout="wide")

st.title("📱 Mobile Sales Prediction Dashboard")
st.markdown("Machine Learning Project - Random Forest & Gradient Boosting")

model_path = "models/best_model.pkl"

if not os.path.exists(model_path):
    st.error("Run train.py first to generate models/best_model.pkl")
    st.stop()

model = joblib.load(model_path)

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "📂 Batch Prediction",
    "ℹ️ Project Info"
])

with tab1:
    st.header("Dataset Overview")

    dataset_path = "data/cleaned_mobile_phone_sales_data.csv"

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        st.subheader("Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:
            selected = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(df[selected], bins=20)
            ax.set_title(selected)
            st.pyplot(fig)

    else:
        st.warning("Dataset not found in data folder")

with tab2:
    st.header("Batch Prediction")

    uploaded = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded is not None:

        data = pd.read_csv(uploaded)

        st.subheader("Input Data")
        st.dataframe(data)

        try:
            preds = model.predict(data)

            results = data.copy()
            results["Predicted_Sales"] = preds

            st.subheader("Prediction Results")
            st.dataframe(results)

            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(range(len(preds)), preds)
            ax.set_title("Predicted Sales")
            st.pyplot(fig)

            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(str(e))

with tab3:
    st.header("Project Information")

    st.markdown("""
### Models Used
- Random Forest Regressor
- Gradient Boosting Regressor

### Evaluation Metrics
- MAE
- MSE
- RMSE
- R² Score
- Cross Validation

### API
Run:

```bash
uvicorn api.main:app --reload
```

Swagger:

http://127.0.0.1:8000/docs
""")

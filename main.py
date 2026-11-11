
# ============================================================
# MACHINE LEARNING FOR SMART CAMPUS MANAGEMENT AT SRMIST
# Three models: Regression, Classification and Clustering
# Data source: smart campus data.xlsx
# ============================================================

import re
from datetime import datetime, time

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart Campus ML | SRMIST",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Machine Learning for Smart Campus Management")
st.subheader("SRM Institute of Science and Technology")
st.write(
    "An educational machine-learning application demonstrating "
    "canteen waiting-time prediction, waste classification and "
    "student transportation clustering."
)

st.divider()


# ------------------------------------------------------------
# 2. LOAD THE EXCEL DATA
# ------------------------------------------------------------

FILE_PATH = "datasets/smart campus data.xlsx"


def clean_column_names(data):
    """Remove extra spaces from Excel column names."""
    data = data.copy()
    data.columns = [
        re.sub(r"\s+", " ", str(column)).strip()
        for column in data.columns
    ]
    return data


@st.cache_data
def load_data(file_path):
    regression_data = pd.read_excel(
        file_path, sheet_name="regression data"
    )

    classification_data = pd.read_excel(
        file_path, sheet_name="classification data"
    )

    clustering_data = pd.read_excel(
        file_path, sheet_name="clustering data"
    )

    regression_data = clean_column_names(regression_data)
    classification_data = clean_column_names(classification_data)
    clustering_data = clean_column_names(clustering_data)

    return regression_data, classification_data, clustering_data


try:
    regression_data, classification_data, clustering_data = load_data(
        FILE_PATH
    )

except Exception as error:
    st.error("Unable to load the Excel workbook.")
    st.write("Check that the file is inside the datasets folder.")
    st.code("datasets/smart campus data.xlsx")
    st.exception(error)
    st.stop()


# ------------------------------------------------------------
# 3. CHECK THE DATA
# ------------------------------------------------------------

required_columns = {
    "regression": [
        "Canteen ID", "Time", "Queue", "Counter",
        "Waiting Time", "Avg Service Time"
    ],
    "classification": [
        "Waste Type", "Material", "Biodegradable",
        "Recyclable", "Weight(in gm)", "Category"
    ],
    "clustering": [
        "Transport Mode", "Distance (in km)",
        "Travel Time (in min)", "Days Per Week",
        "Daily Travel Cost (₹)"
    ]
}

datasets = {
    "Regression": regression_data,
    "Classification": classification_data,
    "Clustering": clustering_data
}

for name, data in datasets.items():
    if data.empty:
        st.error(f"{name} dataset is empty.")
        st.stop()

for name, columns in required_columns.items():
    data = datasets[name.capitalize()]

    missing_columns = [column for column in columns if column not in data.columns]

    if missing_columns:
        st.error(
            f"Missing columns in {name} dataset: {missing_columns}"
        )
        st.stop()

if any(data.isnull().any().any() for data in datasets.values()):
    st.error(
        "Missing values were found in the workbook. "
        "Please check the Excel data before training."
    )
    st.stop()


# ------------------------------------------------------------
# 4. PREPARE TIME DATA FOR REGRESSION
# ------------------------------------------------------------

def convert_time_to_minutes(value):
    """Convert Excel time or a time string into minutes after midnight."""

    if pd.isna(value):
        return np.nan

    if isinstance(value, datetime):
        return value.hour * 60 + value.minute

    if isinstance(value, time):
        return value.hour * 60 + value.minute

    if isinstance(value, (int, float)):
        # Excel may store time as a fraction of one day.
        return float(value) * 1440

    parsed_time = pd.to_datetime(str(value), errors="coerce")

    if pd.isna(parsed_time):
        raise ValueError(f"Invalid time value: {value}")

    return parsed_time.hour * 60 + parsed_time.minute


regression_data = regression_data.copy()

regression_data["Time_Minutes"] = regression_data["Time"].apply(
    convert_time_to_minutes
)

REGRESSION_FEATURES = [
    "Canteen ID",
    "Time_Minutes",
    "Queue",
    "Counter",
    "Avg Service Time"
]

REGRESSION_TARGET = "Waiting Time"

CLASSIFICATION_FEATURES = [
    "Waste Type",
    "Material",
    "Biodegradable",
    "Recyclable",
    "Weight(in gm)"
]

CLASSIFICATION_TARGET = "Category"

CLUSTERING_FEATURES = [
    "Transport Mode",
    "Distance (in km)",
    "Travel Time (in min)",
    "Days Per Week",
    "Daily Travel Cost (₹)"
]


# ------------------------------------------------------------
# 5. REGRESSION MODEL
# CANTEEN WAITING-TIME PREDICTION
# ------------------------------------------------------------

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Canteen ID"]
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            [
                "Time_Minutes",
                "Queue",
                "Counter",
                "Avg Service Time"
            ]
        )
    ]
)

regression_model = Pipeline([
    ("preprocessor", regression_preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

X_reg = regression_data[REGRESSION_FEATURES]
y_reg = regression_data[REGRESSION_TARGET]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Evaluate using a separate test portion.
regression_evaluation_model = Pipeline([
    ("preprocessor", regression_preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

regression_evaluation_model.fit(X_reg_train, y_reg_train)
regression_test_predictions = regression_evaluation_model.predict(X_reg_test)

regression_mae = mean_absolute_error(
    y_reg_test, regression_test_predictions
)

regression_rmse = np.sqrt(
    mean_squared_error(y_reg_test, regression_test_predictions)
)

regression_r2 = r2_score(
    y_reg_test, regression_test_predictions
)

# Train the final prediction model on all available observations.
regression_model.fit(X_reg, y_reg)


# ------------------------------------------------------------
# 6. CLASSIFICATION MODEL
# WASTE CATEGORY CLASSIFICATION
# ------------------------------------------------------------

classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            [
                "Waste Type",
                "Material",
                "Biodegradable",
                "Recyclable"
            ]
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            ["Weight(in gm)"]
        )
    ]
)

classification_model = Pipeline([
    ("preprocessor", classification_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

X_class = classification_data[CLASSIFICATION_FEATURES]
y_class = classification_data[CLASSIFICATION_TARGET]

# Use stratification when each category has enough observations.
class_counts = y_class.value_counts()
number_of_classes = y_class.nunique()
test_observations = int(np.ceil(len(y_class) * 0.2))

use_stratification = (
    class_counts.min() >= 2
    and test_observations >= number_of_classes
)

X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class if use_stratification else None
)

classification_evaluation_model = Pipeline([
    ("preprocessor", classification_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

classification_evaluation_model.fit(X_class_train, y_class_train)

classification_test_predictions = (
    classification_evaluation_model.predict(X_class_test)
)

classification_accuracy = accuracy_score(
    y_class_test, classification_test_predictions
)

# Train final classifier using all available observations.
classification_model.fit(X_class, y_class)


# ------------------------------------------------------------
# 7. CLUSTERING MODEL
# STUDENT TRANSPORTATION PATTERN GROUPING
# ------------------------------------------------------------

clustering_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Transport Mode"]
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            [
                "Distance (in km)",
                "Travel Time (in min)",
                "Days Per Week",
                "Daily Travel Cost (₹)"
            ]
        )
    ]
)

number_of_clusters = min(3, len(clustering_data))

clustering_model = Pipeline([
    ("preprocessor", clustering_preprocessor),
    ("kmeans", KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    ))
])

X_cluster = clustering_data[CLUSTERING_FEATURES]

cluster_labels = clustering_model.fit_predict(X_cluster)

clustering_results = clustering_data.copy()
clustering_results["Cluster"] = cluster_labels


# ------------------------------------------------------------
# 8. SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("Smart Campus ML")
page = st.sidebar.radio(
    "Navigate to:",
    [
        "Dashboard",
        "Waiting-Time Prediction",
        "Waste Classification",
        "Transportation Clustering",
        "About the Project"
    ]
)


# ------------------------------------------------------------
# 9. DASHBOARD
# ------------------------------------------------------------

if page == "Dashboard":

    st.header("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Regression Records", len(regression_data))
    col2.metric("Classification Records", len(classification_data))
    col3.metric("Clustering Records", len(clustering_data))

    st.success("All three datasets loaded and models trained successfully.")

    st.subheader("Machine-Learning Models")

    st.write("**1. Regression:** Predicts canteen waiting time in minutes.")

    st.write("**2. Classification:** Predicts the category of a waste item.")

    st.write("**3. Clustering:** Groups students according to transportation patterns.")

    st.divider()

    st.subheader("Regression Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{regression_mae:.2f} min")
    col2.metric("RMSE", f"{regression_rmse:.2f} min")
    col3.metric("R² Score", f"{regression_r2:.2f}")

    st.caption(
        "Evaluation uses a held-out portion of the 30 observations. "
        "Results may vary with a different dataset."
    )

    st.subheader("Classification Model Evaluation")

    st.metric(
        "Test Accuracy",
        f"{classification_accuracy * 100:.2f}%"
    )

    st.caption(
        "Accuracy is measured on a held-out portion of the dataset. "
        "The small dataset means this is an educational demonstration."
    )


# ------------------------------------------------------------
# 10. WAITING-TIME PREDICTION PAGE
# ------------------------------------------------------------

elif page == "Waiting-Time Prediction":

    st.header("🍽️ Canteen Waiting-Time Prediction")

    st.write(
        "Enter the current canteen conditions to estimate "
        "the expected waiting time."
    )

    canteen_options = sorted(
        regression_data["Canteen ID"].astype(str).unique().tolist()
    )

    with st.form("regression_form"):

        canteen_id = st.selectbox(
            "Select Canteen ID",
            canteen_options
        )

        selected_time = st.time_input(
            "Current Time",
            value=time(12, 0)
        )

        queue = st.number_input(
            "Number of people in queue",
            min_value=0,
            max_value=1000,
            value=10
        )

        counter = st.number_input(
            "Number of active counters",
            min_value=1,
            max_value=100,
            value=2
        )

        avg_service_time = st.number_input(
            "Average Service Time (minutes)",
            min_value=0.1,
            max_value=120.0,
            value=2.0,
            step=0.1
        )

        submit_regression = st.form_submit_button(
            "Predict Waiting Time"
        )

    if submit_regression:

        time_minutes = (
            selected_time.hour * 60 + selected_time.minute
        )

        new_data = pd.DataFrame([{
            "Canteen ID": canteen_id,
            "Time_Minutes": time_minutes,
            "Queue": queue,
            "Counter": counter,
            "Avg Service Time": avg_service_time
        }])

        prediction = regression_model.predict(new_data)[0]
        prediction = max(0, prediction)

        st.success(
            f"Estimated Waiting Time: {prediction:.2f} minutes"
        )

        st.info(
            "This is a machine-learning estimate based on the "
            "patterns in the supplied dataset."
        )


# ------------------------------------------------------------
# 11. WASTE CLASSIFICATION PAGE
# ------------------------------------------------------------

elif page == "Waste Classification":

    st.header("♻️ Waste Classification")

    st.write(
        "Enter the properties of a waste item to predict "
        "its category."
    )

    waste_options = sorted(
        classification_data["Waste Type"].astype(str).unique().tolist()
    )

    material_options = sorted(
        classification_data["Material"].astype(str).unique().tolist()
    )

    biodegradable_options = sorted(
        classification_data["Biodegradable"].astype(str).unique().tolist()
    )

    recyclable_options = sorted(
        classification_data["Recyclable"].astype(str).unique().tolist()
    )

    with st.form("classification_form"):

        waste_type = st.selectbox(
            "Waste Type",
            waste_options
        )

        material = st.selectbox(
            "Material",
            material_options
        )

        biodegradable = st.selectbox(
            "Biodegradable",
            biodegradable_options
        )

        recyclable = st.selectbox(
            "Recyclable",
            recyclable_options
        )

        weight = st.number_input(
            "Weight (in grams)",
            min_value=0.1,
            max_value=100000.0,
            value=25.0,
            step=1.0
        )

        submit_classification = st.form_submit_button(
            "Classify Waste"
        )

    if submit_classification:

        new_data = pd.DataFrame([{
            "Waste Type": waste_type,
            "Material": material,
            "Biodegradable": biodegradable,
            "Recyclable": recyclable,
            "Weight(in gm)": weight
        }])

        prediction = classification_model.predict(new_data)[0]

        st.success(f"Predicted Waste Category: {prediction}")

        st.info(
            "The category is predicted from the waste characteristics "
            "learned by the classification model."
        )


# ------------------------------------------------------------
# 12. TRANSPORTATION CLUSTERING PAGE
# ------------------------------------------------------------

elif page == "Transportation Clustering":

    st.header("🚌 Student Transportation Pattern Clustering")

    st.write(
        "Enter a student's transportation details to identify "
        "the cluster with the most similar learned pattern."
    )

    transport_options = sorted(
        clustering_data["Transport Mode"].astype(str).unique().tolist()
    )

    with st.form("clustering_form"):

        transport_mode = st.selectbox(
            "Transport Mode",
            transport_options
        )

        distance = st.number_input(
            "Distance travelled (km)",
            min_value=0.0,
            max_value=500.0,
            value=3.0,
            step=0.5
        )

        travel_time = st.number_input(
            "Travel Time (minutes)",
            min_value=0.0,
            max_value=1000.0,
            value=15.0
        )

        days_per_week = st.number_input(
            "Days travelled per week",
            min_value=0,
            max_value=7,
            value=5
        )

        daily_cost = st.number_input(
            "Daily Travel Cost (₹)",
            min_value=0.0,
            max_value=10000.0,
            value=0.0,
            step=5.0
        )

        submit_clustering = st.form_submit_button(
            "Find Transportation Cluster"
        )

    if submit_clustering:

        new_student = pd.DataFrame([{
            "Transport Mode": transport_mode,
            "Distance (in km)": distance,
            "Travel Time (in min)": travel_time,
            "Days Per Week": days_per_week,
            "Daily Travel Cost (₹)": daily_cost
        }])

        predicted_cluster = clustering_model.predict(new_student)[0]

        st.success(f"Assigned to Cluster {predicted_cluster + 1}")

        cluster_students = clustering_results[
            clustering_results["Cluster"] == predicted_cluster
        ]

        st.write(
            f"Number of students in this cluster: "
            f"{len(cluster_students)}"
        )

        st.write("Common transportation mode in this cluster:")

        common_mode = (
            cluster_students["Transport Mode"]
            .mode()
        )

        if not common_mode.empty:
            st.info(str(common_mode.iloc[0]))

        st.caption(
            "Cluster numbers are labels generated by K-Means. "
            "They do not represent a ranking or performance score."
        )


# ------------------------------------------------------------
# 13. ABOUT THE PROJECT
# ------------------------------------------------------------

elif page == "About the Project":

    st.header("About This Project")

    st.write(
        "This project demonstrates how machine learning can support "
        "smart campus management at SRMIST."
    )

    st.subheader("Objectives")

    st.write(
        "**Regression:** Estimate canteen waiting time using queue "
        "conditions, time, counters and service time."
    )

    st.write(
        "**Classification:** Predict waste categories using material "
        "and physical properties."
    )

    st.write(
        "**Clustering:** Group students based on transportation "
        "mode, distance, travel time, frequency and cost."
    )

    st.subheader("Algorithms Used")

    st.write("- Random Forest Regressor")
    st.write("- Random Forest Classifier")
    st.write("- K-Means Clustering")

    st.subheader("Dataset Information")

    st.write(
        "The application reads three separate sheets from the supplied "
        "Excel workbook. Each sheet currently contains 30 observations."
    )

    st.warning(
        "This is an educational prototype using a small dataset. "
        "Its predictions and clusters should not be treated as "
        "validated real-world campus measurements."
    )


# ------------------------------------------------------------
# 14. FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Machine Learning for Smart Campus Management at SRMIST | "
    "Educational Project"
)

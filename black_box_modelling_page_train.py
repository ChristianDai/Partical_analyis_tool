"""User interface for training a black box model. User can upload a dataset, select a target column,
select a model, set hyperparameters, train the model, and download the trained model.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_percentage_error, r2_score, f1_score, accuracy_score
import pickle
import base64
import matplotlib.pyplot as plt
from black_box_modelling_page_inference import black_box_modelling_page_inference

def load_data():
    st.header("Load your data to get started")
    delimiter = st.text_input("Delimiter", value=";", key="delimiter")
    decimal = st.text_input("Decimal symbol", value=".", key="decimal")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.data = pd.read_csv(uploaded_file, delimiter=delimiter, decimal=decimal)
        elif uploaded_file.name.endswith('.xlsx'):
            st.session_state.data = pd.read_excel(uploaded_file)
    if 'data' in st.session_state:
        st.session_state.target_column = st.selectbox("Select the target column", st.session_state.data.columns)
    if 'data' in st.session_state:
        st.write("First few rows of your dataset:")
        temp_data = st.session_state.data.head()
        temp_data = temp_data.style.set_properties(**{'color': 'red'}, subset=[st.session_state.target_column])
        st.dataframe(temp_data)

def set_model_parameters(model):
    if model == "Random Forest":
        return set_random_forest_parameters()
    elif model == "XGBoost":
        return set_xgboost_parameters()
    elif model == "KNN":
        return set_knn_parameters()
    elif model == "Linear":
        return set_linear_parameters()
    elif model == "Logistic Regression":
        return set_logistic_regression_parameters()
    elif model == "Gradient Boosting":
        return set_gradient_boosting_parameters()

def set_gradient_boosting_parameters():
    #function for the ui elements to select the hyperparameters for the gradient boosting model
    params = {}
    params['n_estimators'] = st.number_input("Number of estimators", min_value=1, value=100, key="n_estimators_gradient_boosting")
    params['max_depth'] = st.number_input("Max depth", min_value=1, value=3, key="max_depth_gradient_boosting")
    params['learning_rate'] = st.number_input("Learning rate", min_value=0.01, value=0.1, key="learning_rate_gradient_boosting")
    return params

def set_linear_parameters():
    #function for the ui elements to select the hyperparameters for the linear model
    return {}

def set_logistic_regression_parameters():
    #function for the ui elements to select the hyperparameters for the logistic regression model
    params = {}
    params['C'] = st.number_input("Regularization strength", min_value=0.01, value=1.0, key="C_logistic_regression")
    return params

def set_random_forest_parameters():
    #function for the ui elements to select the hyperparameters for the random forest model
    params = {}
    params['n_estimators'] = st.number_input("Number of estimators", min_value=1, value=100, key="n_estimators_random_forest")
    params['max_depth'] = st.number_input("Max depth", min_value=1, value=3, key="max_depth_random_forest")
    return params

def set_knn_parameters():
    #function for the ui elements to select the hyperparameters for the knn model
    params = {}
    params['n_neighbors'] = st.number_input("Number of neighbors", min_value=1, value=5, key="n_neighbors_knn")
    return params

def set_xgboost_parameters():
    #function for the ui elements to select the hyperparameters for the xgboost model
    params = {}
    params['n_estimators'] = st.number_input("Number of estimators", min_value=1, value=100, key="n_estimators_xgboost")
    params['max_depth'] = st.number_input("Max depth", min_value=1, value=3, key="max_depth_xgboost")
    params['learning_rate'] = st.number_input("Learning rate", min_value=0.01, value=0.1, key="learning_rate_xgboost")
    params['subsample'] = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, key="subsample_xgboost")
    params['colsample_bytree'] = st.number_input("Colsample bytree", min_value=0.1, max_value=1.0, value=1.0, key="colsample_bytree_xgboost")
    return params

def train_model_regression(X_train, y_train, model_type, params):
    if model_type == "Random Forest":
        model = RandomForestRegressor(**params)
    elif model_type == "XGBoost":
        model = XGBRegressor(**params)
    elif model_type == "KNN":
        model = KNeighborsRegressor(**params)
    elif model_type == "Linear":
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_model_classification(X_train, y_train, model_type, params):
    if model_type == "Random Forest":
        model = RandomForestClassifier(**params)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(**params)
    elif model_type == "KNN":
        model = KNeighborsClassifier(**params)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mape, r2

def evaluate_model_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def download_model(model):
    output = pickle.dumps(model)
    # Display download button
    st.download_button(
        label="Download trained model",
        data=output,
        file_name="model.pkl",
        mime="application/octet-stream",
        key=f"download_{model}"
    )
    

def black_box_modelling_page():
    #sidebar radio with option to select inference or training
    sub_page = st.selectbox(
        "Choose a task:",
        ["Train your model and download it", "Upload your model and make inference"]
    )
    if sub_page == "Train your model and download it":
        black_box_modelling_page_train()
    elif sub_page == "Upload your model and make inference":
        black_box_modelling_page_inference()

def plot_predicted_vs_actual_regression(method):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    predicted_values = st.session_state.methods_models[method].predict(input_columns)

    fig, ax = plt.subplots()
    ax.scatter(actual_values, predicted_values, label='Data Points')
    ax.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'r--', lw=2, label='Ideal Line')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    return fig

def plot_actual_and_predicted_regression(method):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    predicted_values = st.session_state.methods_models[method].predict(input_columns)

    fig, ax = plt.subplots()
    ax.scatter(range(len(actual_values)), actual_values, label='Actual Values')
    ax.scatter(range(len(predicted_values)), predicted_values, label='Predicted Values')
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Values")
    ax.legend()
    return fig

def regression_task():
    method = st.selectbox("Choose a model", ["Random Forest", "XGBoost", "KNN", "Linear"])
    if st.button("Add Method"):
        if method not in st.session_state.methods_config:
            st.session_state.methods_config[method] = None

    if st.session_state.methods_config:
        tabs = st.tabs(list(st.session_state.methods_config.keys()))

        for i, method in enumerate(st.session_state.methods_config.copy().keys()):
            with tabs[i]:
                #each tab consists of method configuration column and a remove button in separate column
                conf_col, remove_button_col = st.columns([8, 1])
                with conf_col:
                    st.subheader(f"{method} Configuration")
                    st.session_state.methods_config[method] = set_model_parameters(method)
                with remove_button_col:
                    if st.button("❌", key=f"{method}_remove"):
                        st.session_state.methods_config.pop(method)
                        # we need to rerun the page to remove the method from the UI
                        st.rerun()

    if st.button("Train selected models"):
        X = st.session_state.data.drop(columns=[st.session_state.target_column])
        y = st.session_state.data[st.session_state.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.test_size, random_state=42)

        with st.spinner("Training the models..."):
            for method, params in st.session_state.methods_config.items():
                model = train_model_regression(X_train, y_train, method, params)
                st.session_state.methods_results[method] = evaluate_model_regression(model, X_test, y_test)
                # Save the trained model in session state
                st.session_state.methods_models[method] = model

        if 'methods_results' in st.session_state:
            st.success("Models trained successfully!")
            for method, results in st.session_state.methods_results.items():
                container = st.container(border=True)

               
                st.write(f"{method}:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"MAPE: {results[0]}")
                    st.write(f"R2: {results[1]}")
                with col2:
                    # Download the trained model
                    download_model(st.session_state.methods_models[method])
                # Plot graphs
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_predicted_vs_actual_regression(method))
                with col2:
                    st.pyplot(plot_actual_and_predicted_regression(method))
                

def classification_task():
    method = st.selectbox("Choose a model", ["Random Forest", "KNN", "Logistic Regression", "Gradient Boosting"])
    if st.button("Add Method"):
        if method not in st.session_state.methods_config:
            st.session_state.methods_config[method] = None

    if st.session_state.methods_config:
        tabs = st.tabs(list(st.session_state.methods_config.keys()))
        for i, method in enumerate(st.session_state.methods_config.copy().keys()):
            with tabs[i]:
                conf_col, remove_button_col = st.columns([8, 1])
                with conf_col:
                    st.subheader(f"{method} Configuration")
                    st.session_state.methods_config[method] = set_model_parameters(method)
                with remove_button_col:
                    if st.button("❌", key=f"{method}_remove"):
                        st.session_state.methods_config.pop(method)
                        st.rerun()

    if st.button("Train selected models"):
        X = st.session_state.data.drop(columns=[st.session_state.target_column])
        y = st.session_state.data[st.session_state.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.test_size, random_state=42)
        with st.spinner("Training the models..."):
            for method, params in st.session_state.methods_config.items():
                model = train_model_classification(X_train, y_train, method, params)
                st.session_state.methods_results[method] = evaluate_model_classification(model, X_test, y_test)
                st.session_state.methods_models[method] = model
        if 'methods_results' in st.session_state:
            st.success("Models trained successfully!")
            for method, results in st.session_state.methods_results.items():
                st.write(f"{method}:")
                st.write(f"Accuracy: {results[0]}")
                st.write(f"F1 Score: {results[1]}")
                download_model(st.session_state.methods_models[method])
            
def black_box_modelling_page_train():
    st.title("Black box modelling")

    with st.expander("Load data", expanded=True):
        load_data()

    if ('data' and 'target_column') in st.session_state:
        
        # Initialize session state variables to store the configuration and results of the methods
        if 'methods_config' not in st.session_state:
            st.session_state.methods_config = {}
            st.session_state.methods_results = {}
            st.session_state.methods_models = {}

        task = st.selectbox("Choose a task", ["Regression", "Classification"])

        #if target column is numerical and user selected classification task, show error
        if task == "Classification" and st.session_state.data[st.session_state.target_column].dtype == np.number:
            st.error("Classification task can only be performed on categorical target columns.")
            return

        #if target column is categorical and user selected regression task, show error
        if task == "Regression" and st.session_state.data[st.session_state.target_column].dtype != np.number:
            st.error("Regression task can only be performed on numerical target columns.")
            return
        
        # Small intructions for the user
        st.write("Please select the model, the hyperparameters and the test size for the train test split.")

        # Define train test split with slider
        test_size = st.slider("Choose test size (test/train split ratio):", min_value=0.1, max_value=0.9, value=0.2, step=0.05)
        st.session_state.test_size = test_size  # Store the test size in session state

        # Select model
        if task == "Regression":
            regression_task()
        elif task == "Classification":
            classification_task()

        

        
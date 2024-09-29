import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gplearn.genetic import SymbolicRegressor

# Step 1: User uploads the dataset
st.title("Symbolic Regression Application")
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

    st.write("First 5 rows of the dataset:")
    st.write(st.session_state.df.head())

    # Step 2: User selects target column
    st.header("Step 2: Select Target Column")
    target_column = st.selectbox("Select the target column", st.session_state.df.columns)

    # Initialize or load a list to store preprocessing steps
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = []

    # Container for displaying the current state of the dataframe
    df_container = st.empty()

    # Function to apply preprocessing steps and update dataframe
    def apply_preprocessing(df, preprocessing_method):
        if preprocessing_method == "StandardScaler":
            scaler = StandardScaler()
            st.session_state.df = scaler.fit_transform(st.session_state.df)
        elif preprocessing_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            st.session_state.df = scaler.fit_transform(st.session_state.df)
        # Display updated dataframe
        #df_container.write(df)

    # Step 3: User applies preprocessing
    st.header("Step 3: Apply Preprocessing")
    preprocessing_method = st.selectbox("Select preprocessing method", ["None", "StandardScaler", "MinMaxScaler"])
    if preprocessing_method != "None":
        st.write(f"Selected preprocessing: {preprocessing_method}")
        if st.button("Apply Preprocessing"):
            st.session_state.preprocessing_steps.append(preprocessing_method)
            apply_preprocessing(st.session_state.df, preprocessing_method)
            #print(st.session_state.df.head())
    st.dataframe(st.session_state.df)
    if st.button("Skip Preprocessing or Go to Modules Selection"):
        st.session_state.skip_preprocessing = True

    # Step 4: User selects symbolic regression methods
    if st.session_state.get('skip_preprocessing', False):
        st.header("Step 4: Select Symbolic Regression Methods")
        selected_methods = st.multiselect("Select regression methods", ["SymbolicRegressor"])

        regressor_params = {}
        if "SymbolicRegressor" in selected_methods:
            st.subheader("SymbolicRegressor Parameters")
            regressor_params['population_size'] = st.number_input("Population Size", 100, 1000, 500)
            regressor_params['generations'] = st.number_input("Generations", 10, 1000, 100)
            regressor_params['stopping_criteria'] = st.number_input("Stopping Criteria", 0.01, 1.0, 0.01)
            regressor_params['p_crossover'] = st.number_input("Crossover Probability", 0.0, 1.0, 0.9)
            regressor_params['p_subtree_mutation'] = st.number_input("Subtree Mutation Probability", 0.0, 1.0, 0.01)
            regressor_params['p_hoist_mutation'] = st.number_input("Hoist Mutation Probability", 0.0, 1.0, 0.01)
            regressor_params['p_point_mutation'] = st.number_input("Point Mutation Probability", 0.0, 1.0, 0.01)
            regressor_params['max_samples'] = st.number_input("Max Samples", 0.1, 1.0, 1.0)
            regressor_params['verbose'] = st.checkbox("Verbose", True)

        if st.button("Run"):
            st.header("Step 5: Results and Logs")
            st.write("Running Symbolic Regressor...")

            # Prepare data
            X = st.session_state.current_df.drop(columns=[target_column])
            y = st.session_state.current_df[target_column]

            # Initialize and run Symbolic Regressor
            regressor = SymbolicRegressor(**regressor_params)
            regressor.fit(X, y)

            st.write("Training complete. Results:")
            st.write(f"Best Program: {regressor._program}")
            st.write(f"Score: {regressor.score(X, y)}")

    # Display the preprocessing steps applied
    st.sidebar.header("Preprocessing Steps Applied")
    st.sidebar.write(st.session_state.preprocessing_steps)
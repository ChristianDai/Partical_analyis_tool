import streamlit as st
import pandas as pd

def load_data(delimiter, decimal):
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.original_data = pd.read_csv(uploaded_file, delimiter=delimiter, decimal=decimal)
        elif uploaded_file.name.endswith('.xlsx'):
            st.session_state.original_data = pd.read_excel(uploaded_file)
        return True
    return False

    # Step 1: Load data
    st.header("Step 1: Load Data")
    

    if load_data(delimiter, decimal):
        st.write("First few rows of your dataset:")
        st.write(st.session_state.original_data.head())

        # Step 2: Select target column
        st.header("Step 2: Select Target Column")
        st.session_state.target_column = st.selectbox("Select the target column", st.session_state.original_data.columns)
        preprocessing_option = st.radio("Would you like to preprocess the data?", ("Skip Preprocessing", "Go to Data Preprocessing"))

        if preprocessing_option == "Go to Data Preprocessing":

            # Step 3: Data Preprocessing
            st.header("Step 3: Data Preprocessing")

            # Initialize session state for preprocessed data and steps
            if 'preprocessed_data' not in st.session_state:
                st.session_state.preprocessed_data = st.session_state.original_data.copy()
            if 'preprocessing_steps' not in st.session_state:
                st.session_state.preprocessing_steps = []

            preprocess_method = st.selectbox("Select a preprocessing method", ["Augmenter", "NNFunctions", "Normalization", "Standardization", "Write your own code", "multiply all columns by 10"])
            preprocessing_configuration = get_preprocessing_configuration(preprocess_method)
            st.write(st.session_state)
            if st.button("Preview Preprocessing", key="preview_button"):
                st.write("Preview of preprocessed data:")
                st.session_state.preprocessed_data = apply_nnfunctions(st.session_state.preprocessed_data, config=preprocessing_configuration)
                st.write(st.session_state.preprocessed_data.head())

            if st.button("Save Preprocessing", key="save_button"):
                preprocess_data(st.session_state.preprocessed_data, preprocess_method, config=preprocessing_configuration)
                st.session_state.preprocessing_steps.append((preprocess_method))
                #st.experimental_rerun()  # To update the interface after saving preprocessing

            if st.button("Reset DataFrame", key="reset_button"):
                st.session_state.preprocessed_data = reset_data()
                st.session_state.preprocessing_steps = []
                #st.experimental_rerun()  # To update the interface after resetting

            display_preprocessed_data()

            if st.button("Skip Preprocessing", key="skip_button"):
                st.session_state.skip_preprocessing = True

        # Step 4: Select and configure symbolic regression methods
        if 'skip_preprocessing' in st.session_state and st.session_state.skip_preprocessing:
            st.header("Step 4: Configure Symbolic Regression Methods")
            algorithm, params = choose_algorithm()

            # Run algorithm (implementation later)
            if st.button("Run", key="run_button"):
                st.write(f"Running {algorithm} with parameters: {params}")
                st.write("Results of the algorithm will be displayed here")
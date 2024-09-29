import streamlit as st
import pandas as pd
import pickle

def load_model():
    with st.expander("Load your Model", expanded=True):
        st.header("Load your model")
        uploaded_file = st.file_uploader("Upload your model", type=["pkl"])
        if uploaded_file is not None:
            st.session_state.model = pickle.load(uploaded_file)
            st.success("Model loaded successfully!")

def load_data():
    with st.expander("Load your Data", expanded=True):
        st.header("Load your data")
        delimiter = st.text_input("Delimiter", value=";", key="delimiter")
        decimal = st.text_input("Decimal symbol", value=".", key="decimal")
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file, delimiter=delimiter, decimal=decimal)
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state.data = pd.read_excel(uploaded_file)
            st.session_state.selected_columns = list(st.session_state.data.columns)
        if 'data' in st.session_state:
            st.write("First few rows of your dataset:")
            st.write(st.session_state.data.head())

def select_columns():
    if 'data' in st.session_state:
        st.header("Select columns for inference")
        selected_columns = st.multiselect(
            "Select columns to keep",
            st.session_state.data.columns.values,
            default=st.session_state.data.columns.values
        )
        st.session_state.selected_columns = selected_columns
        st.session_state.X = st.session_state.data[selected_columns]

def make_inference():
    if 'model' in st.session_state and 'X' in st.session_state:
        if st.button("Make inference"):
            predictions = st.session_state.model.predict(st.session_state.X)
            st.write("Predictions:")
            # Create a DataFrame with selected columns and predictions
            temp_df = pd.concat([st.session_state.X, pd.Series(predictions, name="predictions")], axis=1)
            
            # Display the styled DataFrame
            st.dataframe(temp_df.style.set_properties(**{'color': 'red'}, subset=["predictions"]))
            
            # Prepare CSV for download (without styling)
            csv = temp_df.to_csv(index=False)
            st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

def black_box_modelling_page_inference():
    st.title("Upload model and make inference")
    
    load_model()
    
    if 'model' in st.session_state:
        load_data()
    
    if 'data' and 'model' in st.session_state:
        select_columns()

    if 'X' in st.session_state:
        make_inference()

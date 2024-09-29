import streamlit as st
import pandas as pd
import time
from config1 import AUGMENTER_DEFAULT_CONFIG, NNFUNCTIONS_DEFAULT_CONFIG

def load_data():
    st.header("Load your data to get started")
    delimiter = st.text_input("Delimiter", value=";", key="delimiter")
    decimal = st.text_input("Decimal symbol", value=".", key="decimal")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.original_data = pd.read_csv(uploaded_file, delimiter=delimiter, decimal=decimal)
        elif uploaded_file.name.endswith('.xlsx'):
            st.session_state.original_data = pd.read_excel(uploaded_file)
    if 'original_data' in st.session_state:
        st.write("First few rows of your dataset:")
        st.write(st.session_state.original_data.head())


def apply_nnfunctions(config):
    from preprocessing.nnfunctions import NNfunctions
    nnfunctions = NNfunctions(config)
    st.session_state.preprocessed_data = nnfunctions.fit_transform(st.session_state.preprocessed_data)

def apply_augmenter(config):
    from preprocessing.augmenter import Augmenter
    augmenter = Augmenter(**config)
    st.session_state.preprocessed_data = augmenter.fit_transform(st.session_state.preprocessed_data)

def apply_user_code():
    from streamlit_ace import st_ace
    import pandas as pd
    import numpy as np
    code = st_ace('''import numpy as np\nimport pandas as pd\n# Write your code here, transform "df" variable. Example: \ndf['new_column'] = df['old_column'] * 10''', language="python")
    if code:
        try:
            # get context for the code
            local_context = {
                'pd': pd,
                'np': np,
                'df': st.session_state.preprocessed_data.copy()
            }

            # execute the code
            exec(code, {}, local_context)
            
            # update the preprocessed data
            st.session_state.preprocessed_data = local_context['df']
            
            st.success("Code run successfully!")
        except Exception as e:
            st.error(f"Error happened: {e}")

def apply_preprocessing(method):
    """function to get the configuration for the selected method"""
    if method == "Augmenter":
        with st.form('Configuration'):
            st.subheader("Augmenter Configuration")
            config = get_config_from_dict(AUGMENTER_DEFAULT_CONFIG)
            submit = st.form_submit_button('Submit')
            if submit:
                apply_augmenter(config)
    elif method == "NNFunctions":
        with st.form('Configuration'):
            st.subheader("NNFunctions Configuration")
            config = get_config_from_dict(NNFUNCTIONS_DEFAULT_CONFIG)
            submit = st.form_submit_button('Submit')
            if submit:
                apply_nnfunctions(config)
    elif method == "Write your own code":
        apply_user_code()
    elif method == "Edit data manually":
        st.write("Edit your data manually. Double click on a cell to edit it.")
        st.session_state.preprocessed_data = st.data_editor(st.session_state.preprocessed_data)
        

def get_config_from_dict(config_dict, parent_key=''):
    """
    takes default configuration dictionary and returns the configuration from the user input
    depending on the data type of the value in the dictionary"""
    config = {}
    for key, value in config_dict.items():
        full_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            config[key] = get_config_from_dict(value, parent_key=full_key)
        elif isinstance(value, (int, float)):
            config[key] = st.number_input(full_key, value=value, key=full_key)
        elif isinstance(value, str):
            config[key] = st.text_input(full_key, value=value, key=full_key)
        elif isinstance(value, bool):
            config[key] = st.checkbox(full_key, value=value, key=full_key)
    return config

def proceed_to_preprocessing():
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = st.session_state.original_data.copy()

def preprocessing_page():
    st.title("Preprocessing")
    with st.expander("Load data", expanded=True):
        load_data()
        if 'original_data' in st.session_state:
            st.button("Proceed to preprocessing", on_click=proceed_to_preprocessing)

    if 'preprocessed_data' in st.session_state:     
        preprocess_method = st.selectbox("Select a preprocessing method", ["Augmenter", "NNFunctions", "Write your own code", "Edit data manually"])
        apply_preprocessing(preprocess_method)
        st.dataframe(st.session_state.preprocessed_data)
        

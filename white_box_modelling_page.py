import os
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import redirect as rd
from sybren_page import update_sybren_model_settings, run_sybren
from sympy import *
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from config1 import SIMULATED_ANNEALING_DEFAULT_CONFIG, SimulatedAnnealingConfiguration, GPLEARN_DEFAULT_CONFIG, PYSR_DEFAULT_CONFIG
from retrieve_dataset_statistics import determine_dataset_statistics
from sqlalchemy import text

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

def run_simulated_annealing(config):
    from util.SymbolicRegressionAnnealer import Symbolic_Regression_Annealer
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    target_column = st.session_state.data[st.session_state.target_column].values.reshape(-1, 1)
    simannealing_config = SimulatedAnnealingConfiguration.from_dict(config)
    simannealing_config.number_of_independent_variables = np.shape(input_columns)[1]
    simannealer = Symbolic_Regression_Annealer(input_columns, target_column, simannealing_config)
    #redirecting the output of the logger to a string buffer
    beststate, bestenergy = simannealer.anneal()
    expression_str = str(beststate)
    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'sqr': lambda x: x ** 2,
        'abs': lambda x: abs(x),
        'sqrt': lambda x: x ** 0.5,
        'negative': lambda x: -x,
    }
    for i, col in enumerate(st.session_state.data.columns):
        converter[f'X{i}'] = Symbol(col)
    expression_sympy = sympify(expression_str, locals=converter)
    expression_sympy = expression_sympy.xreplace({n: round(n, 5) for n in expression_sympy.atoms(Number)})
    return Equality(Symbol(st.session_state.target_column), expression_sympy)

def run_gplearn(config):
    from gplearn.genetic import SymbolicRegressor
    gp_model = SymbolicRegressor(**config)
    gp_model.fit(st.session_state.data.drop(columns=[st.session_state.target_column]), st.session_state.data[st.session_state.target_column])
    best_program = gp_model._program
    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'sqr': lambda x: x ** 2,
        'abs': lambda x: abs(x),
        'sqrt': lambda x: x ** 0.5,
        'neg': lambda x: -x,
        'square': lambda x: x ** 2,
        'cube': lambda x: x ** 3,
    }
    for i, col in enumerate(st.session_state.data.columns):
        converter[f'X{i}'] = Symbol(col)
    expression_str = str(best_program)
    expression_sympy = sympify(expression_str, locals=converter)
    expression_sympy = expression_sympy.xreplace({n: round(n, 5) for n in expression_sympy.atoms(Number)})
    return Equality(Symbol(st.session_state.target_column), expression_sympy)

def run_pysr(config):
    from pysr import PySRRegressor
    import pickle

    pysr_model = PySRRegressor(**config)
    # pickle the model to 'sr_model.pkl' file
    with open('temp_pysr_model.pkl', 'wb') as file:
        pickle.dump(pysr_model, file)
    # write data to a csv, X - first columns, y has to be the last column 
    temp_df = pd.concat([st.session_state.data.drop(columns=[st.session_state.target_column]), st.session_state.data[st.session_state.target_column]], axis=1)
    # rename the columns to X0, X1, X2, ...
    temp_df.columns = [f'X{i}' for i in range(len(temp_df.columns) - 1)] + [st.session_state.target_column]
    temp_df.to_csv('temp_data.csv', index=False)
    # run the model through the subprocess
    os.system('python run_pysr.py temp_data.csv temp_pysr_model.pkl')
    # load the model from the file
    with open('temp_pysr_model.pkl', 'rb') as file:
        pysr_model = pickle.load(file)
    
    rhs = pysr_model.sympy()
    # convert X0, X1, X2, ... to symbols
    converter = {}
    for i, col in enumerate(st.session_state.data.columns):
        converter[f'X{i}'] = Symbol(col)
    rhs = rhs.subs(converter)
    equality = Equality(Symbol(st.session_state.target_column), rhs)
    return equality

def get_config_from_dict(config_dict, parent_key=''):
    # This function is used to get the configuration from the dictionary, if config is not complex, this function can be used
    config = {}
    col1, col2 = st.columns(2)
    i = 0
    for key, value in config_dict.items():
        current_col = col1 if i % 2 == 0 else col2
        full_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            config[key] = get_config_from_dict(value, parent_key=full_key)
        elif isinstance(value, bool):
            config[key] = current_col.checkbox(full_key, value=value, key=full_key)
        elif isinstance(value, (int, float)):
            config[key] = current_col.number_input(full_key, value=value, key=full_key)
        elif isinstance(value, str):
            config[key] = current_col.text_input(full_key, value=value, key=full_key)
        elif isinstance(value, list):
            if all(isinstance(j, (int, float)) for j in value):
                config[key] = st.multiselect(full_key, options=value, default=value, key=full_key)
            elif all(isinstance(j, str) for j in value):
                with st.container():
                    nested_col1, nested_col2 = st.columns(2)
                    selected_values = []
                    for k, element in enumerate(value):
                        nested_col = nested_col1 if k % 2 == 0 else nested_col2
                        if nested_col.checkbox(element, value=element in value):
                            selected_values.append(element)
                    config[key] = selected_values
        i += 1
    return config

def get_method_config(method):
    if method == "Simmulated Annealing":
        config = get_config_from_dict(SIMULATED_ANNEALING_DEFAULT_CONFIG)
        return config
    elif method == "SyBReN":
        return update_sybren_model_settings()
    elif method == "gplearn":
        return get_config_gplearn(GPLEARN_DEFAULT_CONFIG)
    elif method == "PySR":
        return get_config_pysr(PYSR_DEFAULT_CONFIG)

def get_config_gplearn(default_config):
    config = {}
    col1, col2 = st.columns(2)
    #container for functions
    with st.container(border=True):
        cont_col1, cont_col2 = st.columns(2)
        functions = default_config['function_set']
        selected_functions = []
        for i, function in enumerate(functions):
            current_col = cont_col1 if i % 2 == 0 else cont_col2
            if current_col.checkbox(function.name, value=function.name in selected_functions, key=f'gplearn_function_{function.name}'):
                selected_functions.append(function)
        config['function_set'] = selected_functions
    config['population_size'] = col1.number_input("Population Size", value=default_config['population_size'], key='gplearn_population_size')
    config['tournament_size'] = col2.number_input("Tournament Size", value=default_config['tournament_size'], key='gplearn_tournament_size')
    config['init_method'] = col2.selectbox("Initialization Method", options=['half and half', 'full', 'grow'], key='gplearn_init_method')
    config['generations'] = col1.number_input("Generations", value=default_config['generations'], key='gplearn_generations')
    config['stopping_criteria'] = col2.number_input("Stopping Criteria", value=default_config['stopping_criteria'], key='gplearn_stopping_criteria')
    config['p_crossover'] = col1.number_input("Crossover Probability", value=default_config['p_crossover'], key='gplearn_p_crossover')
    config['p_subtree_mutation'] = col2.number_input("Subtree Mutation Probability", value=default_config['p_subtree_mutation'], key='gplearn_p_subtree_mutation')
    config['p_hoist_mutation'] = col1.number_input("Hoist Mutation Probability", value=default_config['p_hoist_mutation'], key='gplearn_p_hoist_mutation')
    config['p_point_mutation'] = col2.number_input("Point Mutation Probability", value=default_config['p_point_mutation'], key='gplearn_p_point_mutation')
    config['max_samples'] = col1.number_input("Max Samples", value=default_config['max_samples'], key='gplearn_max_samples')
    config['parsimony_coefficient'] = col2.number_input("Parsimony Coefficient", value=default_config['parsimony_coefficient'], key='gplearn_parsimony_coefficient')
    return config

def get_config_pysr(default_config):
    config = {}
    col1, col2 = st.columns(2)
    config['model_selection'] = col1.selectbox('Model Selection', options=['accuracy', 'best', 'score'], index=0, key='pysr_model_selection')
    config['niterations'] = col2.number_input('Number of Iterations', value=default_config['niterations'], key='pysr_niterations')
    config['populations'] = col1.number_input('Populations', value=default_config['populations'], key='pysr_populations')
    config['population_size'] = col2.number_input('Population Size', value=default_config['population_size'], key='pysr_population_size')
    config['maxsize'] = col1.number_input('Max Size', value=default_config['maxsize'], min_value=7, key='pysr_maxsize')
    config['weight_optimize'] = col2.number_input('Weight Optimize', min_value=float(0), max_value=float(1), value=default_config['weight_optimize'], key='pysr_weight_optimize')
    #container for binary operators
    with st.container(border=True):
        binary_operators = ['+', '-', '*', '/', '^']
        selected_operators = []
        cont_col1, cont_col2 = st.columns(2)
        for i, operator in enumerate(binary_operators):
            current_col = cont_col1 if i % 2 == 0 else cont_col2
            if current_col.checkbox('\\'+operator, value=operator in default_config["binary_operators"], key=f'pysr_binary_operator_{operator}'):
                selected_operators.append(operator)
        config['binary_operators'] = selected_operators
    #container for unary operators
    with st.container(border=True):
        unary_operators = ['neg','square', 'cube', 'exp', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan']
        selected_operators = []
        cont_col1, cont_col2 = st.columns(2)
        for i, operator in enumerate(unary_operators):
            current_col = cont_col1 if i % 2 == 0 else cont_col2
            if current_col.checkbox(operator, value=operator in default_config["unary_operators"], key=f'pysr_unary_operator_{operator}'):
                selected_operators.append(operator)
        config['unary_operators'] = selected_operators
    config["extra_sympy_mappings"] = {"inv": lambda x: 1 / x,"myfunction1": lambda x: x ** 9,}
    return config


def remove_method(method):
    st.session_state.methods_config.pop(method)
    st.session_state.methods_results.pop(method)


def calculate_r2(equation):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    input_symbols = symbols(st.session_state.data.columns.tolist())

    predicted_values = []
    for row in input_columns:
        prediction = equation.rhs.subs(dict(zip(input_symbols, row)))
        if prediction.is_real and not prediction.has('I'):
            predicted_values.append(float(prediction))
        else:
            try:
                real_part = prediction.as_real_imag()[0]
                if real_part.is_real:
                    predicted_values.append(float(real_part))
            except (TypeError, ValueError) as e:
                print(f"Error converting prediction: {prediction}, error: {e}")
                continue

    r2 = r2_score(actual_values, predicted_values)
    return r2

def calculate_mape(equation):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    input_symbols = symbols(st.session_state.data.columns.tolist())

    predicted_values = []
    for row in input_columns:
        prediction = equation.rhs.subs(dict(zip(input_symbols, row)))
        if prediction.is_real and not prediction.has('I'):
            predicted_values.append(float(prediction))
        else:
            try:
                real_part = prediction.as_real_imag()[0]
                if real_part.is_real:
                    predicted_values.append(float(real_part))
            except (TypeError, ValueError) as e:
                print(f"Error converting prediction: {prediction}, error: {e}")
                continue

    mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
    return mape

def calculate_mse(equation):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    input_symbols = symbols(st.session_state.data.columns.tolist())

    predicted_values = []
    for row in input_columns:
        prediction = equation.rhs.subs(dict(zip(input_symbols, row)))
        if prediction.is_real and not prediction.has('I'):
            predicted_values.append(float(prediction))
        else:
            try:
                real_part = prediction.as_real_imag()[0]
                if real_part.is_real:
                    predicted_values.append(float(real_part))
            except (TypeError, ValueError) as e:
                print(f"Error converting prediction: {prediction}, error: {e}")
                continue

    mse = mean_squared_error(actual_values, predicted_values)
    return mse

def calculate_mae(equation):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    input_symbols = symbols(st.session_state.data.columns.tolist())

    predicted_values = []
    for row in input_columns:
        prediction = equation.rhs.subs(dict(zip(input_symbols, row)))
        if prediction.is_real and not prediction.has('I'):
            predicted_values.append(float(prediction))
        else:
            try:
                real_part = prediction.as_real_imag()[0]
                if real_part.is_real:
                    predicted_values.append(float(real_part))
            except (TypeError, ValueError) as e:
                print(f"Error converting prediction: {prediction}, error: {e}")
                continue

    mae = mean_absolute_error(actual_values, predicted_values)
    return mae

def plot_predicted_vs_actual(equation):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    input_symbols = symbols(st.session_state.data.columns.tolist())
    
    predicted_values = []
    actual_values_filtered = []
    for row, actual in zip(input_columns, actual_values):
        prediction = equation.rhs.subs(dict(zip(input_symbols, row)))
        if prediction.is_real:
            predicted_values.append(float(prediction))
            actual_values_filtered.append(actual)

    fig, ax = plt.subplots()
    ax.scatter(actual_values_filtered, predicted_values, label='Data Points')
    ax.plot([min(actual_values_filtered), max(actual_values_filtered)], [min(actual_values_filtered), max(actual_values_filtered)], 'r--', lw=2, label='Ideal Line')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend()
    return fig

def plot_actual_and_predicted(equation):
    input_columns = st.session_state.data.drop(columns=[st.session_state.target_column]).values
    actual_values = st.session_state.data[st.session_state.target_column].values
    input_symbols = symbols(st.session_state.data.columns.tolist())
    
    predicted_values = []
    actual_values_filtered = []
    for row, actual in zip(input_columns, actual_values):
        prediction = equation.rhs.subs(dict(zip(input_symbols, row)))
        if prediction.is_real:
            predicted_values.append(float(prediction))
            actual_values_filtered.append(actual)

    fig, ax = plt.subplots()
    ax.scatter(range(len(actual_values_filtered)), actual_values_filtered, label='Actual Values')
    ax.scatter(range(len(predicted_values)), predicted_values, label='Predicted Values')
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Values")
    ax.legend()
    return fig

def run_white_box_modelling():
    for method, config in st.session_state.methods_config.items():
        start_time = time.time()
        if method == "Simmulated Annealing":
            st.session_state.methods_results[method] = run_simulated_annealing(config)
        elif method == "SyBReN":
            st.session_state.methods_results[method] = run_sybren()
        elif method == "gplearn":
            st.session_state.methods_results[method] = run_gplearn(config)
        elif method == "PySR":
            st.session_state.methods_results[method] = run_pysr(config)
        else:
            st.error(f"Method {method} not implemented")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.session_state.methods_time[method] = elapsed_time

def remove_temp_files():
    if os.path.exists('temp_pysr_model.pkl'):
        os.remove('temp_pysr_model.pkl')
    if os.path.exists('temp_data.csv'):
        os.remove('temp_data.csv')
    # pysr creates temp files starting with 'hall_of_fame_'
    for file in os.listdir():
        if file.startswith('hall_of_fame_'):
            os.remove(file)
    

def save_run_details(conn, model_name, r2, rmse, mae, mape):
    with conn.session as session:
        result = session.execute(text("""
            INSERT INTO runs (model_name, r2, rmse, mae, mape)
            VALUES (:model_name, :r2, :rmse, :mae, :mape)
        """), {
            "model_name": model_name,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        })
        session.commit()
        return result.lastrowid

def save_dataset_details(conn, run_id, data_frame, target_column):
    statistics = determine_dataset_statistics(data_frame, target_column)

    with conn.session as session:
        session.execute(text("""
            INSERT INTO dataset_details (
                run_id, num_samples, num_features, num_missing_values, 
                biggest_dependent_value, smallest_dependent_value, smallest_absolute_dependent_value,
                biggest_independent_value, smallest_independent_value, smallest_absolute_independent_value,
                dependent_mean, dependent_std_dev, dependent_quartile_1, dependent_median, dependent_quartile_3,
                dependent_geometric_mean, dependent_geometric_std_dev, dependent_outliers_proportion,
                dependent_skewness, dependent_kurtosis
            ) VALUES (
                :run_id, :num_samples, :num_features, :num_missing_values, 
                :biggest_dependent_value, :smallest_dependent_value, :smallest_absolute_dependent_value,
                :biggest_independent_value, :smallest_independent_value, :smallest_absolute_independent_value,
                :dependent_mean, :dependent_std_dev, :dependent_quartile_1, :dependent_median, :dependent_quartile_3,
                :dependent_geometric_mean, :dependent_geometric_std_dev, :dependent_outliers_proportion,
                :dependent_skewness, :dependent_kurtosis
            )
        """), {
            "run_id": run_id,
            "num_samples": statistics["Number of Data Points"],
            "num_features": statistics["Number of Columns"],
            "num_missing_values": data_frame.isnull().sum().sum(),
            "biggest_dependent_value": statistics["Biggest Dependent Value"],
            "smallest_dependent_value": statistics["Smallest Dependent Value"],
            "smallest_absolute_dependent_value": statistics["Smallest Absolute Dependent Value"],
            "biggest_independent_value": statistics["Biggest Independent Value"],
            "smallest_independent_value": statistics["Smallest Independent Value"],
            "smallest_absolute_independent_value": statistics["Smallest Absolute Independent Value"],
            "dependent_mean": statistics["Mean Dependent Variable"],
            "dependent_std_dev": statistics["Standard Deviation Dependent Variable"],
            "dependent_quartile_1": statistics["Quartile 1 Dependent Variable"],
            "dependent_median": statistics["Median Dependent Variable"],
            "dependent_quartile_3": statistics["Quartile 3 Dependent Variable"],
            "dependent_geometric_mean": statistics["Geometric Mean of Absolute Values Dependent Variable"],
            "dependent_geometric_std_dev": statistics["Geometric Standard Deviation of Absolute Values Dependent Variable"],
            "dependent_outliers_proportion": statistics["Proportion of IQR Method Outliers Dependent Variable"],
            "dependent_skewness": statistics["Skewness of Dependent Variable"],
            "dependent_kurtosis": statistics["Kurtosis of Dependent Variable"]
        })
        session.commit()

def write_details_to_db(model_name, r2, rmse, mae, mape):
    conn = st.connection('symbolic_regression_db', type='sql')
    try:
        # Write run details and get the run_id
        run_id = save_run_details(conn, model_name, r2, rmse, mae, mape)
        
        # Use the run_id to write dataset details
        save_dataset_details(conn, run_id, st.session_state.data, st.session_state.target_column)
        
        st.success(f"Run details and dataset details saved successfully with run_id: {run_id}")
    except Exception as e:
        st.error(f"An error occurred while saving to the database: {str(e)}")
        conn.reset()  # Reset the connection if an error occurs


def white_box_modelling_page():
    st.title("White box modelling")
    with st.expander("Load data", expanded=True):
        load_data()

    if 'data' and 'target_column' in st.session_state:
        st.header("Select one or multiple methods for white box modelling")

        if 'methods_config' not in st.session_state:
            st.session_state.methods_config = {}
            st.session_state.methods_results = {}
            st.session_state.methods_time = {}

        method = st.selectbox("Select a method", ["Simmulated Annealing", "SyBReN", "gplearn", "PySR"])
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
                        st.session_state.methods_config[method] = get_method_config(method)
                    with remove_button_col:
                        if st.button("❌", key=f"{method}_remove"):
                            st.session_state.methods_config.pop(method)
                            # we need to rerun the page to remove the method from the UI
                            st.rerun()

        if st.button("Run"):
            st.text('Logs:')
            log_placeholder = st.code("", language='text')
            #with rd.stdout(to=log_placeholder, buffer_separator='\n'), rd.stderr(to=log_placeholder, buffer_separator='\n'):
            with st.spinner("Running..."):
                run_white_box_modelling()
            
    
            st.write("Results:")
            for method, result in st.session_state.methods_results.items():
                if result is not None:
                    #print the equation
                    st.write(f"Method: {method}")
                    st.latex(latex(result))
                    #calculate R2 and MAPE
                    r2 = calculate_r2(result)
                    mape = calculate_mape(result)
                    mse = calculate_mse(result)
                    mae = calculate_mae(result)
                    st.write(f"R2: {r2:.6f}")
                    st.write(f"MAPE: {mape:.6f}")
                    st.write(f"MSE: {mse:.6f}")
                    st.write(f"MAE: {mae:.6f}")
                    st.write(f"Time taken: {st.session_state.methods_time[method]:.2f} seconds")
                    #plot the results
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = plot_predicted_vs_actual(result)
                        st.pyplot(fig1)
                    with col2:
                        fig2 = plot_actual_and_predicted(result)
                        st.pyplot(fig2)
                    #write the details to the database
                    write_details_to_db(method, r2, mse, mae, mape)
                    #remove the temp files
                    remove_temp_files()
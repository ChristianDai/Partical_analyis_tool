import streamlit as st
import time
import sys
import os
import numpy as np
import pandas as pd
import Model_Setting
from contextlib import contextmanager
from sympy import *
from config1 import SIMULATED_ANNEALING_DEFAULT_CONFIG, SimulatedAnnealingConfiguration

def update_sybren_model_settings():
    st.subheader("Model Settings")

    Model_Setting.file_Dataset = 'temp_dataset.csv'

    #checkboxes for different models
    model_options = {
        "GPactive": "GP Active",
        "DAactive": "DA Active",
        "DNNactive": "DNN Active",
        "FSactive": "FS Active",
        "CSTactive": "CST Active",
        "Modfuncactive": "Mod Active",
        "Pysractive": "Pysr Active",
        "GNNactive": "GNN Active"
    }
    modules_selection_expander = st.expander("Select the models to be used")
    for key, label in model_options.items():
        setattr(Model_Setting, key, modules_selection_expander.checkbox(label, value=getattr(Model_Setting, key)))

    if Model_Setting.Pysractive:
        update_pysr_model1()
    if Model_Setting.GPactive:
        update_gp_model1()
    if Model_Setting.DNNactive:
        update_nn()
    if Model_Setting.FSactive:
        update_fs()
    if Model_Setting.Modfuncactive:
        update_modfunc_model()
    if Model_Setting.GNNactive:
        update_gnn()



def update_gp_model1():
    with st.expander("GP Model"):
        Model_Setting.gp_model1.population_size = st.number_input('Population Size', min_value=1, value=Model_Setting.gp_model1.population_size)
        Model_Setting.gp_model1.generations = st.number_input('Generations', min_value=1, value=Model_Setting.gp_model1.generations, key='gp_model1_generations')
        Model_Setting.gp_model1.init_method = st.selectbox('Init Method', options=['half and half', 'full', 'grow'])
        Model_Setting.gp_model1.p_crossover = st.number_input('Crossover Percentage', min_value=0.0, max_value=1.0, value=Model_Setting.gp_model1.p_crossover)
        Model_Setting.gp_model1.p_subtree_mutation = st.number_input('Subtree Mutation Percentage', min_value=0.0, max_value=1.0, value=Model_Setting.gp_model1.p_subtree_mutation)
        Model_Setting.gp_model1.p_hoist_mutation = st.number_input('Hoist Mutation Percentage', min_value=0.0, max_value=1.0, value=Model_Setting.gp_model1.p_hoist_mutation)
        Model_Setting.gp_model1.p_point_mutation = st.number_input('Point Mutation Percentage', min_value=0.0, max_value=1.0, value=Model_Setting.gp_model1.p_point_mutation)
        Model_Setting.gp_model1.parsimony_coefficient = st.number_input('Parsimony Coefficient', min_value=0.0, max_value=1.0, value=Model_Setting.gp_model1.parsimony_coefficient)
        with st.container(border=True):
            function_set_options = ['add', 'sub', 'mul', 'div', 'sqrt','inv','cos','abs','log','sin','tan']
            selected_functions = []
            for func in function_set_options:
                if st.checkbox(func, value=func in Model_Setting.gp_model1.function_set, key=f'gp_model1_function_{func}'):
                    selected_functions.append(func)
            Model_Setting.gp_model1.function_set = selected_functions
        Model_Setting.GPmodel1_parameter=Model_Setting.get_GPmodel1_parameters()


def update_pysr_model1():
    with st.expander("PySR"):
        Model_Setting.Pysrmodel1.model_selection = st.selectbox('Model Selection', options=['accuracy', 'best', 'score'], index=0, key='pysr1_model_selection')
        Model_Setting.Pysrmodel1.population_size = st.number_input('Population Size', min_value=1, value=Model_Setting.Pysrmodel1.population_size, key='pysr1_population_size')
        Model_Setting.Pysrmodel1.maxsize = st.number_input('Max Size', min_value=1, value=Model_Setting.Pysrmodel1.maxsize, key='pysr1_maxsize')
        Model_Setting.Pysrmodel1.populations = st.number_input('Populations', min_value=1, value=Model_Setting.Pysrmodel1.populations, key='pysr1_populations')
        Model_Setting.Pysrmodel1.niterations = st.number_input('Iterations', min_value=1, value=Model_Setting.Pysrmodel1.niterations, key='pysr1_niterations')
        Model_Setting.Pysrmodel1.weight_optimize = st.number_input('Weight optimize', min_value=float(0), max_value=float(1), value=Model_Setting.Pysrmodel1.weight_optimize, key='pysr1_weight_optimize', format="%f")
        #container for binary operators
        with st.container(border=True):
            binary_operators = ['+', '-', '*', '/', '^']
            selected_operators = []
            for operator in binary_operators:
                if st.checkbox('\\'+operator, value=operator in Model_Setting.Pysrmodel1.binary_operators, key=f'pysr1_binary_operator_{operator}'):
                    selected_operators.append(operator)
            Model_Setting.Pysrmodel1.binary_operators = selected_operators
        #container for unary operators
        with st.container(border=True):
            unary_operators = ['neg','square', 'cube', 'exp', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan']
            selected_operators = []
            for operator in unary_operators:
                if st.checkbox(operator, value=operator in Model_Setting.Pysrmodel1.unary_operators, key=f'pysr1_unary_operator_{operator}'):
                    selected_operators.append(operator)
            Model_Setting.Pysrmodel1.unary_operators = selected_operators
        Model_Setting.Pysrmodel1.nested_constraints = st.text_area('Nested Constraints', value=None, key='pysr1_nested_constraints')
            
    
def update_fs():
    with st.expander("Fragment Selection"):
        Model_Setting.FS.population_size = st.number_input('Population Size', min_value=1, value=Model_Setting.FS.population_size, key='fs_population_size')
        Model_Setting.FS.generations = st.number_input('Generations', min_value=1, value=Model_Setting.FS.generations, key='fs_generations')
        Model_Setting.FS.metric = st.selectbox('Metric', options=['pearson', 'spearman', 'kendall'], index=0, key='fs_metric')
        #container for function set
        with st.container(border=True):
            function_set_options = ['add', 'mul', 'sub', 'div', 'sqrt','inv','log','abs','sin','cos', 'tan', 'pow']
            selected_functions = []
            col1, col2 = st.columns(2)
            for i, func in enumerate(function_set_options):
                current_col = col1 if i % 2 == 0 else col2
                if current_col.checkbox(func, value=func in Model_Setting.FS.function_set, key=f'fs_function_{func}'):
                    selected_functions.append(func)
            Model_Setting.FS.function_set = selected_functions

def update_nn():
    with st.expander("Neural Network"):
        Model_Setting.NN_Config['epoch'] = st.number_input('Epoch', min_value=1, value=Model_Setting.NN_Config['epoch'], key='nn_epoch')
        Model_Setting.NN_Config['bs'] = st.number_input('Batch Size', min_value=1, value=Model_Setting.NN_Config['bs'], key='nn_batch_size')
        Model_Setting.NN_Config['one-step regression'] = st.checkbox('Partial regression', value=Model_Setting.NN_Config['one-step regression'], key='nn_one_step_regression')
        Model_Setting.NN_Config['steps'] = st.number_input('Steps', min_value=1, value=Model_Setting.NN_Config['steps'], key='nn_steps')
        #container for equation library
        col1, col2 = st.columns(2)
        with col1:
            Model_Setting.NN_Config['Plus'] = st.checkbox('Plus', value=Model_Setting.NN_Config['Plus'], key='nn_plus')
            Model_Setting.NN_Config['Multiply'] = st.checkbox('Multiply', value=Model_Setting.NN_Config['Multiply'], key='nn_multiply')
            Model_Setting.NN_Config['ab**n'] = st.checkbox('ab^n', value=Model_Setting.NN_Config['ab**n'], key='nn_ab**n')
            Model_Setting.NN_Config['an_plus_cosb'] = st.checkbox('a^n+cos(b)', value=Model_Setting.NN_Config['an_plus_cosb'], key='nn_an_plus_cosb')
            Model_Setting.NN_Config['a+b**n'] = st.checkbox('a+b^n', value=Model_Setting.NN_Config['a+b**n'], key='nn_a+b**n')
        with col2:
            Model_Setting.NN_Config['Minus'] = st.checkbox('Minus', value=Model_Setting.NN_Config['Minus'], key='nn_minus')
            Model_Setting.NN_Config['Division'] = st.checkbox('Division', value=Model_Setting.NN_Config['Division'], key='nn_division')
            Model_Setting.NN_Config['a**n*cosb'] = st.checkbox('a^n*cos(b)', value=Model_Setting.NN_Config['a**n*cosb'], key='nn_a**n*cosb')
            Model_Setting.NN_Config['an_expb'] = st.checkbox('a^n*exp(b)', value=Model_Setting.NN_Config['an_expb'], key='nn_an_expb')
            Model_Setting.NN_Config['a-b**n'] = st.checkbox('a-b^n', value=Model_Setting.NN_Config['a-b**n'], key='nn_a-b')

def update_modfunc_model():
    with st.expander("Mod Function"):
        #container for function set
        with st.container(border=True):
            function_set_options = ['inv', 'squared', 'sqrt', 'log', 'sin','exp','cos','tan','asin','acos','atan']
            selected_functions = []
            for func in function_set_options:
                if st.checkbox(func, value=func in Model_Setting.gp_modelmod.function_set, key=f'gp_model_function_{func}'):
                    selected_functions.append(func)
            Model_Setting.gp_modelmod.function_set = selected_functions
        #container for ModGp
        with st.container(border=True):
            Model_Setting.gp_modelmod.population_size = st.number_input('Population Size', min_value=1, value=Model_Setting.gp_modelmod.population_size, key='gp_model_population_size')
            Model_Setting.gp_modelmod.generations = st.number_input('Generations', min_value=1, value=Model_Setting.gp_modelmod.generations)
            Model_Setting.gp_modelmod.init_method = st.selectbox('Init Method', options=['half and half', 'full', 'grow'], key='modfunc_gp_init_method')
        with st.container(border=True):
            Model_Setting.PysrmodelMod.model_selection = st.selectbox('Model Selection', options=['best', 'accuracy', 'score'], index=0)
            Model_Setting.PysrmodelMod.populations = st.number_input('Max Size', min_value=1, value=Model_Setting.PysrmodelMod.populations)
            Model_Setting.PysrmodelMod.maxsize = st.number_input('Max Size', min_value=1, value=Model_Setting.PysrmodelMod.maxsize, key='pysr_maxsize')
            Model_Setting.PysrmodelMod.population_size = st.number_input('Population Size', min_value=1, value=Model_Setting.PysrmodelMod.population_size, key='pysr_population_size')
            Model_Setting.PysrmodelMod.niterations = st.number_input('Iterations', min_value=1, value=Model_Setting.PysrmodelMod.niterations)

def update_gnn():
    with st.expander("GNN"):
        Model_Setting.n_features = st.number_input('Number of Features', min_value=1, value=Model_Setting.n_features)
        Model_Setting.extend_time = st.number_input('Extend Time', min_value=1, value=Model_Setting.extend_time)
        Model_Setting.num_batches = st.number_input('Number of Batches', min_value=1, value=Model_Setting.num_batches)
        Model_Setting.GNNpysrmodel.populations = st.number_input('Populations', min_value=1, value=Model_Setting.GNNpysrmodel.populations)
        Model_Setting.GNNpysrmodel.maxsize = st.number_input('Max Size', min_value=1, value=Model_Setting.GNNpysrmodel.maxsize, key='gnn_maxsize')
        Model_Setting.GNNpysrmodel.population_size = st.number_input('Population Size', min_value=1, value=Model_Setting.GNNpysrmodel.population_size)
        Model_Setting.GNNpysrmodel.niterations = st.number_input('Iterations', min_value=1, value=Model_Setting.GNNpysrmodel.niterations)
        #container for binary operators
        with st.container(border=True):
            binary_operators = ['+', '-', '*', '/', '^']
            selected_operators = []
            col1, col2 = st.columns(2)
            for operator in binary_operators:
                current_col = col1 if binary_operators.index(operator) % 2 == 0 else col2
                if current_col.checkbox('\\'+operator, value=operator in Model_Setting.GNNpysrmodel.binary_operators, key=f'gnn_binary_operator_{operator}'):
                    selected_operators.append(operator)
            Model_Setting.GNNpysrmodel.binary_operators = selected_operators
        #container for unary operators
        with st.container(border=True):
            unary_operators = ['sin','sqrt', 'log', 'exp', 'cos', 'abs', 'tan']
            selected_operators = []
            col1, col2 = st.columns(2)
            for operator in unary_operators:
                current_col = col1 if unary_operators.index(operator) % 2 == 0 else col2
                if current_col.checkbox(operator, value=operator in Model_Setting.GNNpysrmodel.unary_operators, key=f'gnn_unary_operator_{operator}'):
                    selected_operators.append(operator)
            Model_Setting.GNNpysrmodel.unary_operators = selected_operators


@contextmanager
def change_directory(path):
    original_sys_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_sys_path)

def parse_results():
    #parse results from Final_result.txt
    #results are in the next non-empty line after line containing "Result of"
    #try to parse the results in sympy format
    with open('Final_result.txt', 'r') as f:
        lines = f.readlines()
    result_line = None
    for i, line in enumerate(lines):
        if "Result of" in line:
            for j in range(i+1, len(lines)):
                if lines[j].strip() != '':
                    result_line = lines[j].strip()
                    break
            break
    if result_line is None:
        return None
    #parse the result line to sympy format
    #sympify wont work on equations with = sign, so we split the equation and take the right side
    rhs = result_line.split('=')[1]
    expression_sympy = sympify(rhs)
    return expression_sympy
                
def run_sybren():
    with change_directory('SyBReN'):
        #change order of columns, target column should be the last column
        temp_data = st.session_state.data[[col for col in st.session_state.data.columns if col != st.session_state.target_column] + [st.session_state.target_column]]
        temp_data.to_csv('temp_dataset.csv', index=False)
        from _program import RUN
        RUN()
        #return results in sympy format
        result_rhs = parse_results()
        if result_rhs is None:
            st.error("Error in running SyBReN. Please check the logs for more information.")
            return None
        #return Equation in sympy format with target column as LHS
        return Eq(Symbol(st.session_state.target_column), result_rhs)


def sybren_page():
    st.title("SyBREn")
    if 'original_data' not in st.session_state:
        st.warning("Please upload a dataset first.")
        time.sleep(3)
        return
    else:
        st.header("Select one or multiple methods for white box modelling")

        if 'preprocessed_data' not in st.session_state:
            st.session_state.preprocessed_data = st.session_state.original_data.copy()

        update_sybren_model_settings()
        if st.button("Run"):
            run_sybren()
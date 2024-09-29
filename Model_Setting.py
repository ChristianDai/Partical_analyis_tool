from pysr import PySRRegressor
import os
from gplearn.genetic import SymbolicRegressor
import numpy as np
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn import genetic
import pandas as pd
'''
*****************************************# Framework Setting #********************************************
'''
DAactive = 0
Pysractive = 0
GPactive = 0
CSTactive = 0
FSactive = 0
DNNactive = 0
Modfuncactive = 0
GNNactive = 0

'''
*****************************************# Chose Dataset and set Path #********************************************
'''
Nu = 1
process = 'Dataset_ModellingV01_fortest3'
file_Dataset='SP18.csv'

'''
*****************************************# Pysr Model #********************************************
'''
Pysrmodel1 = PySRRegressor(
    model_selection='accuracy',
    unary_operators=[],
    binary_operators=["/", "*", "-", '+'],
    populations=15,
    population_size=30,
    maxsize=10,
    progress=True,
    niterations=40,
    constraints=None,
    extra_sympy_mappings={'inv': lambda x: 1 / x, "myfunction1": lambda x: x ** (9)},
)
def get_Pysrmodel1_parameters():
    return {
    "model_selection": Pysrmodel1.model_selection,
    "unary_operators": Pysrmodel1.unary_operators,
    "binary_operators": Pysrmodel1.binary_operators,
    "population_size":Pysrmodel1.population_size,
    "populations": Pysrmodel1.populations,
    "maxsize": Pysrmodel1.maxsize,
    "progress": Pysrmodel1.progress,
    "niterations":Pysrmodel1.niterations,
    "nested_constraints":Pysrmodel1.nested_constraints
}

'''
*****************************************# GP1 Model #********************************************
'''
function_set = ['add', 'sub', 'mul', 'div']
gp_model1 = SymbolicRegressor(population_size=1000, function_set=function_set, init_method='half and half',
                              generations=5, stopping_criteria=0.1,
                              p_crossover=0.6, p_subtree_mutation=0.1,
                              p_hoist_mutation=0.05, p_point_mutation=0.2,
                              max_samples=0.9, verbose=1,
                              parsimony_coefficient=0.01, random_state=0,
                              )
def get_GPmodel1_parameters():
    return {
        "population_size":gp_model1.population_size,
        "function_set": gp_model1.function_set,
        "init_method": gp_model1.init_method,
        "generations": gp_model1.generations,
        "stopping_criteria": gp_model1.stopping_criteria,
        "p_crossover": gp_model1.p_crossover,
        "p_subtree_mutation": gp_model1.p_subtree_mutation,
        "p_hoist_mutation": gp_model1.p_hoist_mutation,
        "p_point_mutation": gp_model1.p_point_mutation,
        "max_samples": gp_model1.max_samples,
        "verbose": gp_model1.verbose,
        "parsimony_coefficient": gp_model1.parsimony_coefficient,
        "random_state": gp_model1.random_state
    }

'''
*****************************************# Pysr2 Model #********************************************
'''
Pysrmodel2=Pysrmodel1
def get_Pysrmodel2_parameters():
    return {
    "model_selection": Pysrmodel1.model_selection,
    "unary_operators": Pysrmodel1.unary_operators,
    "binary_operators": Pysrmodel1.binary_operators,
    "population_size":Pysrmodel1.population_size,
    "populations": Pysrmodel1.populations,
    "maxsize": Pysrmodel1.maxsize,
    "progress": Pysrmodel1.progress,
    "niterations":Pysrmodel1.niterations
}

'''
*****************************************# GP2 Model #********************************************
'''
function_set = ['add', 'sub', 'mul', 'div', 'neg', 'inv', 'log']
gp_model2=gp_model1
def get_GPmodel2_parameters():
    return {
        "population_size": gp_model2.population_size,
        "function_set": gp_model2.function_set,
        "init_method": gp_model2.init_method,
        "generations": gp_model2.generations,
        "stopping_criteria": gp_model2.stopping_criteria,
        "p_crossover": gp_model2.p_crossover,
        "p_subtree_mutation": gp_model2.p_subtree_mutation,
        "p_hoist_mutation": gp_model2.p_hoist_mutation,
        "p_point_mutation": gp_model2.p_point_mutation,
        "max_samples": gp_model2.max_samples,
        "verbose": gp_model2.verbose,
        "parsimony_coefficient": gp_model2.parsimony_coefficient,
        "random_state": gp_model2.random_state}

'''
*****************************************# GPMod Model #********************************************
'''
Mod_Config={
    'inv':False,
    'squared':False,
    'sqrt':False,
    'exp':False,
    'Log':False,
    'sin':False,
    'cos':False,
    'tan':False,
    'asin':False,
    'acos':False,
    'atan':False,
}

function_set = ['add', 'sub', 'mul', 'div', 'neg', 'inv', 'log']
gp_modelmod = SymbolicRegressor(population_size=1000, function_set=function_set, init_method='full',
                                generations=5, stopping_criteria=0.01,
                                p_crossover=0.6, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.2,
                                max_samples=0.9, verbose=1,
                                parsimony_coefficient=0.01, random_state=0,
                                )

GPmodelMod_parameter = {
    "population_size": gp_modelmod.population_size,
    "function_set": gp_modelmod.function_set,
    "init_method": gp_modelmod.init_method,
    "generations": gp_modelmod.generations,
    "stopping_criteria": gp_modelmod.stopping_criteria,
    "p_crossover": gp_modelmod.p_crossover,
    "p_subtree_mutation": gp_modelmod.p_subtree_mutation,
    "p_hoist_mutation": gp_modelmod.p_hoist_mutation,
    "p_point_mutation": gp_modelmod.p_point_mutation,
    "max_samples": gp_modelmod.max_samples,
    "verbose": gp_modelmod.verbose,
    "parsimony_coefficient": gp_modelmod.parsimony_coefficient,
    "random_state": gp_modelmod.random_state}

'''
*****************************************# PysrMod Model #********************************************
'''
PysrmodelMod = PySRRegressor(
    model_selection='accuracy',
    unary_operators=[],
    binary_operators=["/", "*", "-", '+'],
    populations=15,
    maxsize=40,
    progress=False,
    extra_sympy_mappings={'inv': lambda x: 1 / x, "myfunction1": lambda x: x ** (9)},
)

PysrmodelMod_parameter = {
    "model_selection": PysrmodelMod.model_selection,
    "unary_operators": PysrmodelMod.unary_operators,
    "binary_operators": PysrmodelMod.binary_operators,
    "populations": PysrmodelMod.populations,
    "maxsize": PysrmodelMod.maxsize,
    "progress": PysrmodelMod.progress,
}
'''
*****************************************# Fragment Selection Model1 #********************************************
'''
FS = genetic.SymbolicTransformer(
    population_size=1000,
    generations = 5,
    hall_of_fame=100,
    parsimony_coefficient=0.0001,
    function_set=['add', 'sub', 'mul', 'div'],
    tournament_size=10,
    init_depth=(2, 3),
    metric='pearson',
    const_range=(-5.0, 5.0),
    p_crossover=0.7,
    p_subtree_mutation=0.01,
    p_hoist_mutation=0.01,
    p_point_mutation=0.01,
    p_point_replace=0.27,
    max_samples=0.9,
    stopping_criteria=0.99999,
    verbose=1,
    random_state=0,
    n_jobs=-1,
)

def get_FS_params():
    return {
        "generations": FS.generations,
        "population_size": FS.population_size,
        "hall_of_fame": FS.hall_of_fame,
        "parsimony_coefficient": FS.parsimony_coefficient,
        "function_set": FS.function_set,
        "tournament_size": FS.tournament_size,
        "init_depth": FS.init_depth,
        "metric": FS.metric,
        "const_range": FS.const_range,
        "p_crossover": FS.p_crossover,
        "p_subtree_mutation": FS.p_subtree_mutation,
        "p_hoist_mutation": FS.p_hoist_mutation,
        "p_point_mutation": FS.p_point_mutation,
        "p_point_replace": FS.p_point_replace,
        "max_samples": FS.max_samples,
        "stopping_criteria": FS.stopping_criteria,
        "verbose": FS.verbose,
        "random_state": FS.random_state,
        "n_jobs": FS.n_jobs}

'''
*****************************************# Fragment Selection Model2 #********************************************
'''
FS2 = genetic.SymbolicTransformer(generations=2, population_size=2000,
                                  hall_of_fame=100, n_components=1,
                                  metric='pearson',
                                  parsimony_coefficient=0.001,
                                  max_samples=0.9, verbose=1,
                                  random_state=2)

FS2_params = {
    "generations": FS2.generations,
    "population_size": FS2.population_size,
    "hall_of_fame": FS2.hall_of_fame,
    "n_components": FS2.n_components,
    "metric": FS2.metric,
    "parsimony_coefficient": FS2.parsimony_coefficient,
    "max_samples": FS2.max_samples,
    "verbose": FS2.verbose,
    "random_state": FS2.random_state
}

'''
*****************************************# Fragment Selection Model3 #********************************************
'''
FS3 = genetic.SymbolicRegressor(population_size=1000,
                                generations=10, stopping_criteria=0.001,
                                const_range=(-10, 10),
                                p_crossover=0.7, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                max_samples=0.9, verbose=1,
                                parsimony_coefficient=0.1, random_state=0)

FS3_params = {
    "population_size": FS3.population_size,
    "generations": FS3.generations,
    "stopping_criteria": FS3.stopping_criteria,
    "const_range": FS3.const_range,
    "p_crossover": FS3.p_crossover,
    "p_subtree_mutation": FS3.p_subtree_mutation,
    "p_hoist_mutation": FS3.p_hoist_mutation,
    "p_point_mutation": FS3.p_point_mutation,
    "max_samples": FS3.max_samples,
    "verbose": FS3.verbose,
    "parsimony_coefficient": FS3.parsimony_coefficient,
    "random_state": FS3.random_state,
}


'''
*****************************************# NN Model1 #********************************************
'''
NN_Model_Manual = 0
NN_Model_Automatic = 1

NN_Config = {
    'one-step regression': False,
    'steps':1,
    'epoch': 100,
    'bs': 2048,
    # equation library
    'Plus': False,
    'Minus': False,
    'Multiply': False,
    'Division': False,
    'ab**n': False,
    'a**n*cosb': False,
    'an_plus_cosb': False,
    'an_expb': False,
    'a+b**n': False,
    'a-b**n': False,
    # value of n
    'Params_for_ab**n': [2, 3, 4, -2, -3, -4],
    'Params_for_a**n*cosb': [2, 3, 4, -2, -3, -4],
    'Params_for_an_plus_cosb': [2, 3, 4, -2, -3, -4],
    'Params_for_an_expb': [2, 3, 4, -2, -3, -4],
    'Params_for_a+b**n': [2, 3, 4, -2, -3, -4],
    'Params_for_a-b**n': [2, 3, 4, -2, -3, -4],
    'HPO':False,
}

'''
*****************************************# Graph Neutral Network Model #********************************************
'''
n_features = 7
extend_time = 1
num_batches = 1
GNNpysrmodel = PySRRegressor(
            model_selection='bic',
            unary_operators=[], # ["inv(x) = 1/x", "square(x) = x^2"]
            binary_operators=["/", "*", "-", '+'],
            populations=30,
            population_size=30,
            maxsize=15,
            progress=False,
            niterations=100,
            #select_k_features=4,
            extra_sympy_mappings={'inv': lambda x: 1 / x, "square": lambda x: x ** 2},
            denoise=True
        )
        
if __name__ == "__main__":
    from _program import RUN
    RUN()

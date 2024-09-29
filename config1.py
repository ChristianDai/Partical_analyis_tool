import numpy as np
from util import SyntaxTree as st
from util import EvalTools
from gplearn.functions import make_function

# gplearn
def _protected_div(x1, x2):
    """Closure of division for zero and negative arguments. From GPlearn"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)

def _protected_log(x1):
    """Closure of log for zero and negative arguments. From GP learn"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.0)

def _protected_inv(x1):
    """Closure of inverse for zero arguments. From GP learn"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x1) > 0.001, np.divide(1, x1), 1.0)

def _protected_sqrt(x1):
    """Closure of square root for negative arguments. From GPlearn"""
    return np.sqrt(np.abs(x1))

def square(x1):
    return np.square(x1)

def cube(x1):
    return np.power(x1, 3)

add = make_function(function=np.add, name='add', arity=2)
sub = make_function(function=np.subtract, name='sub', arity=2)
mul = make_function(function=np.multiply, name='mul', arity=2)
div = make_function(function=_protected_div, name='div', arity=2)
sqrt = make_function(function=_protected_sqrt, name='sqrt', arity=1)
log = make_function(function=_protected_log, name='log', arity=1)
abs = make_function(function=np.abs, name='abs', arity=1)
neg = make_function(function=np.negative, name='neg', arity=1)
inv = make_function(function=_protected_inv, name='inv', arity=1)
square = make_function(function=square, name='square', arity=1)
cube = make_function(function=cube, name='cube', arity=1)
sin = make_function(function=np.sin, name='sin', arity=1)
cos = make_function(function=np.cos, name='cos', arity=1)
tan = make_function(function=np.tan, name='tan', arity=1)
        
GPLEARN_DEFAULT_CONFIG = {
    'population_size': 500,
    'tournament_size': 20,
    'function_set': [add, sub, mul, div, sqrt, log, abs, neg, inv, sin, cos, tan],
    'init_method': 'full',
    'generations': 5,
    'stopping_criteria': 0.1,
    'p_crossover': 0.6,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.2,
    'max_samples': 0.9,
    'verbose': 1,
    'parsimony_coefficient': 0.01,
    'random_state': 0,
}

NNFUNCTIONS_DEFAULT_CONFIG = {
    'one-step regression': True,
    'steps':1,
    'epoch': 200,
    'bs': 2048,
    # equation library
    'Plus': True,
    'Minus': True,
    'Multiply': True,
    'Division': True,
    'ab**n': True,
    'a**n*cosb': True,
    'an_plus_cosb': True,
    'an_expb': True,
    'a+b**n': True,
    'a-b**n': True,
    # value of n
    'Params_for_ab**n': [2, 3, 4, -2, -3, -4],
    'Params_for_a**n*cosb': [2, 3, 4, -2, -3, -4],
    'Params_for_an_plus_cosb': [2, 3, 4, -2, -3, -4],
    'Params_for_an_expb': [2, 3, 4, -2, -3, -4],
    'Params_for_a+b**n': [2, 3, 4, -2, -3, -4],
    'Params_for_a-b**n': [2, 3, 4, -2, -3, -4],
    'HPO':False,
}

AUGMENTER_DEFAULT_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "gen_x_times": 5,
    "top_filter_quantile": 0.99,
    "bot_filter_quantile": 0.0001,
    "is_post_process": True,
    "adversarial_model_params": {
        "metrics": "rmse",
        "max_depth": 2,
        "max_bin": 100,
        "learning_rate": 0.001,
        "random_state": 42,
        "n_estimators": 500,
    },
    "pregeneration_frac": 2,
    "only_generated_data": False,
    "gen_params": {"batch_size": 50, "patience": 50, "epochs": 1000},
}

SIMULATED_ANNEALING_DEFAULT_CONFIG = {
    "MC_for_constant_optimization": True,
    "constants_minimal_value": -10,
    "constants_maximal_value": 10,
    "number_of_independent_variables": 8,
    "arity_one_functions": [
        "abs",
        "negative",
        "cos",
        "sin",
        "tan",
        "log",
        "sqrt",
        "square",
    ],
    "arity_two_functions": ["add", "sub", "mul", "div", "pow"],
    "cooling_strategy": "linear",
    "number_of_steps_SA": 80000,
    "Tmax": 0.1,
    "Tmin": 2e-6,
    "initial_acceptance_rate": 0.6,
    "find_tmax_for_acceptance_rate": True,
    "size_penalty_factor": 0.002,
    "move_selection_strategy": 1,
    "strategy3_size_limit": 15,
    "strategy1_change_size_probability": 0.8,
    "strategy1_change_increases_size_probability": 0.35,
    "optimized_constant_selection": False,
    "optimized_constant_iterations": 15,
}

#rewrite from below as dict
PYSR_DEFAULT_CONFIG = {
    "model_selection": "accuracy",
    "unary_operators": [],
    "binary_operators": ["/", "*", "-", "+"],
    "populations": 15,
    "population_size": 33,
    "maxsize": 15,
    "progress": False,
    "niterations": 40,
    "weight_optimize": 0.001,
    "extra_sympy_mappings": {
        "inv": lambda x: 1 / x,
        "myfunction1": lambda x: x ** 9,
    },
}

class SimulatedAnnealingConfiguration:
    """This class is uset to specify all the parameters and settings for the symbolic regressiion sumulated annealing algorithm
    To modifiy the functions that make up the search space, modify the methods in this class
    """

    def __init__(self) -> None:
        # self.constant_optimization_strategy = "MC" # "MC" for Markov Chain
        ## definition of the search space
        self.constants_minimal_value = -10
        self.constants_maximal_value = 10
        self.number_of_independent_variables = 8  ## if the simulated_annealing.py module of the modular correlation searcher is used, this will be determined automatically
        #                                           based on the data set. In that case this paremeter is overwritten.

        self.arity_one_functions = [
            self.abs,
            self.negative,
            self.cos,
            self.sin,
            self.tan,
            self.log,
            self.sqrt,
            self.square,
        ]  ## these functions are defined below and are the functions that can appear in the expression tree.
        self.arity_two_functions = [
            self.add,
            self.sub,
            self.mul,
            self.div,
            self.pow,
        ]  # It is possible to modify the functions and to add or remove functions.

        self.initial_state = st.SyntaxTree(
            st.Operation(const=1)
        )  ## an initial syntax tree representing an equation that describes the data set. In this case it is a constant.

        ## cooling schedule
        self.cooling_strategy = "linear"  # "linear" or "exponential"
        self.number_of_steps_SA = (
            80000  # number of iterations in the simulated anneailng algorithm.
        )
        self.Tmax = 0.1  # initial temperature
        self.Tmin = 2e-6  # final temperature
        self.initial_acceptance_rate = 0.6  # initial acceptance rate this is used to calculate a suitable t_max for linear cooling
        self.find_tmax_for_acceptance_rate = True  # if True and and linear cooling is used, Tmax will be overwirtten with a suitable temperature based on the initial acceptance rate

        ## energy settings
        self.error_metric = (
            EvalTools.MAPE()
        )  ## Error metric out of the EvalTools file that shall be used to evaluate the model performance.
        self.size_penalty_factor = 0.002  ### used to prevent big trees

        # moves settings
        self.move_selection_strategy = (
            1  ## 1: make_move_prefer_small 2: equal probabilities     keep 1
        )
        self.strategy3_size_limit = 15  ## redundant
        self.strategy1_change_size_probability = 0.80
        self.strategy1_change_increases_size_probability = 0.35

        self.optimized_constant_selection = False  ## if true, a hill climbing algorithm is used after every move to optimize one randomly selected constant node in the expression tree
        self.optimized_constant_iterations = (
            15  ## number of iterations in the hill climbing algorithm
        )
    
    @classmethod
    def from_dict(cls, config_dict):
        #class method to create a SimulatedAnnealingConfiguration object from a dictionary
        ARITY_ONE_FUNCTIONS = {
            "negative": st.Function("negative", 1, lambda x: -x),
            "square": st.Function("sqr", 1, np.square),
            "sin": st.Function("sin", 1, np.sin),
            "cos": st.Function("cos", 1, np.cos),
            "tan": st.Function("tan", 1, np.tan),
            "abs": st.Function("abs", 1, np.abs),
            "sqrt": st.Function("sqrt", 1, lambda x: np.sqrt(np.abs(x))),
            "log": st.Function("log", 1, cls._protected_log),
        }
        ARITY_TWO_FUNCTIONS = {
            "add": st.Function("add", 2, np.add),
            "sub": st.Function("sub", 2, np.subtract),
            "mul": st.Function("mul", 2, np.multiply),
            "div": st.Function("div", 2, cls.divide_with_threshhold),
            "pow": st.Function("pow", 2, cls._protected_power),
        }
        config = cls()
        for key, value in config_dict.items():
            if key in ["arity_one_functions", "arity_two_functions"]:
                func_dict = ARITY_ONE_FUNCTIONS if key == "arity_one_functions" else ARITY_TWO_FUNCTIONS
                functions = [func_dict[func_name] for func_name in value]
                setattr(config, key, functions)
            else:
                setattr(config, key, value)
        """
        initial_state always = st.SyntaxTree(st.Operation(const=1)) and error_metric always = EvalTools.MAPE()
        they are not included in the config_dict
        """
        setattr(config, "initial_state", st.SyntaxTree(st.Operation(const=1)))
        setattr(config, "error_metric", EvalTools.MAPE())
        return config



    def get_settings_string(self):
        """returns a string with the current settings"""
        settings = []

        # cooling schedule settings
        settings.append(f"cooling_strategy={self.cooling_strategy}")
        settings.append(f"number_of_steps_SA={self.number_of_steps_SA}")
        settings.append(f"Tmax={self.Tmax}")
        settings.append(f"Tmin={self.Tmin}")

        # search space settings
        settings.append(f"constants_minimal_value={self.constants_minimal_value}")
        settings.append(f"constants_maximal_value={self.constants_maximal_value}")
        settings.append(
            f"number_of_independent_variables={self.number_of_independent_variables}"
        )

        # energy settings
        settings.append(f"error_metric={self.error_metric}")
        settings.append(f"size_penalty_factor={self.size_penalty_factor}")

        # move settings
        settings.append(f"move_selection_strategy={self.move_selection_strategy}")
        settings.append(f"strategy3_size_limit={self.strategy3_size_limit}")
        settings.append(
            f"strategy1_change_size_probability={self.strategy1_change_size_probability}"
        )
        settings.append(
            f"strategy1_change_increases_size_probability={self.strategy1_change_increases_size_probability}"
        )
        settings.append(
            f"optimized_constant_selection={self.optimized_constant_selection}"
        )
        settings.append(
            f"optimized_constant_iterations={self.optimized_constant_iterations}\n"
        )

        return "\n".join(settings)

    ### define the arity one functions. Some functions differ form the mathematical definition to be able to handle exceptions like division by zero, so the expression tree can always be evaluated. However these modifications must be taken into account when interpreting the results.:

    def negative(x: float):
        return -x

    negative = st.Function("negative", 1, negative)
    square = st.Function("sqr", 1, np.square)
    sin = st.Function("sin", 1, np.sin)
    cos = st.Function("cos", 1, np.cos)
    tan = st.Function("tan", 1, np.tan)
    abs = st.Function("abs", 1, np.abs)

    ##define arity two functions:

    def add(x: float, y: float):
        return x + y

    add = st.Function("add", 2, add)

    def sub(x: float, y: float):
        return x - y

    sub = st.Function("sub", 2, sub)

    def mul(x: float, y: float):
        return x * y

    mul = st.Function("mul", 2, mul)

    @staticmethod
    def divide_with_threshhold(x: float, y: float, min_divisor: float = 0.01):
        """see protected division from https://github.com/trevorstephens/gplearn/blob/main/gplearn/functions.py

        every devisor bigger than min_devisor will be divided normally. Returns 1 for every other case

        mabe use y_max aus dataset als mögliches output limit threshhold
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(np.abs(y) > min_divisor, np.divide(x, y), 1)
            return a

    div = st.Function("div", 2, divide_with_threshhold)

    def _protected_sqrt(x1):
        """Closure of square root for negative arguments. From GPlearn"""
        return np.sqrt(np.abs(x1))

    sqrt = st.Function("sqrt", 1, _protected_sqrt)

    @staticmethod
    def _protected_log(x1):
        """Closure of log for zero and negative arguments. From GP learn"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.0)

    log = st.Function("log", 1, _protected_log)

    @staticmethod
    @np.vectorize
    def _protected_power(x1: float, x2: float, min_divisor: float = 0.01):
        """raises  abs(x1) to the power x2. If x2 is negative and abs(x1) is smaler than min_divisior returns 1 to avoid division by zero limits output to 1e10. Limits the exponent to +-4"""
        try:
            X1 = float(x1)
            ## limit the exponent to 4
            X2 = float(x2)
            if x2 > 4:
                X2 = float(4)
            if x2 < -4:
                X2 = float(-4)
            if X2 > 0 or abs(X1) > min_divisor:
                result = np.float64(abs(X1) ** X2)
            else:
                result = np.float64(1)
            return min(1e10, result)
        except OverflowError:
            return float(1)

    pow = st.Function("pow", 2, _protected_power)


class GeneticProgrammingConfiguration:
    """Configuration settings for Genetic Programming based Symbolic Regression."""

    def __init__(self):
        self.population_size = 1000
        self.generations = 20
        self.tournament_size = 20
        self.GP_FUNCTIONS = [
            st.Function("add", 2, np.add),
            st.Function("sub", 2, np.subtract),
            st.Function("mul", 2, np.multiply),
            st.Function("div", 2, self._protected_div),
            st.Function("sqrt", 1, np.sqrt),
            st.Function("log", 1, self._protected_log),
            st.Function("abs", 1, np.abs),
            st.Function("neg", 1, np.negative),
            st.Function("inv", 1, self._protected_inv),
        ]
        self.function_set = [func.get_name() for func in self.GP_FUNCTIONS]
        self.const_range = (-10, 10)
        self.init_depth = (2, 6)
        self.init_method = "half and half"
        self.mutation_rate = 0.1
        self.crossover_rate = 0.9
        self.parsimony_coefficient = 0.001
        self.max_samples = 1.0
        self.verbose = 1
        # Add any additional parameters that gplearn's SymbolicRegressor accepts

    def get_settings_string(self):
        """Returns a string representation of the current settings for logging or debugging purposes."""
        settings = [
            f"Population Size: {self.population_size}",
            f"Generations: {self.generations}",
            f"Tournament Size: {self.tournament_size}",
            f"Function Set: {', '.join(self.function_set)}",
            f"Constant Range: {self.const_range}",
            f"Initial Depth: {self.init_depth}",
            f"Initialization Method: {self.init_method}",
            f"Mutation Rate: {self.mutation_rate}",
            f"Crossover Rate: {self.crossover_rate}",
            f"Parsimony Coefficient: {self.parsimony_coefficient}",
            f"Max Samples: {self.max_samples}",
            f"Verbose: {self.verbose}",
        ]
        return "\n".join(settings)

    def _protected_div(self, x1, x2):
        """Closure of division for zero and negative arguments. From GPlearn"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)

    def _protected_log(self, x1):
        """Closure of log for zero and negative arguments. From GP learn"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.0)

    def _protected_inv(self, x1):
        """Closure of inverse for zero arguments. From GP learn"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(x1) > 0.001, np.divide(1, x1), 1.0)

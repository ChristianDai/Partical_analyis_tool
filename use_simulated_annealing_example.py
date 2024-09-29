"""
This is an example of how a complete ModularSearcher would be setup and then executed
"""
from util import EvalTools as et
from ModularCorrelationSearch import ConfigManager
from ModularCorrelationSearch import ModularSearcher
import pandas as pd
import numpy as np
from util.MainLogger import Logger
from util import SyntaxTree as st
from util import EvalTools
import os
import matplotlib.pyplot as plt

# define the settings for the simulated annealing module in the SimulatedAnnealingConfiguration class and create an object
class SimulatedAnnealingConfiguration:

    """This class is uset to specify all the parameters and settings for the symbolic regressiion sumulated annealing algorithm
    To modifiy the functions that make up the search space, modify the methods in this class"""
    def __init__(self) -> None:
            
            ## definition of the search space
            self.constants_minimal_value= -10
            self.constants_maximal_value = 10
            self.number_of_independent_variables = 2   ## if the simulated_annealing.py module of the modular correlation searcher is used, this will be determined automatically
            #                                           based on the data set. In that case this paremeter is overwritten.

            self.arity_one_functions = [self.abs,self.negative,self.cos,self.sin,self.tan,self.log,self.sqrt,self.square]  ## these functions are defined below and are the functions that can appear in the expression tree.
            self.arity_two_functions = [self.add,self.sub,self.mul,self.div,self.pow]                                      #It is possible to modify the functions and to add or remove functions.
            
            self.initial_state = st.SyntaxTree(st.Operation(const=1))  ## an initial syntax tree representing an equation that describes the data set. In this case it is a constant.

            ## cooling schedule
            self.cooling_strategy = "linear"    # "linear" or "exponential"
            self.number_of_steps_SA = 80000   # number of iterations in the simulated anneailng algorithm.
            self.Tmax = 0.1                   # initial temperature
            self.Tmin = 2e-6                  # final temperature
            self.initial_acceptance_rate=0.6  # initial acceptance rate this is used to calculate a suitable t_max for linear cooling
            self.find_tmax_for_acceptance_rate =True   # if True and and linear cooling is used, Tmax will be overwirtten with a suitable temperature based on the initial acceptance rate

            ## energy settings
            self.error_metric = EvalTools.MAPE()    ## Error metric out of the EvalTools file that shall be used to evaluate the model performance. 
            self.size_penalty_factor = 0.002         ### used to prevent big trees

            # moves settings
            self.move_selection_strategy = 1       ## 1: make_move_prefer_small 2: equal probabilities     keep 1
            self.strategy3_size_limit = 15          ## redundant
            self.strategy1_change_size_probability = 0.80       
            self.strategy1_change_increases_size_probability = 0.35

            self.optimized_constant_selection = False       ## if true, a hill climbing algorithm is used after every move to optimize one randomly selected constant node in the expression tree
            self.optimized_constant_iterations = 15   ## number of iterations in the hill climbing algorithm
            
            
            
            
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
        settings.append(f"number_of_independent_variables={self.number_of_independent_variables}")
        
        # energy settings
        settings.append(f"error_metric={self.error_metric}")
        settings.append(f"size_penalty_factor={self.size_penalty_factor}")
        
        # move settings
        settings.append(f"move_selection_strategy={self.move_selection_strategy}")
        settings.append(f"strategy3_size_limit={self.strategy3_size_limit}")
        settings.append(f"strategy1_change_size_probability={self.strategy1_change_size_probability}")
        settings.append(f"strategy1_change_increases_size_probability={self.strategy1_change_increases_size_probability}")
        settings.append(f"optimized_constant_selection={self.optimized_constant_selection}")
        settings.append(f"optimized_constant_iterations={self.optimized_constant_iterations}\n")
        
        return "\n".join(settings)


    ### define the arity one functions. Some functions differ form the mathematical definition to be able to handle exceptions like division by zero, so the expression tree can always be evaluated. However these modifications must be taken into account when interpreting the results.:

    def negative(x: float):
        return -x
    negative = st.Function("negative",1,negative)

    square = st.Function("sqr",1,np.square)
    sin = st.Function("sin",1,np.sin)
    cos = st.Function("cos",1,np.cos)
    tan = st.Function("tan",1,np.tan)
    abs = st.Function("abs",1,np.abs)
    
    ##define arity two functions:

    def add(x: float, y: float):
        return x+y
    add = st.Function("add",2,add)

    def sub(x: float, y:float):
        return x-y
    sub = st.Function("sub",2,sub)

    def mul(x: float, y:float):
        return x*y
    mul = st.Function("mul",2,mul)


    

    def divide_with_threshhold(x: float, y:float, min_divisor: float = 0.01):
        """see protected division from https://github.com/trevorstephens/gplearn/blob/main/gplearn/functions.py

        every devisor bigger than min_devisor will be divided normally. Returns 1 for every other case
        
        mabe use y_max aus dataset als mögliches output limit threshhold
        """
        with np.errstate(divide="ignore",invalid="ignore"):
            a = np.where(np.abs(y)>min_divisor,np.divide(x,y),1)
            return a    
    div = st.Function("div",2,divide_with_threshhold)

    def _protected_sqrt(x1):
        """Closure of square root for negative arguments. From GPlearn"""
        return np.sqrt(np.abs(x1))
    sqrt = st.Function("sqrt",1,_protected_sqrt)

    def _protected_log(x1):
        """Closure of log for zero and negative arguments. From GP learn"""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)   
    log = st.Function("log",1, _protected_log)

    @np.vectorize
    def _protected_power(x1: float,x2: float,min_divisor:float = 0.01):
        """raises  abs(x1) to the power x2. If x2 is negative and abs(x1) is smaler than min_divisior returns 1 to avoid division by zero limits output to 1e10. Limits the exponent to +-4"""
        try:
            X1 = float(x1)
            ## limit the exponent to 4
            X2=float(x2)
            if x2>4:
                X2 = float(4)
            if x2<-4:
                X2 = float(-4)
            if X2>0 or abs(X1)>min_divisor:
                result = np.float64(abs(X1)**X2)
            else:
                result = np.float64(1)   
            return min(1e10,result)     
        except OverflowError:
            return float(1)
    pow = st.Function("pow",2,_protected_power)


simulatedannealingconfiguration=SimulatedAnnealingConfiguration()



# Instantiate a ConfigManager-Object
cfm = ConfigManager()

# Specify the desired Modules to be used
cfm.set_modules(["Test", "Module123", "Average","Simulated_annealing"])

# Specify the target column
cfm.set_target_column(1)

# Setup custom Parameters if needed. General Parameters such as runtime could also be setup here, but since they aren't
# used in any of the test-modules they were left out
"""cfm.set_custom_parameters({"Test_value": 2,
                           "Test_multip": 4,
                           "Average_num_rows": 100,
                           "Simulated_annealing_configuration":simulatedannealingconfiguration,
                           "Genetic_programming_configuration":genetticprogrammingconfiguration})"""

cfm.set_custom_parameters({"Test_value": 2,
                           "Test_multip": 4,
                           "Average_num_rows": 100,
                           "Simulated_annealing_configuration":simulatedannealingconfiguration})

# This would set the runtime of a given module to a maximum of 3000 seconds
# The difference between declaring the runtime in the custom parameters and via set_max_runtime() is that the runtime in
# the custom parameters acts as a suggestion while the runtime in set_max_runtime() is actually enforced and the
# execution of a module would be halted after it is reached
# NOT IMPLEMENTED YET
# cfm.set_max_runtime(3000)

# Add custom metrics via EvalTools classes
# This is also only a suggestion to every Module and doesn't strictly HAVE to be implemented
# A Module would simply have to call the evaluate-method on every custom EvalTool to get a metric from it though
cfm.add_eval_tool(name="MeanAccuracy", eval_tool=et.MeanAccuracy(), cutoff=0.8)
cfm.add_eval_tool(name="MAE", eval_tool=et.MAE(), cutoff=0.5)


# Instantiate the ModularSearcher-object with the ConfigManager-object containing all the parameters and the dataset
# that is to be searched

#read a data set from a csv file
cwd=os.getcwd()
print("cwd=",cwd)

filename = 'datasets/Virtual_Dataset3.csv'
# Read the CSV file into a NumPy array
dataset = pd.read_csv(filename, delimiter=',')

mds = ModularSearcher(config=cfm, dataset=dataset)

# This part executes every module and stores the results in "res"
res = mds.run()

# Results can then be accessed by indexing res with the module-name
Logger.write("Main", "------------------")
Logger.write("Main", "Content Test:" + str(res["Test"]))
Logger.write("Main", "LISP Test:" + str([str(x) for x in res["Test"]]))
Logger.write("Main", "Content Module123:" + str(res["Module123"]))
Logger.write("Main", "LISP Module123:" + str([str(x) for x in res["Module123"]]))
Logger.write("Main", "Content Average:" + str(res["Average"]))
Logger.write("Main", "LISP Average:" + str([str(x) for x in res["Average"]]))
Logger.write("Main", "Content Simulated_annealing:" + str(res["Simulated_annealing"]))
Logger.write("Main", "LISP Simulated_annealing:" + str([str(x) for x in res["Simulated_annealing"]]))

# They can also be evaluated
eval_tool = et.R2()
Logger.write("Main","R2 values:")
acc_test = eval_tool.evaluate(res["Test"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "MAE_Test: " + str(acc_test))
acc_module123 = eval_tool.evaluate(res["Module123"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "MAE_Module123: " + str(acc_module123))
acc_average = eval_tool.evaluate(res["Average"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "MAE_Average: " + str(acc_average))
acc_Simulated_annealing = eval_tool.evaluate(res["Simulated_annealing"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "Simulated_annealing: " + str(acc_Simulated_annealing))

eval_tool = et.MAPE()
Logger.write("Main","MAPE values:")
acc_test = eval_tool.evaluate(res["Test"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "MAE_Test: " + str(acc_test))
acc_module123 = eval_tool.evaluate(res["Module123"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "MAE_Module123: " + str(acc_module123))
acc_average = eval_tool.evaluate(res["Average"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "MAE_Average: " + str(acc_average))
acc_Simulated_annealing = eval_tool.evaluate(res["Simulated_annealing"][0], mds.get_input_columns(), mds.get_target_column())
Logger.write("Main", "Simulated_annealing: " + str(acc_Simulated_annealing))


# plot the data and the equation found by simulated annealing

# x1numbers = np.zeros(100)  ##x1 data for plotting. not used, but this way I could use the singlevalues_input_eval_2d to generate the y values
#predictedy=res["Simulated_annealing"][0].singlevalues_input_eval_multi(dataset[:,:dataset.shape[1]-1])
# print(predictedy)
#x0numbers = np.linspace(1, np.shape(dataset)[0]+1,np.shape(dataset)[0]) ##x0 data for plotting
#plt.plot(x0numbers,predictedy,color='red',label='found equation')
#plt.legend()
#plt.show()
"""
This is the core-part of the program. Here, two classes are defined. First of the ConfigManager and then the ModularSearcher.
The ConfigManager is basically just a collection of relevant variables for every module in general and then of specific
variables for every unique module. Also, there are some methods defined to interface with the given variables.
The ModularSearcher then uses the configured ConfigManager-object and imports the specified modules (given in _module_names).
Then it will run every Module and store its result.
"""
from typing import *

import numpy as np
import pandas as pd
import os
import inspect
import importlib

from preprocessing.preprocessing import PreprocessingModule
from util import EvalTools
from util.MainLogger import Logger
from util.DataContainerTemplate import DataContainer

import time


class ConfigManager:
    def __init__(self):
        self._max_runtime = -1  # Runtime, given in seconds
        self._eval_tools = {}
        self._custom_parameters = {}
        self._target_column = None
        self._module_names = []
        self.preprocessing_modules = []
        self.preprocessing_default_configs = {
            'normalizer': {'method': 'min-max'},
            'feature_scaler': {'scale_factor': 1.0} # this are examples, you can set them in separate file and import them
        }

    def set_max_runtime(self, runtime: int) -> None:
        """
        Set the maximum runtime of each module. Every module should adhere to the given runtime,
        but its ultimately up to the module to implement this
        :param runtime: The desired Runtime, given in seconds
        :return: None
        """
        self._max_runtime = runtime

    def add_eval_tool(self, name: str, eval_tool: EvalTools.EvalTool, cutoff: float = None) -> None:
        """
        Add a precision metric that the module should use.
        Every module should be able to calculate custom precision metrics,
        but its ultimately up to the module to implement it
        :param name: The name of the metric
        :param eval_tool: The EvalTool that should be used for Evaluation
        :param cutoff: A desired cutoff. If this value or a better one is reached, the module can stop its execution
        :return: None
        """
        self._eval_tools[name] = {
            "tool": eval_tool,
            "cutoff": cutoff
        }

    def get_eval_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the Dictionary of specified Eval-Tools that the user want's to use
        :return: Dictionary of all Eval-Tools to be used
        """
        return self._eval_tools

    def set_custom_parameters(self, parameters: {}) -> None:
        """
        Set the custom parameters the selected modules need in order to run. Structure of the Dictionary is as follows:
        {
        "parameter_name": value,
        ...
        }
        :param parameters: The Dictionary with all the required parameters
        :return: None
        """
        self._custom_parameters = parameters

    def get_custom_parameters(self) -> Dict[str, Any]:
        """
        Return the custom parameters currently set for this Config-Manager-Object
        :return: Custom-Parameters Dictionary
        """
        return self._custom_parameters

    def set_target_column(self, column):
        """
        Set the target column by name or index.
        :param column: Name or index of the column to be used as the target.
        """
        self._target_column = column

    def get_target_column(self) -> str:
        """
        Returns the name of the configured target column.
        """
        return self._target_column
    
    def add_preprocessing_module(self, step_name: str, params=None) -> None:
        """
        Add a preprocessing step with parameters.
        :param step_name: The name of the preprocessing step.
        :param params: Dictionary of parameters for the step. If None, default parameters are used.
        """
        if params is None:
            params = self.default_configs.get(step_name, {})
        self.preprocessing_modules.append({'name': step_name, 'params': params})

    def get_preprocessing_modules(self) -> List[Dict[str, Any]]:
        """Return a list of preprocessing steps with their configurations."""
        return self.preprocessing_modules

    def remove_preprocessing_step(self, step_name: str) -> None:
        self.preprocessing_steps = [step for step in self.preprocessing_steps if step['name'] != step_name]

    def set_preprocessing_params(self, step_name: str, **params) -> None:
        for step in self.preprocessing_steps:
            if step['name'] == step_name:
                step['params'].update(params)
                break

    def set_modules(self, module_names: List[str]) -> None:
        """
        Set the modules that are to be executed by their name.
        (A Module with the Python-script "ModuleABC.py" is named "ModuleABC")
        :param module_names: A List of all module-names that are to be executed
        :return: None
        """
        self._module_names = module_names

    def get_modules(self) -> List[str]:
        """
        Returns the list of all module-names that are currently selected
        :return: List of Module-Names
        """
        return self._module_names


class ModularSearcher:
    def __init__(self, config: ConfigManager, dataset: pd.DataFrame, has_header: bool = True):
        """
        The ModularSearcher needs a ConfigManager object and a dataframe it's supposed to run over in order to work
        :param config: The >configured< ConfigManager object
        :param dataset: The dataset that is to be searched
        """
        self._config = config
        if not has_header:
            dataset.columns = ['x_' + str(i+1) for i in range(dataset.shape[1])]
        self._dataset = dataset
        self._warn_time = 1000.0  # Runtime of a Module in seconds after which a Warning should be issued
  
        self._target_column_name = self._determine_target_column(config.get_target_column(), has_header)
        if self._target_column_name not in self._dataset.columns:
            raise ValueError(f"Target column '{self._target_column_name}' not found in the dataset")

        self._input_column_names = [col for col in self._dataset.columns if col != self._target_column_name]

    def _determine_target_column(self, target_column, has_header):
        if isinstance(target_column, int):  # if target column is specified by index
            if target_column < 0 or target_column >= self._dataset.shape[1]:
                raise ValueError("Target column index out of range")
            return self._dataset.columns[target_column]
        elif isinstance(target_column, str) and not has_header:
            raise ValueError("Target column names cannot be used without header")
        return target_column
    
    def get_target_column(self) -> np.array:
        return self._dataset[self._target_column_name].to_numpy()

    def get_input_columns(self) -> np.array:
        return self._dataset[self._input_column_names].to_numpy()
    
    def get_target_column_name(self) -> str:
        return self._target_column_name
    
    def get_input_column_names(self) -> List[str]:
        return self._input_column_names

    def set_warn_time(self, time_s: float) -> None:
        """
        Set the delta-time for a Module at which a Warning should be issued
        :param time_s: Time in seconds
        :return: None
        """
        self._warn_time = time_s

    def update_data_structure(self) -> None:
        """
        Update self._input_columns based on the current self.dataset
        Reqiured if the dataset was changed after the initialization of the ModularSearcher
        :return: None"""
        self._input_column_names = [col for col in self._dataset.columns if col != self._target_column_name]

    @staticmethod
    def load_preprocessing_module(module_name: str, module_config):
        """Automatically find and instantiate the primary preprocessing class in the module."""
        module_path = f"preprocessing.{module_name.lower()}" 
        try:
            module = importlib.import_module(module_path)
            print(f"Module loaded: {module}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                print(f"Inspecting {name}: is a subclass of PreprocessingModule? {issubclass(obj, PreprocessingModule)}")
                if issubclass(obj, PreprocessingModule):
                    instance = obj(module_config) 
                    print(f"Instance of {name} created.")
                    return instance
            raise ImportError(f"No valid preprocessing class found in module {module_name}.")
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
            return None

    def run_preprocessing(self):
        """Applies preprocessing modules configured in ConfigManager to the dataset."""
        for module_info in self._config.get_preprocessing_modules():
            module_instance = self.load_preprocessing_module(module_info['name'], module_config=module_info['params'])
            if module_instance is None:
                continue
            self._dataset = module_instance.fit_transform(self._dataset)
            self.update_data_structure()
            Logger.write("ModularSearcher-Preprocessing", f"Preprocessing step {module_info['name']} completed.")
            Logger.write("ModularSearcher-Preprocessing", f"after preprocessing first 5 rows look so: {self._dataset.head()}")

    @staticmethod
    def _import_modules(module_names: List[str]) -> None:
        """
        Import desired Modules by finding the corresponding script-file to the name
        -> Script-name: VeryFunctionalModule.py => Module-name: "VeryFunctionalModule"
        :param module_names: List ([]) of names of the Modules to be imported
        :return: None
        """
        Logger.write("ModularSearcher", "Importing Modules...")
        modules = {}
        # Iterate trough current directory
        for filename in os.listdir("Modules"):
            if filename.endswith(".py"):
                modname = filename[:-3]
                if modname in module_names:
                    # If a fitting module is found that is also in the module_list, import it
                    modules[modname] = importlib.import_module("Modules." + modname)
        # Finally, update the globals so that the imports are not only in this function-scope
        globals().update(modules)
        Logger.write("ModularSearcher", "Finished importing modules!")

    def run(self) -> Dict[str, List[DataContainer]]:
        """  
        Runs the ModularSearcher
        :return: Results in form of a Dictionary: {"module_name": [SyntaxTree]}
        """
        self.run_preprocessing()  # Run preprocessing before main processing
        # Get all Modules that are to be used
        modules = self._config.get_modules()
        Logger.write("ModularSearcher", "Starting execution for modules:" + str(modules))

        # Import every Module to be used
        self._import_modules(modules)

        # Main Loop for executing every Module and storing its result
        # This could maybe be done with multithreading later on, but since every module could also use multithreading
        # it might be easier to just call them one after another
        results = {}
        for module in modules:
            t1 = time.perf_counter()
            Logger.write("ModularSearcher", "Executing module '" + module + "'...")
            temp_res = globals()[module].run(self._config, self.get_input_columns(), self.get_target_column())
            # Check if a List was returned
            assert isinstance(temp_res, List)
            # Check if every entry in the List is a DataContainer, if not evaluation later on will be impossible
            assert all(isinstance(entry, DataContainer) for entry in temp_res)
            results[module] = temp_res
            dt_s = (time.perf_counter() - t1)
            if dt_s > self._warn_time:
                Logger.warn("ModularSearcher", "Module took longer than " + str(self._warn_time) + " seconds to run! ("
                            + str(dt_s) + "s)")
            else:
                Logger.write("ModularSearcher", "Module took " + str(dt_s) + " seconds to run")

        # Filter results and further evaluation
        # Not implemented for this proof of concept
        Logger.write("ModularSearcher", "All modules ran! Returning results!")
        return results

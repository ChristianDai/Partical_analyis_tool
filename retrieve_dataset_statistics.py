

"""This program is used to extract metadata like the number of datapoints, the number of columns, the magnitude of the biggest and smalles value etc. from a given data set.
We assume that the last column in the data set contains the dependent variable (the variable the AI models are supposed to predict) and has numeric values
the header line shall contain the variable name and the unit in [] brackets for each column. """


"""

descriptive values:

IMPLEMENTED:
number_of_data_points
number_of_columns
biggest_dependent_value
smallest_dependent_value
smallest_absolute_dependent_value
biggest_independent_value
smallest_independent_value
smallest_absolute_independent_value

dependent_mean
dependent_standard_deviation
dependent_media = Q2
dependent Q1
dependent Q3 (quartile)
dependent Geometric Mean
dependent Geometric Mean deviation
dependent number of outliers(IQR)/number of data points dependent variable
dependnt Skewness: Skewness measures the asymmetry of the data distribution
dependnt Kurtosis: Kurtosis measures the "tailedness" of the distribution.

MORE IDEAS:

for one variable? mean over all columns? other ideas on how to handle variable number of independent varaibles in data sets
independent_mean
independent_standard_deviation
independent_media = Q2
independent Q1     quartile
independent Q3 
independend_mode

Z score for Outliers

mean over all input columns:  number_of_outliers/number_of_data points independent variabl

dependent unit: seconds power
dependent unit: kg power


coefficient for how similar dependent varaible is to normal distribution for example Shapiro-Wilk Test,Anderson-Darling Test


harmonic mean not useful because it cannot handle negative values or zero
Measures of Spread for Categorical Data: For categorical data, you can use measures like Gini index, Shannon entropy, or the number of unique categories to assess data diversity
Heteroscedasticity: This measures the change in variability of the data as the values increase and can be relevant in time series and regression analysis.
Coefficient of Variation (CV)

maybe Bool: time series data?


"""


import pandas as pd
import os
import typing
import numpy as np
import scipy.stats as stats
import re

def get_dependent_unit_string(data_set1: pd.DataFrame, target_column: str):
    # unit is marked by (unit) or [unit] brackets
    patterns = [r'\((.*?)\)', r'\[(.*?)\]']
    header = data_set1.columns
    target_column_header = target_column
    print(header)
    print(target_column_header)
    extracted_text = []

    # find all text parts enclosed in one of the patterns
    for pattern in patterns:
        matches = re.findall(pattern, target_column_header)
        extracted_text.extend(matches)
    
    # if the target column header contains no () or [] we assume there is no unit
    if extracted_text == []:
        extracted_text.extend('1')
    return extracted_text[0]
        

def print_all_unit_strings(data_set1: pd.DataFrame):
    # unit is marked by (unit) or [unit] brackets
    patterns = [r'\((.*?)\)', r'\[(.*?)\]']
    header = data_set1.columns
    #print(header)
    for column_header in header:
        #print(column_header)
        extracted_text = []

        for pattern in patterns:
            matches = re.findall(pattern, column_header)
            extracted_text.extend(matches)
        if extracted_text == []:
            extracted_text.extend('1')
        print(extracted_text[0])

def determine_number_of_data_points(data_frame: pd.DataFrame):
    return data_frame.shape[0]

def determine_number_of_columns(data_frame:pd.DataFrame):
    return data_frame.shape[1]

def determine_biggest_dependent_value(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].max()

def determine_smallest_dependent_value(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].min()

def determine_smallest_absolute_dependent_value(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].abs().min()

def determine_biggest_independent_value(data_frame: pd.DataFrame, target_column: str):
    numeric_df = data_frame.drop(columns=[target_column])
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    return numeric_df.max().max()

def determine_smallest_independent_value(data_frame: pd.DataFrame, target_column: str):
    numeric_df = data_frame.drop(columns=[target_column])
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    return numeric_df.min().min()

def determine_smallest_absolute_independent_value(data_frame: pd.DataFrame, target_column: str):
    numeric_df = data_frame.drop(columns=[target_column])
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.abs()
    return numeric_df.min().min()

def determine_mean_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].mean()

def determine_standard_deviation_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].std()

def determine_quartile_1_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].quantile(q=0.25)

def determine_median_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].median()

def determine_quartile_3_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    return data_frame[target_column].quantile(q=0.75)

def determine_geometric_mean_abs_values_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    "uses absolute values of the dependent column"
    data_array = data_frame[target_column].abs().to_numpy()
    return stats.gmean(data_array)

def determine_geometric_std_deviation_abs_values_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    "uses absolute values of the dependent column"
    data_array = data_frame[target_column].abs().to_numpy()
    return stats.gstd(data_array)


def determine_proportion_of_IQR_method_outliers_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    "interquartile range method"
    q1 = data_frame[target_column].quantile(q=0.25)
    q3 = data_frame[target_column].quantile(q=0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data_frame[target_column].to_numpy()
    number_of_outliers = sum((datapoint < lower_bound or datapoint > upper_bound) for datapoint in data)
    print('outliers ', number_of_outliers)
    number_of_data_points = np.shape(data)[0]
    print('datapoints', number_of_data_points)
    return number_of_outliers / number_of_data_points
    

def determine_skewness_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    "measure for asymmetry in the distribution. 0 for normal distribution >0 means the distribution is skewed to the left so The majority of the data points are on the left side of the distribution, and there are a few large values on the right side."
    data_array = data_frame[target_column].to_numpy()
    return stats.skew(data_array)

def determine_kurtosis_dependent_variable(data_frame: pd.DataFrame, target_column: str):
    "measures tailedness of distribution (peak of flat) normal distribution has value 3, values > 3 indicate heavy tails-->more extreme values"
    data_array = data_frame[target_column].to_numpy()
    return stats.kurtosis(data_array)


def determine_dataset_statistics(data_frame: pd.DataFrame, target_column: str):
    return {
        "Number of Data Points": determine_number_of_data_points(data_frame),
        "Number of Columns": determine_number_of_columns(data_frame),
        "Biggest Dependent Value": determine_biggest_dependent_value(data_frame, target_column),
        "Smallest Dependent Value": determine_smallest_dependent_value(data_frame, target_column),
        "Smallest Absolute Dependent Value": determine_smallest_absolute_dependent_value(data_frame, target_column),
        "Biggest Independent Value": determine_biggest_independent_value(data_frame, target_column),
        "Smallest Independent Value": determine_smallest_independent_value(data_frame, target_column),
        "Smallest Absolute Independent Value": determine_smallest_absolute_independent_value(data_frame, target_column),
        "Mean Dependent Variable": determine_mean_dependent_variable(data_frame, target_column),
        "Standard Deviation Dependent Variable": determine_standard_deviation_dependent_variable(data_frame, target_column),
        "Quartile 1 Dependent Variable": determine_quartile_1_dependent_variable(data_frame, target_column),
        "Median Dependent Variable": determine_median_dependent_variable(data_frame, target_column),
        "Quartile 3 Dependent Variable": determine_quartile_3_dependent_variable(data_frame, target_column),
        "Geometric Mean of Absolute Values Dependent Variable": determine_geometric_mean_abs_values_dependent_variable(data_frame, target_column),
        "Geometric Standard Deviation of Absolute Values Dependent Variable": determine_geometric_std_deviation_abs_values_dependent_variable(data_frame, target_column),
        "Proportion of IQR Method Outliers Dependent Variable": determine_proportion_of_IQR_method_outliers_dependent_variable(data_frame, target_column),
        "Skewness of Dependent Variable": determine_skewness_dependent_variable(data_frame, target_column),
        "Kurtosis of Dependent Variable": determine_kurtosis_dependent_variable(data_frame, target_column)
    }
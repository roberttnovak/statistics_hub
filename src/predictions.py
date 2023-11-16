import datetime
import json
import logging
import os
from pathlib import Path 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
# from own_utils import create_logger 
from sklearn.utils import all_estimators 
from typing import Union, Tuple, List, Dict, Optional
import numpy as np
from OwnLog import OwnLogger
from PersistanceManager import PersistenceManager, persist_model_to_disk_structure
from cleaning import prepare_dataframe_from_db, process_time_series_data
from own_utils import get_all_args, load_json
from sql_utils import data_importer

#ToDo: Acordarse de meter las funciones de que cada vez que se llaman a las funciones que tienen flags
# training-done.txt etc, al entrar, si ve este archivo, se borre mientras hace el proceso 

# Functions to create 

# This function is designed to format a given ISO 8601 datetime string into a custom string
# suitable for folder naming, representing the runtime timestamp. This way, the execution
# time gets preserved in the folder name, aiding in organization and retrieval.

def format_datetime_string(datetime_str):
    """
        Convert a datetime string into a custom formatted string.

        Parameters:
            datetime_str (str): The input datetime string in ISO 8601 format ('YYYY-MM-DD HH:MM:SS+HH:MM').

        Returns:
            str: The custom formatted string ("year_month_day_hour_minutes_seconds(-UTC{utc})").

        Examples:
            >>> format_datetime_string('2023-04-18 00:00:00+00:00')
            '2023_4_18_0_0_0-UTC0'

            >>> format_datetime_string('2023-04-18 00:00:00')
            '2023_4_18_0_0_0'
    """
        
    try:
        
        # Convert the string to a datetime object
        datetime_obj = datetime.datetime.fromisoformat(datetime_str)

        # Extract individual components
        year = datetime_obj.year
        month = datetime_obj.month
        day = datetime_obj.day
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        second = datetime_obj.second
        
        # Check for UTC offset
        utc_offset = datetime_obj.utcoffset()
        if utc_offset is not None:
            utc_offset = int(utc_offset.total_seconds() // 3600)  # Convert UTC offset to hours
            output_str = f"{year}_{month}_{day}_{hour}_{minute}_{second}-UTC{utc_offset}"
        else:
            output_str = f"{year}_{month}_{day}_{hour}_{minute}_{second}"
        
        return output_str
    except Exception as e:
        raise ValueError(f"Error: {e}")
    
    
def name_folder_train_range(ini_train, fin_train):
    """
    Generate a folder name based on the initial and final training datetime strings.

    Parameters:
        ini_train (str): The initial datetime string of the training in ISO 8601 format ('YYYY-MM-DD HH:MM:SS+HH:MM').
        fin_train (str): The final datetime string of the training in ISO 8601 format ('YYYY-MM-DD HH:MM:SS+HH:MM').

    Returns:
        str: A folder name containing formatted datetime strings for initial and final training.
        
    Examples:
        >>> name_folder_train_range('2023-04-18 00:00:00+00:00', '2023-11-18 00:00:00+00:00')
        'initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_11_18_0_0_0-UTC0'
        
    """
    correct_name_folder_ini_train = format_datetime_string(ini_train)
    correct_name_folder_fin_train = format_datetime_string(fin_train)
    output_str = f"initrain-{correct_name_folder_ini_train}___fintrain-{correct_name_folder_fin_train}"
    return output_str

def get_all_regressors_with_its_parameters():
    """
    Obtain a dictionary of regressors and their parameters from scikit-learn.
    
    This function retrieves all regressors available in scikit-learn and extracts 
    their parameter names. It returns a dictionary where each regressor name is a 
    key, and the value is a list of parameter names for that regressor.
    
    Returns
    -------
    dict
        A dictionary with regressor names as keys and lists of parameter names as values.
        
    Example
    -------
    >>> get_all_regressors()
    {
        'ARDRegression': ['alpha_1', 'alpha_2', ...],
        'AdaBoostRegressor': ['base_estimator', 'learning_rate', ...],
        ...
    }
    
    Raises
    ------
    e : TypeError
        If a regressor cannot be instantiated, a TypeError is raised with a message indicating 
        the regressor name and the specific error encountered.
    """
    
    # Obtain all regressors from scikit-learn
    classifiers = all_estimators(type_filter='regressor')
    
    regressor_params = {}
    for name, RegressorClass in classifiers:
        try:
            # Try to instantiate the regressor
            regressor_instance = RegressorClass()
            # If successful, save the parameters to the dictionary
            regressor_params[name] = list(regressor_instance.get_params().keys())
        except TypeError as e:
            # Handle any TypeError encountered during regressor instantiation
            error_message = (
                f"The regressor {name} could not be instantiated. "
                f"It's likely a meta-estimator or wrapper. Error: {e}"
            )
            print("----------------------")
            print(error_message)
            print("----------------------")
    
    return regressor_params

def get_train_test(
    df, 
    ini_train,
    fin_train,
    fin_test,
    name_time_column="timestamp"
    ):
    """
    Splits the DataFrame df into a training set and a test set based on the provided dates.

    Parameters:
        df (pd.DataFrame): DataFrame to be split.
        ini_train (str): Initial date of the training set in the format 'YYYY-MM-DD'.
        fin_train (str): Final date of the training set in the format 'YYYY-MM-DD'.
        fin_test (str): Final date of the test set in the format 'YYYY-MM-DD'.
        name_time_column (str, optional): Name of the column containing the dates. Defaults to "timestamp".

    Returns:
        dict: A dictionary containing two DataFrames: df_train and df_test, corresponding to the training and test sets respectively.

    Examples:
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.date_range(start="2020-01-01", periods=10),
        ...     "value": range(10)
        ... })
        >>> result = get_train_test(df, "2020-01-01", "2020-01-05", "2020-01-10")
        >>> print(result['df_train'])
           timestamp  value
        0 2020-01-01      0
        1 2020-01-02      1
        2 2020-01-03      2
        3 2020-01-04      3
        4 2020-01-05      4
        >>> print(result['df_test'])
           timestamp  value
        0 2020-01-06      5
        1 2020-01-07      6
        2 2020-01-08      7
        3 2020-01-09      8
        4 2020-01-10      9
    """
    # Ensure date strings are converted to datetime objects
    ini_train = pd.to_datetime(ini_train)
    fin_train = pd.to_datetime(fin_train)
    fin_test = pd.to_datetime(fin_test)

    # Error handling for date range
    if not (ini_train < fin_train < fin_test):
        raise ValueError("Date range is invalid: Ensure ini_train < fin_train < fin_test")

    # Create boolean masks for train and test sets
    train_mask = (df[name_time_column] >= ini_train) & (df[name_time_column] <= fin_train)
    test_mask = (df[name_time_column] > fin_train) & (df[name_time_column] <= fin_test)

    # Filter df using boolean indexing
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    # Construct the result dictionary
    train_test = {
        "df_train": df_train,
        "df_test": df_test
    }

    return train_test


def create_lag_lead_features(df, 
                             n_lags=0, 
                             n_leads=0, 
                             lag_columns=None, 
                             lead_columns=None, 
                             fillna_method='mean'):
    """
    Generate lag and lead features for specified columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        n_lags (int, optional): Number of lag features to create. Defaults to 0.
        n_leads (int, optional): Number of lead features to create. Defaults to 0.
        lag_columns (list of str, optional): List of column names for which to create lag features. Defaults to None.
        lead_columns (list of str, optional): List of column names for which to create lead features. Defaults to None.
        fillna_method (str, optional): Method to fill NaN values created by lag/lead operations.
            Supported methods are 'mean', 'median', 'mode', 'zero', and 'ffill'.
            Defaults to 'mean'.

    Returns:
        pd.DataFrame: DataFrame with the new lag and lead feature columns.

    Raises:
        ValueError: If n_lags or n_leads is negative, or if fillna_method is unsupported, or if specified columns are not in df.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        >>> create_lag_lead_features(df, n_lags=1, n_leads=1, lag_columns=['A'], lead_columns=['B'])
    """

    # Managing errors 
    if n_lags < 0 or n_leads < 0:
        raise ValueError("n_lags and n_leads must be non-negative.")

    if fillna_method not in ['mean', 'median', 'mode', 'zero', 'ffill']:
        raise ValueError(f"Unsupported fillna method: {fillna_method}")

    if lag_columns and any(col not in df.columns for col in lag_columns):
        raise ValueError("Some lag_columns are not in the DataFrame.")

    if lead_columns and any(col not in df.columns for col in lead_columns):
        raise ValueError("Some lead_columns are not in the DataFrame.")


    def fill_with_method(series, method):
        """
        Helper function to fill NaN values based on the specified method.
        """
        if method == 'mean':
            return series.fillna(series.dropna().mean())
        elif method == 'median':
            return series.fillna(series.dropna().median())
        elif method == 'mode':
            mode_value = series.mode()
            if not mode_value.empty:
                return series.fillna(mode_value[0])
            else:
                return series.fillna(0)  # Default to zero if mode is empty
        elif method == 'zero':
            return series.fillna(0)
        elif method == 'ffill':
            return series.fillna(method='ffill')
        else:
            raise ValueError(f"Unsupported fillna method: {method}")

    def create_shifted_features(df, n_shifts, shift_columns, shift_direction):
        """
        Helper function to generate shifted (lag or lead) columns.
        """
        shifted_cols = []
        for variable in shift_columns:
            for i in range(1, n_shifts + 1):
                shift = i if shift_direction == 'lag' else -i
                shifted_column = df[variable].shift(shift)
                # Fill NaN values based on specified method
                shifted_column = fill_with_method(shifted_column, fillna_method)
                col_name = f"{shift_direction}_{variable}_{i}"
                df[col_name] = shifted_column
                shifted_cols.append(col_name)
        return df, shifted_cols

    df = df.copy()
    
    # Create lags and/or leads
    if lag_columns:
        df, _ = create_shifted_features(df, n_lags, lag_columns, 'lag')
    if lead_columns:
        df, _ = create_shifted_features(df, n_leads, lead_columns, 'lead')

    return df



def get_all_regressors_of_sklearn():
    """
    Fetch a dictionary of regressors and their parameters in scikit-learn.
    
    This function iterates through all scikit-learn regressors and tries to 
    instantiate them. It then collects their parameters into a dictionary.

    Returns:
    --------
    dict
        A dictionary where keys are regressor names and values are lists 
        containing parameter names for each regressor.

    Examples:
    --------
    >>> get_all_regressors_of_sklearn()
    {'ARDRegression': ['alpha_1', 'alpha_2', ...], 
     'AdaBoostRegressor': ['base_estimator', 'learning_rate', ...], ...}
    
    """
    
    # Create logger
    # logger = create_logger("get_all_regressors_of_sklearn")
    
    # Fetch all scikit-learn regressors
    classifiers = all_estimators(type_filter='regressor')
    
    # Initialize dictionary to hold regressor parameters
    regressor_params = {}
    
    # Loop through each regressor and collect parameters
    for name, RegressorClass in classifiers:
        try:
            regressor_instance = RegressorClass()
            regressor_params[name] = list(regressor_instance.get_params().keys())
        except TypeError as e:
            # ToDo: Fix own_utils importing. When I run tests dont work... 
            # logger.warning(f"The regressor {name} is not a standalone regressor. It's probably a meta-estimator or wrapper. Error: {e}")
            pass
    return regressor_params


def export_regressor_sklearn_details(filename="all_regressors_of_sklearn_and_its_parameters.json", 
                                     directory=".", 
                                     indent=4,
                                     create_dir_if_not_exists=False):
    """
    Export a dictionary of scikit-learn regressors and their parameters to a JSON file.
    The file will include a comment at the beginning.
    
    Parameters:
    -----------
    filename : str, default="all_regressors_of_sklearn_and_its_parameters.json"
        The name of the file where the parameters will be saved.
        
    directory : str, default="."
        The directory where the file will be saved. Default is the current directory.
        
    indent : int, default=4
        Indentation level for the exported JSON.

    create_dir_if_not_exists : bool, default=False
        Whether to create the directory if it doesn't exist.

    Returns:
    --------
    None
    
    Example:
    --------
    >> export_regressor_sklearn_details(filename="my_regressors.json", directory="/desired/path/", create_dir_if_not_exists=True)
    """
    
    # Check if directory exists and is writable
    if not os.path.isdir(directory):
        if create_dir_if_not_exists:
            os.makedirs(directory)
        else:
            raise FileNotFoundError(f"The specified directory '{directory}' does not exist.")
    
    # Get the parameters of the regressors
    regressor_params = get_all_regressors_of_sklearn()

    # Prepare the JSON with a comment
    data_to_save = {
        "__comment__": "This file contains a mapping from scikit-learn regressors to their respective parameters.",
        "regressor_parameters": regressor_params
    }

    # Create the complete path
    full_path = os.path.join(directory, filename)

    # Save the JSON file at the specified path
    with open(full_path, "w") as json_file:
        json.dump(data_to_save, json_file, indent=indent)


def load_regressor_sklearn_details(filename, directory="."):
    """
    Load a dictionary of scikit-learn regressors and their parameters from a JSON file.
    
    Parameters:
    -----------
    filename : str
        The name of the file where the parameters are saved.
        
    directory : str, default="."
        The directory where the file is located. Default is the current directory.
        
    Returns:
    --------
    dict
        A dictionary containing the parameters of scikit-learn regressors.
        
    Raises:
    -------
    FileNotFoundError
        If the specified directory or file does not exist.
    
    Example:
    --------
    >> params = load_regressor_sklearn_details(filename="my_regressors.json", directory="/desired/path/")
    """

    # Create the complete path
    full_path = os.path.join(directory, filename)

    # Check if directory and file exist
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The specified directory '{directory}' does not exist.")
    
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"The specified file '{full_path}' does not exist.")
    
    # Load the JSON file
    with open(full_path, "r") as json_file:
        data_loaded = json.load(json_file)
    
    return data_loaded.get("regressor_parameters", {})


class PreprocessColumns(BaseEstimator, TransformerMixin):
    def __init__(self, X_name_features: Union[str, List[str]], Y_name_features: Union[str, List[str]]):
        """
        Initializes the transformer with the names of the features.
        
        Parameters:
        - X_name_features: Names of the input features.
        - Y_name_features: Names of the target features.
        """
        self.X_name_features = [X_name_features] if isinstance(X_name_features, str) else X_name_features
        self.Y_name_features = [Y_name_features] if isinstance(Y_name_features, str) else Y_name_features

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessColumns':
        """
        Fit method is used to compute the necessary parameters needed to apply
        the transformation. In this case, it does nothing.
        
        Parameters:
        - df: The data frame to be transformed.
        - y: Target values (not used).
        
        Returns:
        - self.
        """
        # This transformer doesn't need fitting, so return self.
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method is used to apply the transformation to a dataframe.
        
        Parameters:
        - df: The data frame to be transformed.
        
        Returns:
        - df_preprocessed: The transformed data frame.
        """
        df_preprocessed = df[self.X_name_features + self.Y_name_features]
        self.df_preprocessed = df_preprocessed

        return df_preprocessed


class LagLeadBaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for creating lag or lead features for a DataFrame.

    ...

    Attributes
    ----------
    n : int
        The number of lags or leads to create for each specified column.
        
    columns : list of str
        The names of columns to apply the operation.
    """

    def __init__(self, n: int, columns: Union[str, List[str]]):
        self.n = n
        self.columns = [columns] if isinstance(columns, str) else columns

    def fit(self, df: pd.DataFrame, y=None) -> 'LagLeadBaseTransformer':
        """ 
        Fit the transformer. This function does nothing in this particular transformer and returns self.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        y : None
            Ignored. Placeholder to maintain compatibility with sklearn's TransformerMixin.

        Returns
        -------
        self : object
            Returns self.
        """
        # Placeholder for fit. Does nothing in this particular transformer.
        return self

class PreprocessLags(LagLeadBaseTransformer):
    """
    A class used for creating lag features for a DataFrame, inheriting from BasePreprocess.

    Methods
    -------
    transform(df: pd.DataFrame) -> pd.DataFrame
        Create lags of the specified columns in the DataFrame.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Create lags of the specified columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        df_preprocessed : pd.DataFrame
            DataFrame with lagged features.
        """
        df_preprocessed = create_lag_lead_features(df, self.n, 0, self.columns, None)
        self.df_preprocessed = df_preprocessed

        return df_preprocessed

class PreprocessLeads(LagLeadBaseTransformer):
    """
    A class used for creating lead features for a DataFrame, inheriting from BasePreprocess.

    Methods
    -------
    transform(df: pd.DataFrame) -> pd.DataFrame
        Create leads of the specified columns in the DataFrame.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Create leads of the specified columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        df_preprocessed : pd.DataFrame
            DataFrame with lead features.
        """
        df_preprocessed = create_lag_lead_features(df, 0, self.n, None, self.columns)
        self.df_preprocessed = df_preprocessed
        return df_preprocessed

class PreprocessScaler(BaseEstimator, TransformerMixin):
    """
    A class used for scaling features of a DataFrame.

    Attributes
    ----------
    method : str, default="standard"
        The method for scaling features. Can be "standard" for StandardScaler or "minmax" for MinMaxScaler.
    
    fit_scalers : bool, default=True
        Whether to fit the scaler on the provided data.

    minmax_range : tuple, default=(0.01, 0.99)
        The feature range for MinMaxScaler.

    Methods
    -------
    fit(df: pd.DataFrame, y=None) -> 'PreprocessScaler'
        Fit the scaler on the DataFrame columns. Returns self.
        
    transform(df: pd.DataFrame) -> pd.DataFrame
        Scale the features of the DataFrame. Returns the scaled DataFrame.
    
    inverse_transform(df: pd.DataFrame) -> pd.DataFrame
        Inverse scale the features of the DataFrame. Returns the inverse scaled DataFrame.
        
    inverse_transform_by_col(array: np.ndarray, col: str) -> np.ndarray
        Apply inverse transformation on the entire array using the scaler of a specific column.
    """

    def __init__(self, method: str="standard", fit_scalers: bool=True, minmax_range: Tuple[float, float]=(0.01, 0.99)):
        self.scalers = {}
        self.fit_scalers = fit_scalers
        self.method = method
        self.minmax_range = minmax_range

    def fit(self, df: pd.DataFrame, y=None) -> 'PreprocessScaler':
        """Fit the scaler on the DataFrame columns."""
        if self.fit_scalers:
            df_preprocessed = pd.DataFrame(index=df.index, columns=df.columns)
            for col in df.columns:
                scaler = MinMaxScaler(feature_range=self.minmax_range) if self.method == "minmax" else StandardScaler()
                self.scalers[col] = scaler.fit(df[[col]])
        self.df_preprocessed = df_preprocessed
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale the features of the DataFrame."""
        df_preprocessed = df.copy()
        for col in df.columns:
            df_preprocessed[col] = self.scalers[col].transform(df[[col]])
        return df_preprocessed

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse scale the features of the DataFrame."""
        df_inverse = df.copy()
        for col in df.columns:
            df_inverse[col] = self.scalers[col].inverse_transform(df[[col]])
        return df_inverse

    def inverse_transform_by_col(self, array: np.ndarray, col: str) -> np.ndarray:
        """
        Apply inverse transformation on the entire array using the scaler of a specific column.

        Parameters:
        ----------
        array : np.ndarray
            The matrix or array you want to inversely transform.
        col : str
            The name of the column of the scaler you wish to use to inverse transform the entire array.

        Returns:
        -------
        np.ndarray
            Inversed array.
        """
        inversed_data = self.scalers[col].inverse_transform(array)
        return inversed_data
    
    def __getattr__(self, attr):
        return {col: getattr(scaler, attr) for col, scaler in self.scalers.items()}

    
# Postprocessing

def get_last_df_step_preprocessed(dict_machine_learning_preprocessing: Dict[str, Dict]) -> pd.DataFrame:
    """
    Retrieves the DataFrame from the last preprocessing step based on the highest order key.
    
    Parameters
    ----------
    dict_machine_learning_preprocessing : Dict[str, Dict]
        Dictionary containing preprocessing data with an 'order' key indicating the order of processing.
    
    Returns
    -------
    pd.DataFrame
        DataFrame from the last preprocessing step.
    """
    highest_order_key = max(
        dict_machine_learning_preprocessing, 
        key=lambda k: dict_machine_learning_preprocessing[k]['order']
    )
    return dict_machine_learning_preprocessing[highest_order_key]['df']


def get_df_X(df_last_step_preprocessed: pd.DataFrame, X_name_features: List[str], Y_name_features: List[str]) -> pd.DataFrame:
    """
    Generates a DataFrame containing the input features for training a model using scikit-learn's model.fit(x, y) where x = get_df_X(...)
    The DataFrame includes lag and actual values of the specified input and target features.

    Parameters
    ----------
    df_last_step_preprocessed : pd.DataFrame
        Preprocessed DataFrame from which to extract the input features.
    X_name_features : List[str]
        List of names for the input features.
    Y_name_features : List[str]
        List of names for the target features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the input features, including lag and actual values of the specified input and target features,
        to be used as 'x' in model.fit(x, y).
    """
    x_lags_columns = [
    column for column in df_last_step_preprocessed.columns 
    for x in X_name_features 
    if  x in column and "lag" in column
    ]
    
    x_actual_columns = [
        column for column in df_last_step_preprocessed.columns 
        for x in X_name_features 
        if  x==column
    ]
    
    y_lag_columns = [
        column for column in df_last_step_preprocessed.columns 
        for y in Y_name_features 
        if  y in column and "lag" in column
    ]
    
    y_actual_columns = [
        column for column in df_last_step_preprocessed.columns 
        for y in Y_name_features 
        if  y==column
    ]
    return df_last_step_preprocessed[x_actual_columns + x_lags_columns + y_actual_columns + y_lag_columns]

def get_df_Y(df_last_step_preprocessed: pd.DataFrame, Y_name_features: List[str]) -> pd.DataFrame:
    """
    Generates a DataFrame containing the lead values of the specified target features for evaluating prediction errors
    using scikit-learn's model.fit(x, y) where y = get_df_Y(...)

    Parameters
    ----------
    df_last_step_preprocessed : pd.DataFrame
        Preprocessed DataFrame from which to extract the lead values of the target features.
    Y_name_features : List[str]
        List of names for the target features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the lead values of the specified target features, to be used as 'y' in model.fit(x, y),
        for evaluating prediction errors.
    """
    y_lead_columns = [
        column for column in df_last_step_preprocessed.columns 
        for y in Y_name_features 
        if y in column and "lead" in column
    ]
    
    df_Y = df_last_step_preprocessed[y_lead_columns]

    return df_Y

def preprocess_machine_learning_algorithm(
        df: pd.DataFrame,
        X_name_features: List[str],
        Y_name_features: List[str],
        n_lags: int,
        n_leads: int,
        lag_columns: List[str],
        lead_columns: List[str],
        fit: bool = True,
        transformers_order: List[str] = None,
        preprocess_scaler_instance: 'PreprocessScaler' = None
) -> Dict[str, Dict]:
    """
    Preprocesses a DataFrame based on specified transformations for machine learning.

    The function encapsulates the instantiation and application of various data transformations including column selection,
    scaling, lag and lead feature creation.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    X_name_features : List[str]
        List of feature names for the input variables.
    Y_name_features : List[str]
        List of feature names for the target variables.
    n_lags : int
        Number of lag features to create.
    n_leads : int
        Number of lead features to create.
    lag_columns : List[str]
        List of column names to create lag features for.
    lead_columns : List[str]
        List of column names to create lead features for.
    fit : bool, optional
        Whether to fit the scalers, by default True.
    transformers_order : List[str], optional
        Specifies the order of transformations, by default None which results in a predefined order.
    preprocess_scaler_instance : PreprocessScaler, optional
        An instance of PreprocessScaler to be used for scaling, by default None which creates a new instance.

    Returns
    -------
    Dict[str, Dict]
        Dictionary containing the order, DataFrame, and transformer instance for each preprocessing step.
    """
    # Preserve input DataFrame
    df = df.copy()

    # Default order if not provided
    if transformers_order is None:
        transformers_order = ["preprocess_columns", "preprocess_scaler", "preprocess_lags", "preprocess_leads"]

    # Instantiate transformers
    preprocess_columns_instance = PreprocessColumns(X_name_features, Y_name_features)
    preprocess_lags_instance = PreprocessLags(n=n_lags, columns=lag_columns)
    preprocess_leads_instance = PreprocessLeads(n=n_leads, columns=lead_columns)
    preprocess_scaler_instance = PreprocessScaler(fit_scalers=fit)

    # Mapping names to instances
    transformers_dict = {
        "preprocess_columns": preprocess_columns_instance,
        "preprocess_lags": preprocess_lags_instance,
        "preprocess_leads": preprocess_leads_instance,
        "preprocess_scaler": preprocess_scaler_instance
    }

    # Order dictionary
    order_dict = {name: order + 1 for order, name in enumerate(transformers_order)}

    # Organizing transformers in the specified order
    transformers_in_order = [transformers_dict[name] for name in transformers_order]

    # Creating pipeline
    pipeline = make_pipeline(*transformers_in_order)

    # Processing DataFrame
    df_processed = pipeline.fit_transform(df)

    # Organizing the output
    # preprocess = {
    #     name: {
    #         "order": order_dict[name],
    #         "df": getattr(transformers_dict[name], 'df_preprocessed', None),
    #         "transformer": transformers_dict[name]
    #     }
    #     for name in transformers_order
    # }
    preprocess = {
        "preprocess_columns": {
            "order": order_dict["preprocess_columns"], 
            "df": preprocess_columns_instance.df_preprocessed,
            "transformer": preprocess_columns_instance
        },
        "preprocess_lags": {
            "order": order_dict["preprocess_lags"], 
            "df": preprocess_lags_instance.df_preprocessed,
            "transformer": preprocess_lags_instance
        },      
        "preprocess_leads": {
            "order": order_dict["preprocess_leads"], 
            "df": preprocess_leads_instance.df_preprocessed,
            "transformer": preprocess_leads_instance
        },
        "preprocess_scaler": {
            "order": order_dict["preprocess_scaler"], 
            "df": preprocess_scaler_instance.df_preprocessed,
            "transformer": preprocess_scaler_instance
        }
    }

    return preprocess


def create_model_machine_learning_algorithm(
        tidy_data,
        ini_train,
        fin_train,
        fin_test,
        id_device,
        model_sklearn_name,
        X_name_features,
        Y_name_features,
        n_lags,
        n_leads,
        lag_columns,
        lead_columns,
        scale_in_preprocessing=True,
        name_time_column="timestamp",
        name_id_sensor_column="id_device",
        save_preprocessing=True,
        path_to_save_model=None,
        folder_name_model=None,
        folder_name_time_execution="execution-time-no-defined",
        folder_name_preprocessed_data="preprocessed-data-to-use-in-model",
        machine_learning_model_args=None,
        logger: logging.Logger = None,
        **kwargs
):
    """
    Function to create and train a machine learning model using scikit-learn.

    Parameters:
    -----------
    tidy_data : pd.DataFrame
        The tidy data frame to be used for model training and testing.
    ini_train : str
        The start date for the training data.
    fin_train : str
        The end date for the training data.
    fin_test : str
        The end date for the testing data.
    id_device : str
        The identifier for the device.
    model_sklearn_name : str
        Name of the scikit-learn model to train (e.g., "SVR").
    X_name_features : list
        List of feature column names.
    Y_name_features : list
        List of target column names.
    n_lags : int
        Number of lags to consider for features.
    n_leads : int
        Number of leads to consider for targets.
    lag_columns : list
        List of lag column names.
    lead_columns : list
        List of lead column names.
    scale_in_preprocessing : bool, optional
        Whether to scale the data during preprocessing. Default is True.
    name_time_column : str, optional
        Name of the time column. Default is "timestamp".
    name_id_sensor_column : str, optional
        Name of the sensor ID column. Default is "id_device".
    save_preprocessing : bool, optional
        Whether to save the preprocessing steps. Default is True.
    path_to_save_model : str, optional
        Path where the model will be saved. Default is None.
    folder_name_model : str, optional
        Name of the folder where the model will be saved. Default is None.
    folder_name_time_execution : str, optional
        Name of the folder indicating execution time. Default is "execution-time-no-defined".
    folder_name_preprocessed_data : str, optional
        Name of the folder for preprocessed data to be used in the model. Default is "preprocessed-data-to-use-in-model".
    machine_learning_model_args : dict, optional
        Dictionary of arguments for the machine learning model. Default is None.
    kwargs : dict
        Additional arguments for the machine learning model.

    Returns:
    --------
    dict
        Dictionary containing the trained model, preprocessed data, and training and testing sets.

    Directory and File Structure (if path_to_save_model is specified):
        path_to_save_model/
        ├── folder_name_model/
        │   ├── model
        │   └── metadata.json
    """

    # Log model name
    if logger:
        logger.info(f"Model: {model_sklearn_name}")

    # Set folder_name_model if None
    folder_name_model = model_sklearn_name if folder_name_model is None else folder_name_model
    # Save tidy_data: It is necessary to use this DataFrame when using get_df_predictions
    folder_name_range_train = name_folder_train_range(ini_train, fin_train)

    pm = PersistenceManager(
        base_path=path_to_save_model,
        folder_name_model = folder_name_model,
        folder_name_range_train=folder_name_range_train,
        folder_name_time_execution=folder_name_time_execution
    )


    pm.remove_flag("training-done")

    pm.save_preprocessed_data(
        preprocessed_data=tidy_data[ tidy_data[name_id_sensor_column] == id_device ], 
        folder_name_preprocessed_data=folder_name_preprocessed_data,
        name = "tidy_data"
    )

    # --- Auxiliary Functions ---
    # The following auxiliary functions are encapsulated to ensure the code's modularity and readability.
    # 1. get_sklearn_regressor_classes: This function provides a list of available regressor classes from scikit-learn.
    #    It's essential to validate the user input for the machine learning model and ensure it's a valid regressor class.
    # 2. preprocess_data: A crucial step in machine learning is data preprocessing. This function takes raw data
    #    and performs necessary preprocessing steps, preparing it for the machine learning algorithm.
    # These functions help in keeping the main function `create_model_machine_learning_algorithm` clean and understandable.

    def get_sklearn_regressor_classes():
        """
        Obtains a list of available regressor classes in scikit-learn.
        
        This function is useful for identifying which regressors can be used
        when creating and training models in the create_model_machine_learning_algorithm function.
        
        Returns:
        --------
        list
            A list of scikit-learn regressor classes.
            
        Example:
        --------
        >> regressor_classes = get_sklearn_regressor_classes()
        >> print(regressor_classes)
        [<class 'sklearn.linear_model._bayes.ARDRegression'>, <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>, ...]
        """
        classifiers = all_estimators(type_filter='regressor')
        return [RegressorClass for _, RegressorClass in classifiers]


    def preprocess_data(df, X_name_features, Y_name_features, n_lags, n_leads, lag_columns, lead_columns):
        """
        Preprocesses the data for the machine learning algorithm.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The data frame to be preprocessed.
        X_name_features : list
            List of feature column names.
        Y_name_features : list
            List of target column names.
        n_lags : int
            Number of lags to consider for features.
        n_leads : int
            Number of leads to consider for targets.
        lag_columns : list
            List of lag column names.
        lead_columns : list
            List of lead column names.
            
        Returns:
        --------
        tuple
            A tuple containing the preprocessing steps, the preprocessed feature data frame, 
            and the preprocessed target data frame.
        """
        df=df.copy()
        #Filter id_device
        df = df[df[name_id_sensor_column] == id_device].reset_index(drop=True)

        preprocessing = preprocess_machine_learning_algorithm(
            df=df,
            X_name_features=X_name_features,
            Y_name_features=Y_name_features,
            n_lags=n_lags,
            n_leads=n_leads,
            lag_columns=lag_columns,
            lead_columns=lead_columns
        )
        df_last_step = get_last_df_step_preprocessed(preprocessing)
        df_X = get_df_X(df_last_step, X_name_features=X_name_features, Y_name_features=Y_name_features)
        df_Y = get_df_Y(df_last_step, Y_name_features=Y_name_features)
        return preprocessing, df_X, df_Y

    # Log preprocessing info
    if logger:
        logger.info(f"{model_sklearn_name}: Preprocessing data ...")

    # Get training and testing data
    df_train, df_test = get_train_test(
        df=tidy_data,
        ini_train=ini_train,
        fin_train=fin_train,
        fin_test=fin_test,
        name_time_column=name_time_column
    ).values()

    df_train = df_train[ df_train[name_id_sensor_column]==id_device].reset_index(drop=True)
    df_test = df_test[ df_test[name_id_sensor_column]==id_device].reset_index(drop=True)

    # Use kwargs if machine_learning_model_args is None
    machine_learning_model_args = kwargs if machine_learning_model_args is None else machine_learning_model_args

    # Preprocessing
    preprocessing_train, df_train_X, df_train_Y = preprocess_data(df_train, X_name_features, Y_name_features, n_lags, n_leads, lag_columns, lead_columns)
    preprocessing_test, df_test_X, df_test_Y = preprocess_data(df_test, X_name_features, Y_name_features, n_lags, n_leads, lag_columns, lead_columns)
    
    if scale_in_preprocessing:
        scaler = preprocessing_train["preprocess_scaler"]["transformer"]
    else: 
        scaler = None

    if logger:
        logger.info(f"{model_sklearn_name}: Training ...")

    # Creating and training the model
    all_regressors = get_sklearn_regressor_classes()
    RegressorClass = next((cls for cls in all_regressors if cls.__name__ == model_sklearn_name), None)
    if RegressorClass is None:
        raise ValueError(f"No regressor found for {model_sklearn_name} in scikit-learn.")

    base_model = RegressorClass(**machine_learning_model_args)
    model = MultiOutputRegressor(base_model)
    model.fit(df_train_X, df_train_Y)

    # Metadata: combining default arguments with provided arguments
    all_args = get_all_args(RegressorClass.__init__)
    metadata = {**all_args, **machine_learning_model_args}

    # Optional: save the model
    if logger:
        logger.info(f"{model_sklearn_name}: Saving to directory ...")
    if path_to_save_model is not None:

        preprocessed_data = {
            "df_train_X": df_train_X,
            "df_train_Y": df_train_Y,
            "df_test_X": df_test_X,
            "df_test_Y": df_test_Y,
            "df_train": df_train,
            "df_test": df_test
        }

        persist_model_to_disk_structure(
            path_to_save_model = path_to_save_model,
            folder_name_model = folder_name_model,
            folder_name_range_train = folder_name_range_train,
            folder_name_time_execution = folder_name_time_execution,
            model = model,
            metadata = metadata,
            scaler = scaler,
            folder_name_preprocessed_data = folder_name_preprocessed_data if save_preprocessing else None,
            preprocessed_data = preprocessed_data,
            additional_persistable_objects_to_save=None
        )

    return {
        "model": model,
        "preprocessing_train": preprocessing_train,
        "preprocessing_test": preprocessing_test,
        "df_train_X": df_train_X,
        "df_test_X": df_test_X,
        "scaler": scaler
    }

def get_df_predictions(y_pred_inverse_transform,
                       name_model,
                       n_predictions,
                       df,
                       tidy_data,
                       type_model="machine learning"):
    """
    Generate a DataFrame containing model predictions, real values, and associated metadata.
    
    Parameters:
    -----------
    y_pred_inverse_transform : array-like
        Array containing the inverse-transformed predictions from the model.
        
    name_model : str
        The name of the model used for making predictions.
        
    n_predictions : int
        Number of prediction steps.
        
    df : DataFrame
        The training DataFrame containing the 'timestamp' column.

    tidy_data : DataFrame
        The DataFrame with real values for aggregate real values to predictions
        
    type_model : str, optional (default = "machine learning")
        The type of model used: either 'machine learning' or 'esrnn'.
    
    Returns:
    --------
    DataFrame
        DataFrame containing model predictions, real values, and additional information.
    
    Examples:
    ---------
    >>> get_df_predictions(y_pred_inv, 'RandomForest', 5, df_train)
    """
    
    if type_model == "machine learning":
        # For machine learning models, concatenate timestamps with predictions
        df_predictions = pd.concat(
            [df["timestamp"], pd.DataFrame(y_pred_inverse_transform, columns=range(1, n_predictions + 1))],
            axis=1)
        
    elif type_model == "esrnn":
        # For ESRNN models, special handling is needed for predictions
        df_predictions = pd.concat(
            [df[["timestamp"]].iloc[[0]], pd.DataFrame(y_pred_inverse_transform.T.values, columns=range(1, y_pred_inverse_transform.shape[0] + 1))],
            axis=1)
    
    # Melt and sort the DataFrame for easier analysis
    df_predictions = df_predictions.melt(id_vars='timestamp', var_name='n_prediction', value_name='prediction') \
                                    .sort_values(["timestamp", "n_prediction"]) \
                                    .reset_index(drop=True)
    
    # Convert columns to appropriate data types
    df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
    df_predictions['n_prediction'] = pd.to_timedelta(df_predictions['n_prediction'], unit='min')
    
    # Vectorized operation to calculate the prediction timestamp
    df_predictions['timestamp_prediction'] = df_predictions['timestamp'] + df_predictions['n_prediction']
    
    # Add metadata about the training data timeframe
    df_predictions["timestamp_init_train"] = df["timestamp"].min()
    df_predictions["timestamp_end_updated_train"] = df["timestamp"].max()

    # Merge real values for comparison
    df_predictions = df_predictions.rename(columns={"timestamp": "timestamp_real"}).merge(
        tidy_data[["timestamp", "y"]].rename(columns={"y": "value_real"}),
        how="left",
        left_on="timestamp_prediction",
        right_on="timestamp") \
        .drop(columns="timestamp")
    
    # Calculate the mean absolute error (MAE) between predictions and real values
    df_predictions["mae"] = np.abs(df_predictions["prediction"] - df_predictions["value_real"])
    
    # Add the model name for tracking
    df_predictions["model"] = name_model
    
    # The only return statement in the function
    return df_predictions

def process_model_machine_learning(
    path_to_save_model: str,
    folder_name_model: str,
    folder_name_range_train: str,
    folder_name_time_execution: str,
    folder_name_preprocessed_data: str = "preprocessed-data-to-use-in-model",
    save: bool = True,
    folder_name_predictions: str = "predictions",
    name_column_objective_variable_of_scaler: str = "y",
    flag_predictions_done: str = "predictions-done.txt",
    overwrite_predictions: bool = False
) -> dict:
    """
    Process a machine learning model by loading preprocessed data, making predictions on training and 
    test sets, optionally inverting scaling transformations on the predictions, and optionally saving 
    the predictions to disk.
    
    Parameters:
    -----------
    path_to_save_model (str): The path to save the model.
    folder_name_model (str): The folder name for the model.
    folder_name_range_train (str): The folder name for the range of training data.
    folder_name_time_execution (str): The folder name for time of execution data.
    folder_name_preprocessed_data (str, optional): The name of the folder containing preprocessed data.
    save (bool, optional): Whether to save the predictions to disk.
    folder_name_predictions (str, optional): The name of the folder to save predictions in.
    name_column_objective_variable_of_scaler (str, optional): The name of the column used for scaling.
    flag_predictions_done (str, optional): The name of the file to create upon successful prediction.
    overwrite_predictions (bool, optional): Whether to overwrite existing predictions files.
    
    Returns:
    --------
    Tuple
        A tuple containing dictionaries with DataFrames for test and training predictions.
    
    Examples:
    ---------
    >>> process_model_machine_learning("path/to/save/model", "model_name", "range_train_folder", "time_execution_folder")
    """
    # Construct the full path to the preprocessed data directory

    

    pm = PersistenceManager(
        base_path = path_to_save_model,
        folder_name_model = folder_name_model, 
        folder_name_range_train = folder_name_range_train, 
        folder_name_time_execution = folder_name_time_execution
    )
    
    # Load preprocessed data
    df_train_X = pm.load_preprocessed_data(
        folder_name_preprocessed_data=folder_name_preprocessed_data,
        name="df_train_X",
        extension="csv",
        datetime_columns=[]
    )

    df_test_X = pm.load_preprocessed_data(
        folder_name_preprocessed_data=folder_name_preprocessed_data,
        name="df_test_X",
        extension="csv",
        datetime_columns=[]
    )

    df_train = pm.load_preprocessed_data(
        folder_name_preprocessed_data=folder_name_preprocessed_data,
        name="df_train",
        extension="csv",
        datetime_columns=[]
    )

    df_test = pm.load_preprocessed_data(
        folder_name_preprocessed_data=folder_name_preprocessed_data,
        name="df_test",
        extension="csv",
        datetime_columns=[]
    )

    tidy_data = pm.load_preprocessed_data(
        folder_name_preprocessed_data=folder_name_preprocessed_data,
        name="tidy_data",
        extension="csv",
        datetime_columns=["timestamp"]
    )

    scaler = pm.load_scaler(folder_name_preprocessed_data = folder_name_preprocessed_data)

    
    # Load the machine learning model
    model = pm.load_model()
    
    # Generate predictions on the training and test sets
    y_pred_test = model.predict(df_test_X)
    y_pred_train = model.predict(df_train_X)

    # Ensure the predictions are 2D arrays
    assert y_pred_test.ndim == 2, "Predictions must be a 2D array"
    assert y_pred_train.ndim == 2, "Predictions must be a 2D array"

    # Invert scaling transformations on the predictions if a scaler is provided
    if scaler is not None:
        y_pred_test_inverse_transform = scaler.inverse_transform_by_col(y_pred_test, name_column_objective_variable_of_scaler)
        y_pred_train_inverse_transform = scaler.inverse_transform_by_col(y_pred_train, name_column_objective_variable_of_scaler)
    else: 
        y_pred_test_inverse_transform = y_pred_test
        y_pred_train_inverse_transform = y_pred_train
        
    # Generate DataFrames for the predictions
    df_predictions_test = get_df_predictions(
        y_pred_inverse_transform = y_pred_test_inverse_transform, 
        name_model = folder_name_model,
        n_predictions = y_pred_test_inverse_transform.shape[1],
        df = df_test,
        tidy_data= tidy_data,
        type_model = "machine learning"
    )

    df_predictions_test["dataset_type"] = "test"
    
    df_predictions_train = get_df_predictions(
        y_pred_inverse_transform = y_pred_train_inverse_transform, 
        name_model = folder_name_model,
        n_predictions = y_pred_train_inverse_transform.shape[1],
        df = df_train,
        tidy_data = tidy_data,
        type_model = "machine learning"
    )

    df_predictions_train["dataset_type"] = "train"
    

    # Save the predictions to disk if requested
    if save:
        pm.save_predictions(
            predictions = df_predictions_train,
            folder_name_predictions = folder_name_predictions,
            name = "predictions-train",
            overwrite = overwrite_predictions
        )
        pm.save_predictions(
            predictions = df_predictions_test,
            folder_name_predictions = folder_name_predictions,
            name = "predictions-test",
            overwrite = overwrite_predictions
        )
        # Create a file to indicate that the predictions have been completed successfully
        path_flag_predictions = os.path.join(
            path_to_save_model,
            folder_name_model,
            folder_name_range_train,
            folder_name_time_execution
        )
        with open(os.path.join(path_flag_predictions, flag_predictions_done), 'w') as f:
            f.write('Predictions completed successfully.\n')
    
    return {"df_predictions_train":df_predictions_train, "df_predictions_test": df_predictions_test}

def summarise_mae(predictions, freq = None, group_by_timestamp=True):
    """
    Compute various statistics of the Mean Absolute Error (MAE) and the count of predictions for each unique timestamp
    and for each data type (train or test) in the given DataFrame. Optionally groups data within a specified time interval
    before performing the group-by and aggregation operations.
    
    Parameters:
    predictions (pd.DataFrame): A DataFrame containing at least the columns 'timestamp_real', 
                                'timestamp_prediction', 'mae', 'model' and 'n_prediction'.
    freq (str, optional): A string representing the frequency for grouping timestamps (e.g., '5T' for 5 minutes,
                           '1H' for 1 hour, etc.). If None, no additional grouping is performed.
                           Default is None.
    group_by_timestamp (bool, optional): A boolean indicating whether to group by 'timestamp_real' in addition to
                                         'dataset_type'. If False, only groups by 'dataset_type'. Default is True.
    
    Returns:
    pd.DataFrame: A DataFrame containing the columns 'timestamp', 'mean_mae', 'median_mae', 'q1_mae', 'q3_mae',
                  'std_mae', and 'number_of_predictions', where 'timestamp' is the unique timestamp from the 
                  'timestamp_real' column of the original DataFrame (or the start of the interval if freq is provided),
                  'mean_mae' is the mean of the 'mae' values for that timestamp (or interval), 
                  'median_mae' is the median of the 'mae' values for that timestamp (or interval), 
                  'q1_mae' is the first quartile of the 'mae' values for that timestamp (or interval), 
                  'q3_mae' is the third quartile of the 'mae' values for that timestamp (or interval), 
                  'std_mae' is the standard deviation of the 'mae' values for that timestamp (or interval),
                  and 'number_of_predictions' is the count of predictions for that timestamp (or interval).
                  
    Examples:
    >>> predictions_train_test = ...
    >>> summarised_data = summarise_mae(predictions_train)
    >>> print(summarised_data.head())
                      timestamp   mean_mae  median_mae  q1_mae  q3_mae  std_mae  number_of_predictions
    0 2023-04-18 09:31:00+00:00  66.794667         ...     ...     ...     ...                     20
    1 2023-04-18 09:32:00+00:00  19.148333         ...     ...     ...     ...                     20
    2 2023-04-18 09:33:00+00:00   1.900000         ...     ...     ...     ...                     20
    3 2023-04-18 09:34:00+00:00   2.050000         ...     ...     ...     ...                     20
    4 2023-04-18 09:35:00+00:00   2.400000         ...     ...     ...     ...                     20

    >>> predictions_train = ...
    >>> summarised_data = summarise_mae(predictions_train, freq='5T')
    >>> print(summarised_data.head())
                      timestamp   mean_mae  median_mae  q1_mae  q3_mae  std_mae  number_of_predictions
    0 2023-04-18 09:30:00+00:00  66.794667         ...     ...     ...     ...                     20
    1 2023-04-18 09:35:00+00:00  19.148333         ...     ...     ...     ...                     20
    2 2023-04-18 09:40:00+00:00   1.900000         ...     ...     ...     ...                     20
    
    Input DataFrame Structure:
    | timestamp_real            | timestamp_prediction      | mae        | n_prediction    | dataset_type | ... |
    | ------------------------- | ------------------------- |----------- |-----------------|--------------|-----|
    | 2023-04-18 09:31:00+00:00 | 2023-04-18 09:32:00+00:00 | 50         | 0 days 00:01:00 |    train     | ... |
    | 2023-04-18 09:31:00+00:00 | 2023-04-18 09:33:00+00:00 | 60         | 0 days 00:02:00 |    train     | ... |
    | 2023-04-18 09:31:00+00:00 | 2023-04-18 09:34:00+00:00 | 55         | 0 days 00:03:00 |    train     | ... |
    | 2023-04-18 09:32:00+00:00 | 2023-04-18 09:33:00+00:00 | 50         | 0 days 00:03:00 |    train     | ... |
    |            ...            |           ...             |     ...    |       ...       |              | ... |
    
    Output DataFrame Structure:
    | timestamp                 |dataset_type| mean_mae  | median_mae | q1_mae | q3_mae | std_mae | number_of_predictions |
    |---------------------------|------------|---------- |------------|--------|--------|---------|-----------------------|
    | 2023-04-18 09:31:00+00:00 |  train     | 55        | ...        | ...    | ...    | ...     | 3                     |
    |            ...            |            |    ...    | ...        | ...    | ...    | ...     |          ...          |
    """
    # ToDo: Falta testar la parte de la agrupación
    if freq:
        # Resample the DataFrame to the specified frequency, using the 'timestamp_real' column as the index
        predictions = predictions.set_index('timestamp_real').resample(freq).mean().reset_index()

    if group_by_timestamp:
        group_by_cols = ["timestamp_real", "dataset_type", "model"]
    else:
        group_by_cols = ["dataset_type", "model"]    
    
    predictions_summarise = predictions\
    .groupby(group_by_cols)\
    .agg(
        {   "mae": [  # Multiple statistics for the 'mae' column
                ("q1_mae", lambda x: x.quantile(0.25)),
                ("median_mae", "median"),
                ("q3_mae", lambda x: x.quantile(0.75)),
                ("mean_mae", "mean"),
                ("std_mae", "std")
            ],
            "n_prediction": "count"
        })\
    .reset_index()\
    .rename(columns={"timestamp_real": "timestamp", "n_prediction": "number_of_predictions"})
    # Flatten MultiIndex columns
    predictions_summarise.columns = [col[1] if col[1] != '' else col[0] for col in predictions_summarise.columns.values]

    return predictions_summarise

def evaluate_model(
    path_to_save_model,
    folder_name_model,
    folder_name_range_train,
    folder_name_time_execution,
    folder_name_predictions,
    save=False,
    flag_name = "evaluations-done",
    flag_content =  "",
    flag_subfolder = None,
    folder_name_evaluation = "evaluations",
    show_args_in_save = False,
    **kwargs_summarise_mae
):
    """
    Loads prediction data from specified paths, evaluates the model using the summarise_mae function,
    and optionally saves the summary data to disk.
    
    Parameters:
    path_to_save_model (str): The path to the directory where models are saved.
    folder_name_model (str): The name of the folder where the model is stored.
    folder_name_range_train (str): The name of the folder indicating the range of training data.
    folder_name_time_execution (str): The name of the folder indicating the time of model execution.
    folder_name_predictions (str): The name of the folder containing prediction data.
    save (bool, optional): A flag indicating whether to save the summary data to disk. Default is False.
    flag_name (str, optional): Name of the flag file of the process. Default is "evaluations-done"
    flag_content (str, optional): content of the flag file. Default is empty string
    flag_subfolder (str, optional): Subfolder to save  flag file. Default is None 
    show_args_in_save (bool, optional): A flag indicating whether to include the arguments passed to summarise_mae function in the saved file name. Default is False.
    **kwargs_summarise_mae: 
        Additional keyword arguments to be passed to the summarise_mae function. This can include:
        - freq (str, optional): A string representing the frequency for grouping timestamps.
        - group_by_timestamp (bool, optional): A boolean indicating whether to group by 'timestamp_real'.
    
    Returns:
    pd.DataFrame: A DataFrame containing the summary statistics of model evaluation.
    
    Examples:
    >>> path_to_save_model = os.path.join("..", "models")
    >>> folder_name_model = "KNeighborsRegressor"
    >>> folder_name_range_train = "initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0"
    >>> folder_name_time_execution = "execution-time-2023_10_26_13_31_40" 
    >>> folder_name_predictions = "predictions"
    >>> freq = None 
    >>> group_by_timestamp = True
    >>> evaluated_data = evaluate_model(
    ...     path_to_save_model=path_to_save_model,
    ...     folder_name_model=folder_name_model,
    ...     folder_name_range_train=folder_name_range_train,
    ...     folder_name_time_execution=folder_name_time_execution,
    ...     folder_name_predictions=folder_name_predictions,
    ...     freq=freq,
    ...     group_by_timestamp=group_by_timestamp
    ... )
    >>> print(evaluated_data.head())
                      timestamp   mean_mae  median_mae  q1_mae  q3_mae  std_mae  number_of_predictions
    0 2023-04-18 09:31:00+00:00  66.794667         ...     ...     ...     ...                     20
    ...
    """
    # Initialize a PersistenceManager instance with the provided paths and folder names
    pm = PersistenceManager(
        base_path=path_to_save_model,
        folder_name_model=folder_name_model,
        folder_name_range_train=folder_name_range_train,
        folder_name_time_execution=folder_name_time_execution
    )

    if save:
        pm.remove_flag("evaluations-done")
    
    # Load the training and testing predictions
    predictions_train = pm.load_predictions("predictions-train", folder_name_predictions=folder_name_predictions)
    predictions_train["dataset_type"] = "train"
    predictions_test = pm.load_predictions("predictions-test", folder_name_predictions=folder_name_predictions)
    predictions_test["dataset_type"] = "test" 
    
    # Concatenate training and testing predictions into a single DataFrame
    predictions_train_test = pd.concat([predictions_train, predictions_test]).reset_index(drop=True)
    
    # Evaluate the model using the summarise_mae function with the provided keyword arguments
    predictions_train_test_summarise = summarise_mae(predictions_train_test, **kwargs_summarise_mae)
    
    # Optionally save the summary data to disk
    if save:
        save_name = "evaluations"
        if show_args_in_save:
            arg_name = "___".join([f"{arg}-{value}" for arg, value in kwargs_summarise_mae.items()])
            save_name = f"{save_name}___{arg_name}"
        # The logic for saving the summary data can be added here
        pm.save_evaluation_data(predictions_train_test_summarise, name = save_name , folder_name_evaluation=folder_name_evaluation)
        pm.create_flag(flag_name=flag_name, content = flag_content,sub_folder=flag_subfolder)
    
    # Return the summary data
    return predictions_train_test_summarise

def run_time_series_prediction_pipeline(config_path: str, model_name: str, config_log_filename: str = None):
    """
    Execute the time series prediction pipeline from loading configurations, preprocessing data,
    to running the machine learning model and saving the outputs.

    Parameters:
    - config_path (str): Path to the configuration directory.
    - model_name (str): Name of the machine learning model to be used.
    - config_log_filename (str): Name of the log configuration file. Default None

    Returns:
    - None: This function is designed to process the pipeline and does not return a value.

    Examples:
    >>> run_time_series_prediction_pipeline("../config", "KNeighborsRegressor", "config_logs")
    """
    # Load configuration parameters from JSON files

    config_model_parameters = load_json(os.path.join(config_path, "models_parameters"), model_name)

    # Set up logging configuration
    if config_log_filename: 
        config_logs_parameters = load_json(config_path, config_log_filename) 
        own_logger = OwnLogger(
            log_rotation=config_logs_parameters["log_rotation"],
            max_bytes=config_logs_parameters["max_bytes"],
            backup_count=config_logs_parameters["backup_count"],
            when=config_logs_parameters["when"]
        )
        logger = own_logger.get_logger()
    else:
        logger = None

    # Import data from the database
    df = data_importer(
        automatic_importation=config_model_parameters["data_importer_automatic_importation"],
        creds_path=Path(config_model_parameters["data_importer_creds_path"]),
        path_instants_data_saved=Path(config_model_parameters["data_importer_path_instants_data_saved"]),
        database=config_model_parameters["data_importer_database"],
        query=config_model_parameters["data_importer_query"],
        save_importation=config_model_parameters["data_importer_save_importation"],
        file_name=config_model_parameters["data_importer_file_name"],
        logger=logger  
    )

    # Prepare DataFrame from database
    df = prepare_dataframe_from_db(
        df=df,
        cols_for_query=config_model_parameters["prepare_dataframe_from_db_cols_for_query"],
        logger=logger
    )

    # Process time series data: resample and interpolate
    df_resampled_interpolated = process_time_series_data(
        df=df,
        resample_freq=config_model_parameters["preprocess_time_series_data_resample_freq"],
        aggregation_func=config_model_parameters["preprocess_time_series_data_aggregation_func"],
        method=config_model_parameters["preprocess_time_series_data_method"],
        outlier_cols=config_model_parameters["preprocess_time_series_data_outlier_cols"],
        logger=logger
    )

    # Pivot and rename columns for uniformity
    df_preprocessed = pd.pivot_table(
        df_resampled_interpolated.reset_index()[["timestamp", "id_device", "id_variable", "value"]],
        index=["timestamp", "id_device"],
        columns=["id_variable"]
    ).reset_index()

    # Flatten the MultiIndex for columns
    df_preprocessed.columns = ["timestamp", "id_device", "y", "temperatura", "humedad", "tvoc", "presion", "siaq", "diaq"]


    # Model parameters 
    ini_train = config_model_parameters["ini_train"]
    fin_train = config_model_parameters["fin_train"]
    fin_test = config_model_parameters["fin_test"]
    name_time_column = config_model_parameters["name_time_column"]
    name_id_sensor_column = config_model_parameters["name_id_sensor_column"]
    id_device = config_model_parameters["id_device"]
    names_objective_variable = config_model_parameters["names_objective_variable"]
    predictor = config_model_parameters["predictor"]
    X_name_features = config_model_parameters["X_name_features"] if config_model_parameters["X_name_features"] is not None else list(set(df_preprocessed.columns)-set(['y','timestamp','id_device']))
    Y_name_features = config_model_parameters["Y_name_features"]
    n_lags = config_model_parameters["n_lags"]
    n_predictions = config_model_parameters["n_predictions"]
    lag_columns = config_model_parameters["lag_columns"] if config_model_parameters["lag_columns"] is not None else list(set(df_preprocessed.columns)-set(['y','timestamp','id_device'])) + ["y"]
    scale_in_preprocessing = config_model_parameters["scale_in_preprocessing"]
    path_to_save_model = config_model_parameters["path_to_save_model"]
    path_to_save_model = Path(config_model_parameters["path_to_save_model"])
    save_preprocessing = config_model_parameters["save_preprocessing"]
    folder_name_model = config_model_parameters["folder_name_model"]
    folder_name_preprocessed_data = config_model_parameters["folder_name_preprocessed_data"]
    now_str = f"execution-time-{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    folder_name_time_execution = config_model_parameters["folder_name_time_execution"] if config_model_parameters["folder_name_time_execution"] is not None else now_str
    machine_learning_model_args = config_model_parameters["machine_learning_model_args"]

    model_machine_learning = create_model_machine_learning_algorithm(
        tidy_data = df_preprocessed,
        ini_train = ini_train,
        fin_train = fin_train,
        fin_test = fin_test,
        id_device = id_device,
        model_sklearn_name = predictor,
        X_name_features = X_name_features,
        Y_name_features = Y_name_features,
        n_lags = n_lags,
        n_leads = n_predictions,
        lag_columns = lag_columns,
        lead_columns = Y_name_features,
        scale_in_preprocessing = scale_in_preprocessing,
        name_time_column = name_time_column,
        name_id_sensor = name_id_sensor_column,
        save_preprocessing = save_preprocessing,
        path_to_save_model= path_to_save_model,
        folder_name_model= folder_name_model,
        folder_name_time_execution = folder_name_time_execution,
        folder_name_preprocessed_data = folder_name_preprocessed_data,
        machine_learning_model_args = machine_learning_model_args
    )

    return model_machine_learning
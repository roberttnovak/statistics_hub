import datetime
from itertools import product
import json
import logging
import os
import time
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
from ConfigManager import ConfigManager
from OwnLog import OwnLogger
from PersistenceManager import PersistenceManager, persist_model_to_disk_structure
from cleaning import prepare_dataframe_from_db, preprocess_and_standardize_dataframe, process_time_series_data
from own_utils import execute_concurrently, flatten_nested_dict, get_all_args, get_deepest_keys_values, load_json
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





def get_train_test(
    df, 
    ini_train,
    fin_train,
    fin_test,
    name_time_column="timestamp",
    localize_tz = None
    ):
    """
    Splits the DataFrame df into a training set and a test set based on the provided dates.

    Parameters:
        df (pd.DataFrame): DataFrame to be split.
        ini_train (str): Initial date of the training set in the format 'YYYY-MM-DD'.
        fin_train (str): Final date of the training set in the format 'YYYY-MM-DD'.
        fin_test (str): Final date of the test set in the format 'YYYY-MM-DD'.
        name_time_column (str, optional): Name of the column containing the dates. Defaults to "timestamp".
        localize_tz (str, optional): Timezone to localize the timestamps. If None, removes timezone information. 
                                     Example: "UTC" or "Europe/Madrid". Defaults to None.

    Returns:
        dict: A dictionary containing two DataFrames: df_train and df_test, corresponding to the training and test sets respectively.

    Examples:
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.date_range(start="2020-01-01", periods=10, tz="UTC"),
        ...     "value": range(10)
        ... })
        >>> result = get_train_test(df, "2020-01-01", "2020-01-05", "2020-01-10", localize_tz=None)
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
        >>> result = get_train_test(df, "2020-01-01", "2020-01-05", "2020-01-10", localize_tz="Europe/Madrid")
        >>> print(result['df_train'].head())
           timestamp  value
        0 2020-01-01 01:00:00+01:00      0
        1 2020-01-02 01:00:00+01:00      1

    Raises:
        ValueError: If ini_train, fin_train, or fin_test are not ordered as ini_train < fin_train < fin_test.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Ensure date strings are converted to datetime objects
    df[name_time_column] = pd.to_datetime(df[name_time_column])
    ini_train = pd.to_datetime(ini_train)
    fin_train = pd.to_datetime(fin_train)
    fin_test = pd.to_datetime(fin_test)

    # Handle timezone localization
    if localize_tz:
        df[name_time_column] = df[name_time_column].dt.tz_localize(localize_tz, ambiguous='NaT', nonexistent='NaT')
        ini_train = ini_train.tz_localize(localize_tz, ambiguous='NaT', nonexistent='NaT')
        fin_train = fin_train.tz_localize(localize_tz, ambiguous='NaT', nonexistent='NaT')
        fin_test = fin_test.tz_localize(localize_tz, ambiguous='NaT', nonexistent='NaT')
    else:
        df[name_time_column] = df[name_time_column].dt.tz_localize(None)
        ini_train = ini_train.tz_localize(None)
        fin_train = fin_train.tz_localize(None)
        fin_test = fin_test.tz_localize(None)

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





class PreprocessColumns(BaseEstimator, TransformerMixin):
    def __init__(self, X_name_features: Union[str, List[str]], Y_name_features: Union[str, List[str]]):
        """
        Initializes the transformer with the names of the features.
        
        Parameters:
        - X_name_features: Names of the input features.
        - Y_name_features: Names of the target features.
        """
        self.X_name_features = [X_name_features] if isinstance(X_name_features, str) else (X_name_features or [])
        self.Y_name_features = [Y_name_features] if isinstance(Y_name_features, str) else (Y_name_features or [])

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
        measure_time: bool = False
) -> Dict[str, Dict]:
    """
    Preprocesses a DataFrame based on specified transformations for machine learning.

    This function encapsulates the instantiation and application of various data transformations 
    including column selection, scaling, lag and lead feature creation, with optional timing of each step.

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
    measure_time : bool, optional
        Whether to measure the time taken for each preprocessing step, by default False.

    Returns
    -------
    Dict[str, Dict]
        Dictionary containing the order, DataFrame, transformer instance, and optionally time for each step.
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

    # Initialize preprocess dictionary
    preprocess = {}

    # Processing DataFrame step by step with timing
    df_processed = df
    for transformer_name in transformers_order:
        transformer_instance = transformers_dict[transformer_name]

        start_time = time.time() if measure_time else None
        df_processed = transformer_instance.fit_transform(df_processed)
        end_time = time.time() if measure_time else None

        # Collect the results
        preprocess[transformer_name] = {
            "order": order_dict[transformer_name],
            "df": getattr(transformer_instance, 'df_preprocessed', None),
            "transformer": transformer_instance
        }
        if measure_time:
            preprocess[transformer_name]["execution_time_in_seconds"] = end_time - start_time

    return preprocess

#TODO: Revisar la lógica de las diferencias entre Y_name_features y lead_columns
#TODO: Para el tiempo de ejecución, está desglosado en preprocess_machine_learning_algorithm al máximo, pero, de momento, estoy
# guardando solo el acumulado. En algún momento agregar esa lógica. 
def create_model_machine_learning_algorithm(
        tidy_data,
        ini_train,
        fin_train,
        fin_test,
        model_sklearn_name,
        X_name_features,
        Y_name_features,
        n_lags,
        n_leads,
        lag_columns,
        lead_columns,
        scale_in_preprocessing=True,
        name_time_column="timestamp",
        save_preprocessing=True,
        save_parameters_csv = False,
        path_to_save_model=None,
        folder_name_model=None,
        folder_name_time_execution="execution-time-no-defined",
        folder_name_preprocessed_data="preprocessed-data-to-use-in-model",
        machine_learning_model_args=None,
        measure_time=False, 
        metadata_preprocessing = {},
        logger: logging.Logger = None,
        kwargs_save_object: Dict = {},
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
    measure_time : bool, optional
        Whether to measure the time taken for each preprocessing step. Default is False.
    metadata_preprocessing : dict, optional
        Dictionary containing metadata for preprocessing. Default is None.
    logger : logging.Logger, optional
        Logger object for logging information. Default is None.
    kwargs_save_object : dict, optional
        Additional arguments for saving objects associated with de model. 
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
    args_and_values_of_function = locals()

    for arg, value in args_and_values_of_function.items():
        if isinstance(value, pd.DataFrame):
            # If a DataFrame is found, convert it to a string 
            args_and_values_of_function[arg] = value.head().to_string(max_cols=None, show_dimensions=True)
        elif isinstance(value, Path):
            args_and_values_of_function[arg] = str(value)

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
        folder_name_time_execution=folder_name_time_execution,
        rename_if_exists = True
    )


    pm.remove_flag("training-done")

    #TODO: Borrar este save_preprocessed_data pq ya se hace en otro lado. PRimero comporbar que no se rompe nada
    # pm.save_preprocessed_data(
    #     preprocessed_data=tidy_data, 
    #     folder_name_preprocessed_data=folder_name_preprocessed_data,
    #     name = "tidy_data",
    #     **kwargs_save_object
    # )

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

        preprocessing = preprocess_machine_learning_algorithm(
            df=df,
            X_name_features=X_name_features,
            Y_name_features=Y_name_features,
            n_lags=n_lags,
            n_leads=n_leads,
            lag_columns=lag_columns,
            lead_columns=lead_columns,
            measure_time=measure_time
        )
        df_last_step = get_last_df_step_preprocessed(preprocessing)
        df_X = get_df_X(df_last_step, X_name_features=X_name_features, Y_name_features=Y_name_features)
        df_Y = get_df_Y(df_last_step, Y_name_features=Y_name_features)
        execution_times = {transformer : values.get("execution_time_in_seconds", None) for transformer, values in preprocessing.items()}
        return preprocessing, df_X, df_Y, execution_times

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

    # Use kwargs if machine_learning_model_args is None
    machine_learning_model_args = kwargs if machine_learning_model_args is None else machine_learning_model_args

    # Preprocessing
    # Dictionary to store the preprocessing time for each step
    execution_times = {}

    start_time = time.time()
    preprocessing_train, df_train_X, df_train_Y, execution_times_train = preprocess_data(df_train, X_name_features, Y_name_features, n_lags, n_leads, lag_columns, lead_columns)
    end_time = time.time()
    execution_times["execution_times"] = {}
    execution_times["execution_times"]["preprocessing_train"] = {}
    execution_times["execution_times"]["preprocessing_train"]["total"] = end_time - start_time
    execution_times["execution_times"]["preprocessing_train"]["details"] = execution_times_train

    start_time = time.time()
    preprocessing_test, df_test_X, df_test_Y, execution_times_test = preprocess_data(df_test, X_name_features, Y_name_features, n_lags, n_leads, lag_columns, lead_columns)
    end_time = time.time()
    execution_times["execution_times"] = {}
    execution_times["execution_times"]["preprocessing_test"] = {}
    execution_times["execution_times"]["preprocessing_test"]["total"] = end_time - start_time
    execution_times["execution_times"]["preprocessing_test"]["details"] = execution_times_test

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

    start_time = time.time()
    model.fit(df_train_X, df_train_Y)
    end_time = time.time()
    execution_times["execution_times"]["training"] = end_time - start_time


    # Metadata: combining default arguments with provided arguments
    #1st, check if exist previous metadata
    # pm.load_metadata()
    all_args_by_default = get_all_args(RegressorClass.__init__)
    metadata = {**all_args_by_default, **args_and_values_of_function, **execution_times, **metadata_preprocessing}

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
            "df_test": df_test,
            "tidy_data": tidy_data
        }

        df_parameters_with_master_info = get_df_of_paramaters_executed_in_model(
        model_name = folder_name_model,
        metadata = metadata,
        config_path = "../config",
        model_base_path = path_to_save_model, 
        folder_name_range_train =  folder_name_range_train,
        folder_name_time_execution = folder_name_time_execution,
        )


        persist_model_to_disk_structure(
            path_to_save_model = path_to_save_model,
            folder_name_model = folder_name_model,
            folder_name_range_train = folder_name_range_train,
            folder_name_time_execution = folder_name_time_execution,
            model = model,
            metadata = metadata,
            scaler = scaler,
            df_parameters =  df_parameters_with_master_info,
            folder_name_preprocessed_data = folder_name_preprocessed_data if save_preprocessing else None,
            preprocessed_data = preprocessed_data,
            additional_persistable_objects_to_save=None,
            **kwargs_save_object
        )

    return {
        "model": model,
        "preprocessing_train": preprocessing_train,
        "preprocessing_test": preprocessing_test,
        "df_train_X": df_train_X,
        "df_test_X": df_test_X,
        "scaler": scaler,
        "execution_times": execution_times     
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
    flag_content: str = "",
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
    flag_content (str, optional): Content inside .txt flag
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
            overwrite = overwrite_predictions,
        )
        pm.save_predictions(
            predictions = df_predictions_test,
            folder_name_predictions = folder_name_predictions,
            name = "predictions-test",
            overwrite = overwrite_predictions
        )
        #ToDo: Borrar los comentarios de abajo después de testar que todo funciona correctamente
        # Create a file to indicate that the predictions have been completed successfully
        # path_flag_predictions = os.path.join(
        #     path_to_save_model,
        #     folder_name_model,
        #     folder_name_range_train,
        #     folder_name_time_execution
        # )
        # with open(os.path.join(path_flag_predictions, flag_predictions_done), 'w') as f:
        #     f.write('Predictions completed successfully.\n')
        pm.create_flag(flag_name=flag_predictions_done, content = flag_content, sub_folder=None)
    
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
    save_name = None,
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
    save_name (str, optional): Custom name for the saved summary data file. If None, a default name is generated. Default is None.
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
        if save_name is None:  # Use default naming convention if save_name is not provided
            save_name = "evaluations"
            if show_args_in_save:
                arg_name = "___".join([f"{arg}-{value}" for arg, value in kwargs_summarise_mae.items()])
                save_name = f"{arg_name}"
        # The logic for saving the summary data can be added here
        pm.save_evaluation_data(predictions_train_test_summarise, name = save_name , folder_name_evaluation=folder_name_evaluation)
        pm.create_flag(flag_name=flag_name, content = flag_content,sub_folder=flag_subfolder)
    
    # Return the summary data
    return predictions_train_test_summarise

def run_time_series_prediction_pipeline(
        config_user: str, 
        model_name: str, 
        file_data_chosen: str,
        config_log_filename: str = None):
    """
    Execute a time series prediction pipeline, encompassing configuration loading, data preprocessing,
    machine learning model execution, and output saving. It supports custom configurations for model
    parameters, logging, and data preparation.

    Parameters:
    - config_user (str): Path to the user's configuration directory, which includes model parameters and common settings.
    - model_name (str): Identifier for the specific machine learning model to use, as defined in the configuration.
    - file_data_chosen (str): Filename of the data to process, located within the user's data directory.
    - config_log_filename (str, optional): Filename for log configuration settings. If not provided, logging is disabled.

    The pipeline includes steps for:
    1. Loading model-specific and common parameters from configuration files.
    2. Preparing and preprocessing the input time series data.
    3. Creating and training the specified machine learning model with the prepared data.
    4. Optionally setting up a logging mechanism to track the pipeline's execution.

    Returns:
    - The trained machine learning model instance, ready for making predictions or further analysis.
    """
    
    # Load the configuration of the models parameters (the last parameters entered by the user are saved in config file)
    config_models_parameters = load_json(
        folder_path = os.path.normpath(os.path.join(config_user,"config","models_parameters")), 
        json_filename = "parameters"
    )

    # We access config of the specific model 
    config_model_parameters = config_models_parameters[model_name]

    # Load the common arguments of all models (folders to save the models, etc)
    config_model_common_parameters = load_json(
        folder_path = os.path.normpath(os.path.join(config_user,"config", "models_parameters","common_parameters")), 
        json_filename = "common_parameters"
    )  

    # Get the specific parameters of the sklearn regressor entered by the user 
    machine_learning_model_args = config_model_parameters["regressor_params"]

    # We save the the specific parameters of the sklearn regressor so we dont need anymore
    del config_model_parameters["regressor_params"]

    # config_model_parameters are classified so we need to access to the deepest keys
    config_model_parameters = get_deepest_keys_values(config_model_parameters)

    # Get in a single dictionary all parameters in config file to use them when create the model
    config_model_parameters = {**config_model_parameters, **config_model_common_parameters}

    # Set up logging configuration
    if config_log_filename: 
        config_logs_parameters = load_json(
            folder_path = os.path.normpath(os.path.join(config_user,"config","config_logs")), 
            json_filename = config_log_filename
        ) 
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
    df = pd.read_csv(os.path.normpath(os.path.join(config_user,"data",file_data_chosen)))

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
    model_sklearn_name = model_name
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

    model_machine_learning = create_model_machine_learning_algorithm(
        tidy_data = df_preprocessed,
        ini_train = ini_train,
        fin_train = fin_train,
        fin_test = fin_test,
        id_device = id_device,
        model_sklearn_name = model_sklearn_name,
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

#TODO: ¿Mover al PersistenceManager? Quizás tenga que readaptar alguna cosa para la concurrencia
def get_df_of_paramaters_executed_in_model(
        model_name, 
        config_path, 
        model_base_path, 
        folder_name_range_train, 
        folder_name_time_execution,
        metadata,
        save = False,
        **kwargs_save_object
    ):
    """
    Prepares and merges metadata, including classification, legible names, and descriptions of parameters (master information),
    for a specific machine learning model using configuration files and persistence manager data.
    
    Parameters:
    - model_name: str, the name of the machine learning model (e.g., "SVR").
    - config_path: str, the path to the configuration directory (master information).
    - model_base_path: str, the path to the base directory where model data is stored.
    - folder_name_range_train: str, the name of the folder within the model_base_path that contains training range data.
    - folder_name_time_execution: str, the name of the folder within the model_base_path that contains execution time data.
    - metadata: dict, a dictionary containing metadata information about the model.
    - save: bool, whether to save the resulting DataFrame to disk.

    Returns:
    - df_initialisation_of_parameters_of_model: pandas DataFrame, containing merged information about model parameters, their classifications, legible names, descriptions, and the actual values used in the model.

    Example:
    get_df_of_paramaters_executed_in_model(
    model_name = "SVR", 
    config_path = "../config",
    model_base_path = "tenants/admin/models", 
    folder_name_range_train =  "initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0",
    folder_name_time_execution = "execution-time-2024_02_15_10_46_12",
    save = False
    )
    """
    
    # Initialize the configuration manager with the given config path
    config_manager_metadata_parameters = ConfigManager(config_path)

    # Load various configurations related to model parameters
    classification_of_parameters = config_manager_metadata_parameters.load_config(
        "classification_of_parameters", 
        subfolder="models_parameters/metadata"
    )
    legible_names_of_own_parameters = config_manager_metadata_parameters.load_config(
        "legible_names_of_own_parameters", 
        subfolder="models_parameters/metadata"
    )
    descriptions_of_own_parameters = config_manager_metadata_parameters.load_config(
        "descriptions_of_own_parameters", 
        subfolder="models_parameters/metadata"
    )
    all_sklearn_regressors_with_all_info = config_manager_metadata_parameters.load_config(
        "all_sklearn_regressors_with_all_info", 
        subfolder="models_parameters/metadata"
    )

    # Extract the specific model's regressor information
    model_regressor_info = all_sklearn_regressors_with_all_info[model_name]["parameters_info"]

    # Prepare dataframes for classification, legible names, and descriptions of parameters
    df_classification_of_each_parameter = pd.DataFrame(
        [[classification, parameter] for classification, parameters in classification_of_parameters.items() for parameter in parameters],
        columns=["classification_of_parameter", "parameter"]
    )
    df_legible_names_of_own_parameter = pd.DataFrame(
        [[parameter, legible_name] for parameter, legible_name in legible_names_of_own_parameters.items()],
        columns=["parameter", "legible_parameter"]
    )
    df_description_of_own_parameters = pd.DataFrame(
        [[parameter, description] for parameter, description in descriptions_of_own_parameters.items()],
        columns=["parameter", "description"]
    )

    # Prepare a dataframe for sklearn parameters with descriptions
    df_sklearn_parameters = pd.DataFrame(model_regressor_info)[["parameter", "description"]]
    df_sklearn_parameters["legible_parameter"] = df_sklearn_parameters["parameter"]

    # Merge dataframes to include own parameters with their classifications, legible names, and descriptions
    df_own_parameters = df_classification_of_each_parameter.query(
        "classification_of_parameter.isin(['split_train_test_args','preprocess_time_series_data_args','time_serie_args'])"
    )\
    .reset_index(drop=True)\
    .merge(df_legible_names_of_own_parameter, on="parameter")\
    .merge(df_description_of_own_parameters, on="parameter")[["parameter", "legible_parameter", "description"]]

    # Concatenate sklearn and own parameters dataframes
    df_parameters = pd.concat([df_sklearn_parameters, df_own_parameters], axis=0).reset_index(drop=True)

    # Correct parameter name if necessary
    df_parameters.loc[df_parameters["parameter"] == "n_predictions", "parameter"] = "n_leads"

    pm = PersistenceManager(
        base_path = model_base_path,
        folder_name_model = model_name,
        folder_name_range_train = folder_name_range_train,
        folder_name_time_execution = folder_name_time_execution
    )

    machine_learning_model_args = metadata.get("machine_learning_model_args", {})

    # Update metadata with machine learning model arguments
    metadata.update(machine_learning_model_args)

    # Metadata related to preprocessing
    metadata_preprocessing = metadata.get("metadata_preprocessing", {})
    df_metadata_preprocessing = pd.DataFrame(list(metadata_preprocessing.items()), columns=['parameter', 'value'])
    df_metadata_preprocessing["parameter"] = "preprocessing_" + df_metadata_preprocessing["parameter"]

    # Metadata related to time preprocessing 
    metadata_execution_times = metadata.get("execution_times", {})
    metadata_execution_times_flatten = flatten_nested_dict(metadata_execution_times)
    df_metadata_execution_times = pd.DataFrame(list(metadata_execution_times_flatten.items()), columns=['parameter', 'value'])
    df_metadata_execution_times["parameter"] = "execution_times_" + df_metadata_execution_times["parameter"]

    # Merge all information of model 
    df_values_of_model = pd.DataFrame(
        [[k, v] for k, v in metadata.items() if k not in ["metadata_preprocessing", "execution_times"]],
        columns=["parameter", "value"]
    )

    # df_values_of_model = pd.concat([df_values_of_model,df_metadata_preprocessing,df_metadata_execution_times], axis=0)

    # Merge parameters dataframe with values dataframe to get initialization information for each parameter of the model and descriptions
    df_initialisation_of_parameters_of_model = df_parameters.merge(df_values_of_model, how="inner", left_on="parameter", right_on="parameter")
    df_initialisation_of_parameters_of_model = df_initialisation_of_parameters_of_model[["legible_parameter","description","value"]]\
    .rename(columns={"legible_parameter": "parameter"})

    #TODO: Meter la description de cada parameter de lo nuevo del metadata 
    df_initialisation_of_parameters_of_model = pd.concat([df_initialisation_of_parameters_of_model,df_metadata_preprocessing,df_metadata_execution_times], axis=0)

    if save:
        pm.save_dataset(df_initialisation_of_parameters_of_model, "parameters", **kwargs_save_object)


    return df_initialisation_of_parameters_of_model

def load_evaluation_data_for_models(user):
    """
    Loads and consolidates evaluation and parameter data for all models associated with a specific user.

    This function iterates through the hierarchy of models stored in a directory structure based on the user, 
    training range, and execution time. For each model, it attempts to load both evaluation data and parameter 
    data from predefined file formats. All successfully loaded data is then concatenated into separate pandas 
    DataFrames for evaluation data and parameter data, respectively. If any errors occur during data loading 
    (e.g., file not found, data format issues), those errors are logged, and the function continues with the 
    next model.

    Parameters:
    - user (str): The username or identifier for which models' data should be loaded. This is used to construct 
                  the base path for model data storage.

    Returns:
    - dict: A dictionary containing two pandas DataFrames:
        - 'df_evaluations': A DataFrame containing all concatenated evaluation data for the models. If no evaluation 
                            data could be loaded successfully, an empty DataFrame is returned.
        - 'df_parameters': A DataFrame containing all concatenated parameter data for the models. If no parameter 
                           data could be loaded successfully, an empty DataFrame is returned.

    Note:
    - The function relies on a specific directory and file naming convention. Ensure that the model data is 
      organized according to this convention for the function to work correctly.
    - Any exception encountered during the loading of individual model evaluation or parameter data is caught 
      and logged, but does not halt the execution of the function for subsequent models.
    """
    pm = PersistenceManager(base_path=f"tenants/{user}/models")
    model_range_train_execution_time = pm.get_models_hierarchy_list()

    evaluations = []
    parameters = []

    for model, range_train, execution_time in model_range_train_execution_time:
        try:
            pm_instance = PersistenceManager(
                base_path=f"tenants/{user}/models",
                folder_name_model=model,
                folder_name_range_train=range_train,
                folder_name_time_execution=execution_time
            )
            data_evaluation = pm_instance.load_evaluation_data(name='freq-None___group_by_timestamp-False')
            data_evaluation["training_range"] = range_train
            data_evaluation["execution_time"] = execution_time
            evaluations.append(data_evaluation)  

            data_parameters = pm_instance.load_dataset("parameters")
            data_parameters["training_range"] = range_train
            data_parameters["execution_time"] = execution_time
            parameters.append(data_parameters)

        except Exception as e:
            print(f"Error loading data for model {model}, range {range_train}, execution time {execution_time}: {e}")

    if evaluations:
        df_evaluations = pd.concat(evaluations, axis=0)
    else:
        df_evaluations = pd.DataFrame()

    if parameters:
        df_parameters = pd.concat(parameters, axis=0)
    else:
        df_parameters = pd.DataFrame()

    return {"df_evaluations":df_evaluations, "df_parameters":df_parameters}

#TODO: Dejar la parte de la distribution mejor con np.random y más cosas que tengan sentido para controlar
def generate_cv_grid_regressor_sklearn(
    hyperparameter_dict,
    distribution="uniform",
    distribution_per_param=None,
    max_length_numerical=10,
    max_length_categorical=5,
    param_lengths={},
    custom_limits={},
    seed=42,
    selected_params=None,
    forced_types=None
):
    """
    Generate a grid of hyperparameter values for Cross Validation (CV) based on the given domains.

    Parameters
    ----------
    hyperparameter_dict : dict
        Dictionary containing the hyperparameters and their domains for the specified regressor.
    distribution : str, optional
        Default distribution to use for generating numerical values. Options are "uniform" or "log".
        Default is "uniform".
    distribution_per_param : dict, optional
        Dictionary specifying the distribution for specific hyperparameters.
        Example: {"alpha": "log", "max_iter": "uniform"}.
    max_length_numerical : int, optional
        Maximum number of values to generate for numerical parameters if `param_lengths` is not provided.
        Default is 10.
    max_length_categorical : int, optional
        Maximum number of values to generate for categorical parameters if `param_lengths` is not provided.
        Default is 5.
    param_lengths : dict, optional
        Dictionary specifying the maximum number of values to generate for specific hyperparameters.
    custom_limits : dict, optional
        Dictionary specifying custom min and max values for specific hyperparameters with infinite ranges.
        Example: {"alpha": [1e-6, 1e6], "max_iter": [1, 1000]}.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    selected_params : list, optional
        List of hyperparameter names to include in the grid. If None, all parameters are included.
    forced_types : dict, optional
        Dictionary specifying the forced type for specific hyperparameters.
        Example: {"max_iter": int, "n_estimators": int}.

    Returns
    -------
    dict
        A dictionary with hyperparameter names as keys and lists of generated values as values.
    """
    np.random.seed(seed)

    cv_grid = {}
    values = []

    for param, details in hyperparameter_dict.items():

        values = []

        domain = details["domain"]
        domain_type = details["domain_type"]

        if selected_params and param not in selected_params:
            continue
        
        if param in custom_limits and len(custom_limits[param]) == 0 : 
            continue

        length = param_lengths.get(param, max_length_numerical if domain_type == "interval" else max_length_categorical)
        param_distribution = distribution_per_param.get(param, distribution) if distribution_per_param else distribution

        if domain_type == "interval":
            start, end = domain
            if param in custom_limits and len(custom_limits[param])!=0:
                start, end = custom_limits[param]
            if start == "-inf":
                start = 1e-6
            if end == "inf":
                end = 1e6
            values = np.linspace(float(start), float(end), length).tolist() if param_distribution == "uniform" else np.logspace(np.log10(float(start) + 1e-6), np.log10(float(end)), length).tolist()
        elif domain_type == "categorical":
            values = domain[:length]
        elif domain_type == "categorical_or_interval":
            if all(isinstance(x, (int, float)) for x in domain):
                start, end = domain
                if param in custom_limits:
                    start, end = custom_limits[param]
                values = np.linspace(float(start), float(end), length).tolist() if param_distribution == "uniform" else np.logspace(np.log10(float(start) + 1e-6), np.log10(float(end)), length).tolist()
            else:
                if None not in domain :
                    values = domain[:length]
                else:
                    # Expected custom_limits like [None, [min,max]]
                    start, end = custom_limits[param][1]
                    values = np.linspace(float(start), float(end), length).tolist() if param_distribution == "uniform" else np.logspace(np.log10(float(start) + 1e-6), np.log10(float(end)), length).tolist()
        #TODO: Improve best practices with this
        elif domain_type == "categorical_or_array":
            pass
        elif domain_type == "array":
            pass
        elif domain_type == "categorical_or_instance":
            pass
        elif domain_type == "ignore_for_grid_cv":
            pass
        else:
            raise ValueError(f"Unsupported parameter type: {domain_type}")

        # Apply forced type if specified
        if forced_types and param in forced_types:
            target_type_invalid = ["object"]
            if forced_types[param] not in target_type_invalid:
                target_type = eval(forced_types[param])
                try:
                    values = [target_type(value) if value is not None else None for value in values]
                except (ValueError, TypeError) as e:
                    # Some datatypes are int or other datatypes like auto. This is to fix this temporally
                    # Pending robust implementation 
                    if "auto" in values:
                        pass 
                    else: 
                        raise ValueError(f"Could not convert values for {param} to {target_type}: {e}")
            else:
                cv_grid[param] = values
                return cv_grid

        cv_grid[param] = values

    return cv_grid

def recommend_distribution_and_range(regressor_name, param_name, domain, domain_type):
    """
    Recommends a distribution and range for a given parameter of a scikit-learn regressor.

    This function analyzes the parameter name, its domain, and type to provide:
      - A recommended sampling distribution for hyperparameter tuning (e.g., "uniform", "log", "categorical").
      - A range or set of values tailored to the parameter's usage context.
      
    While `regressor_name` is currently unused, it is included to allow future customization 
    for specific regressors. For example, certain parameters might have slightly different 
    ranges depending on the regressor.

    Parameters
    ----------
    regressor_name : str
        The name of the regressor (e.g., "RandomForestRegressor", "SVR"). 
        Currently unused but reserved for future enhancements.
    param_name : str
        The name of the parameter (e.g., "alpha", "n_estimators", "learning_rate").
    domain : list
        The allowed domain of the parameter. Typically a two-element list for intervals 
        (e.g., `[0.0, "inf"]`) or a list of discrete values for categorical parameters 
        (e.g., `["auto", "sqrt", "log2", None]`).
    domain_type : str
        The type of domain, indicating how the parameter values are structured:
        - "interval": Numeric range, potentially unbounded (e.g., `[0.0, "inf"]`).
        - "categorical": Discrete choices (e.g., `[True, False]`, `["linear", "poly"]`).
        - Other combinations like "categorical_or_interval", "categorical_or_callable".

    Returns
    -------
    dict
        A dictionary with the following keys:
        - recommended_distribution : str
            The suggested sampling distribution:
            - "uniform": Uniform sampling over a range.
            - "log": Logarithmic sampling for parameters spanning multiple orders of magnitude.
            - "categorical": Sampling from a set of discrete values.
            - "none": Exclude this parameter from tuning.
        - recommended_range : list
            The recommended range or set of values to sample. For "interval" domains, 
            this will be a numeric range (e.g., `[1e-6, 1e3]`). For "categorical" domains, 
            it will be the list of discrete values.

    Notes
    -----
    - This function uses heuristics based on common parameter names and usage patterns 
      in scikit-learn regressors. It is opinionated and should be adapted for specific needs.
    - Parameters like `random_state`, `verbose`, and `n_jobs` are typically excluded from 
      tuning and assigned a distribution of "none".
    - The inclusion of `regressor_name` allows for future customization or regressor-specific 
      overrides, but it is currently unused in the logic.

    Examples
    --------
    >>> recommend_distribution_and_range("RandomForestRegressor", "n_estimators", [1, "inf"], "interval")
    ('uniform', [50, 300])

    >>> recommend_distribution_and_range("SVR", "C", [0.0, "inf"], "interval")
    ('log', [1e-3, 1e3])

    >>> recommend_distribution_and_range("GradientBoostingRegressor", "learning_rate", [0.0, "inf"], "interval")
    ('log', [1e-4, 1.0])

    >>> recommend_distribution_and_range("RandomForestRegressor", "bootstrap", [True, False], "categorical")
    ('categorical', [True, False])

    OTHER IMPORTANT NOTES:
    1) Many parameters like 'random_state', 'verbose', 'n_jobs', etc. are typically NOT tuned,
       so we often mark them as "none" or provide a minimal range/option.
    2) Boolean or categorical parameters are assigned a "categorical" distribution with their discrete domain.
    3) The numeric ranges here are only heuristics. You can widen or narrow them depending on real use-cases.
    4) If the domain includes special values like None or "auto", you might exclude them or keep them as possible
       categorical choices, depending on your pipeline.
    5) If you see repeated logic for certain param patterns (e.g. "alpha", "lambda", "C", etc.), it's intentional
       because these typically benefit from a log-scale search with a specific range.
    6) For interval-based parameters in [0,1], "uniform" sampling is common. For thresholds or tolerances, "log" is common.
    """

    # Default distribution and range if we cannot match any specific pattern:
    recommended_distribution = "uniform"
    recommended_range = [0.0, 1.0]  # generic fallback

    result = {
    "recommended_distribution": recommended_distribution,  
    "recommended_range": recommended_range
    }

    # Extract potential numeric start/end for convenience
    # domain often looks like [something, "inf"] or [0.0, 1.0]. 
    if isinstance(domain, list) and len(domain) == 2:
        start, end = domain
    else:
        start, end = None, None

    # Helper to check if domain boundary is infinite
    def is_inf(x):
        return (isinstance(x, str) and (x.lower() == "inf" or x.lower() == "-inf"))

    # Convert param name to lowercase for matching
    p_lower = param_name.lower()
    
    # -------------------------------------------------------------------------
    #                 EXCLUDING SOME PARAMETERS
    # -------------------------------------------------------------------------
    # random_state / verbose / n_jobs / copy_X / n_jobs / etc. are typically not tuned.
    # Mark them as "none" to exclude from search if domain_type is interval or something else.
    # -------------------------------------------------------------------------
    exclude_params = {
        "random_state",  # usually fix or ignore
        "verbose",       # typically not tuned
        "n_jobs",        # typically a resource choice
        "nthread",       # not present here, but common in xgboost
        "thread_count",  # not present here, but example
        "copy_x",        # parameter rarely tuned
        "copy_x_train",  # parameter rarely tuned
        "check_inverse"  # for TransformedTargetRegressor
    }
    if p_lower in exclude_params:
        result["recommended_distribution"] = "none"
        result["recommended_range"] = []
        return result


    # -------------------------------------------------------------------------
    #                 CATEGORICAL OR BOOLEAN PARAMETERS
    # -------------------------------------------------------------------------
    # If the domain type is "categorical" or "categorical_or_X" and the domain is a list of discrete values:
    #  -> We do not create a numeric range. We'll keep them as "categorical" if we truly want to tune them.
    # If the domain is boolean, we also keep it as "categorical" with [True, False].
    # -------------------------------------------------------------------------
    if domain_type.startswith("categorical") and domain_type not in ["categorical_or_interval"]:
        # Some parameters that are purely boolean or small sets of discrete choices:
        # Example: "fit_intercept", "positive", "warm_start", "bootstrap", "oob_score", etc.
        # We keep them as "categorical" if they are booleans or small enumerations.

        # If you do NOT want to tune these at all, you could set:
        # recommended_distribution = "none"
        # recommended_range = []
        # But here we'll keep them in "categorical" for demonstration.
        result["recommended_distribution"] = "categorical"
        result["recommended_range"] = domain
        # We do not do any special numeric logic here. 
        # Return now if we consider all "categorical*" done:
        return result

    elif domain_type == "categorical_or_interval":
            # Example: domain = [None, 1, "inf"] for "max_leaf_nodes"
            # We want to keep None as a discrete choice, plus a numeric range for the rest.
            if None in domain:
                # Separate None from the numeric part
                domain_no_none = [d for d in domain if d is not None]
                # domain_no_none might look like [1, "inf"], for instance

                # Make sure it has two elements (start and end)
                if len(domain_no_none) == 2:
                    start_val, end_val = domain_no_none

                    # Helper to detect infinite boundaries
                    def is_inf_local(x):
                        return (isinstance(x, str) and (x.lower() == "inf" or x.lower() == "-inf"))

                    # Handle the upper boundary
                    if is_inf_local(end_val):
                        end_val = 50  # example upper bound
                    else:
                        # if it's not "inf" but a number, keep it as is
                        end_val = end_val if isinstance(end_val, (int, float)) else 50

                    # Define a minimum for start
                    if isinstance(start_val, (int, float)):
                        start_val = max(start_val, 1)
                    else:
                        start_val = 1

                    # We represent everything as "categorical" because we have a discrete option (None)
                    # plus a numeric interval. The recommended_range will include both:
                    recommended_distribution = "categorical"
                    recommended_range = [
                        None,            # discrete choice
                        [start_val, end_val]  # continuous range
                    ]

                    result["recommended_distribution"] = recommended_distribution
                    result["recommended_range"] = recommended_range
                    return result
                else:
                    # If it doesn't match the pattern [None, min, max], you can decide to skip tuning
                    result["recommended_distribution"] = "none"
                    result["recommended_range"] = []
                    return result
            else:
                # If there's no None but domain_type is still "categorical_or_interval",
                # you could treat it as a regular "interval" if all remaining values are numeric,
                # or do something else as a fallback:
                recommended_distribution = "uniform"
                recommended_range = [0.0, 1.0]
                result["recommended_distribution"] = recommended_distribution
                result["recommended_range"] = recommended_range
                return result

    # -------------------------------------------------------------------------
    #                 LOG-SCALE FOR CERTAIN REGULARIZATION / LR
    # -------------------------------------------------------------------------
    # Many regularization params (alpha, lambda, C) or learning rates often
    # need log-scale sampling in a typical range. We do partial overrides below.
    # -------------------------------------------------------------------------
    if ("alpha" in p_lower or "lambda" in p_lower) and domain_type == "interval":
        # e.g. alpha, alpha_1, alpha_2, lambda_1, lambda_2, ccp_alpha, threshold_lambda, ...
        recommended_distribution = "log"
        # For ccp_alpha (used in trees) we pick a smaller upper bound:
        if "ccp_alpha" in p_lower:
            recommended_range = [1e-7, 1.0]
        elif "threshold_lambda" in p_lower:
            # ARD or Bayesian parameters
            recommended_range = [1e-7, 1e3]
        else:
            recommended_range = [1e-7, 1e3]

        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range

        return result

    if p_lower == "c" and domain_type == "interval":
        # Typical for SVM-based regressors
        recommended_distribution = "log"
        recommended_range = [1e-3, 1e3]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "gamma" in p_lower and domain_type == "interval":
        # For SVM or Kernel-based
        recommended_distribution = "log"
        # If domain is [0.0, inf], let's pick a typical range
        recommended_range = [1e-4, 1e1]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "learning_rate" in p_lower and domain_type == "interval":
        # Used in gradient boosting, MLP, or similar
        recommended_distribution = "log"
        recommended_range = [1e-4, 1.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "eta0" in p_lower and domain_type == "interval":
        # This is the initial learning rate in SGD
        recommended_distribution = "log"
        recommended_range = [1e-5, 1.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 ITERATIONS-RELATED PARAMETERS
    # -------------------------------------------------------------------------
    # Common iteration parameters: max_iter, n_iter, n_estimators, etc.
    # Typically uniform search in a moderate range (e.g. [50, 300] for n_estimators).
    # -------------------------------------------------------------------------
    if any(x in p_lower for x in ["max_iter", "n_iter"]) and domain_type == "interval":
        # For robust usage, let's propose up to 2000
        recommended_distribution = "uniform"
        # If the domain is [1, inf], we pick [50, 1000] or [50, 2000] 
        recommended_range = [50, 1000]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "n_estimators" in p_lower and domain_type == "interval":
        recommended_distribution = "uniform"
        recommended_range = [50, 300]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "max_depth" in p_lower and domain_type == "interval":
        # Usually small range for tree-based methods
        recommended_distribution = "uniform"
        recommended_range = [1, 30]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "max_bins" in p_lower and domain_type == "interval":
        # HistGradientBoosting typically allows a wide range. 
        # We'll use [32, 255] as an example.
        recommended_distribution = "uniform"
        recommended_range = [32, 255]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 SAMPLING-RELATED PARAMETERS
    # -------------------------------------------------------------------------
    # For min_samples_split, min_samples_leaf, etc. we often do a small integer range (uniform).
    # For fraction-based parameters (like subsample, min_weight_fraction_leaf, etc.),
    # we do uniform in [0.0, 1.0], sometimes narrower.
    # -------------------------------------------------------------------------
    if "min_samples_split" in p_lower and domain_type == "interval":
        recommended_distribution = "uniform"
        # if the domain is [2, inf], let's define up to 50
        recommended_range = [2, 50]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "min_samples_leaf" in p_lower and domain_type == "interval":
        recommended_distribution = "uniform"
        recommended_range = [1, 20]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "min_weight_fraction_leaf" in p_lower and domain_type == "interval":
        # Usually [0.0, 0.5]
        recommended_distribution = "uniform"
        recommended_range = [0.0, 0.5]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "subsample" in p_lower and domain_type == "interval":
        # Typically [0.5, 1.0] for GBDT
        recommended_distribution = "uniform"
        recommended_range = [0.5, 1.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "validation_fraction" in p_lower and domain_type == "interval":
        # Usually in [0.0, 0.3]
        recommended_distribution = "uniform"
        recommended_range = [0.1, 0.3]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "max_samples" in p_lower and domain_type == "interval":
        # For Bagging or similar, typical usage is (0.5,1.0]
        recommended_distribution = "uniform"
        recommended_range = [0.5, 1.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result
    # -------------------------------------------------------------------------
    #                 TOLERANCE AND EPSILON
    # -------------------------------------------------------------------------
    # Many parameters called "tol" or "epsilon" are best explored in log-scale.
    # -------------------------------------------------------------------------
    if p_lower == "tol" and domain_type == "interval":
        recommended_distribution = "log"
        recommended_range = [1e-6, 1e-2]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "epsilon" in p_lower and domain_type == "interval":
        # For SVR, sometimes a log-scale in [1e-4, 1.0], 
        # for Huber or similar, might differ if domain is [1.0, inf].
        # We'll unify an approach:
        recommended_distribution = "log"
        recommended_range = [1e-4, 1.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if p_lower == "eps" and domain_type == "interval":
        # LassoCV or LarsCV often has eps in the sense of "small regularization".
        recommended_distribution = "log"
        recommended_range = [1e-6, 1e-1]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 DISTANCE / NEIGHBORS / RADIUS
    # -------------------------------------------------------------------------
    if "n_neighbors" in p_lower and domain_type == "interval":
        recommended_distribution = "uniform"
        recommended_range = [1, 50]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "leaf_size" in p_lower and domain_type == "interval":
        # For KNeighbors or RadiusNeighbors
        recommended_distribution = "uniform"
        recommended_range = [20, 50]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if "radius" in p_lower and domain_type == "interval":
        # For RadiusNeighborsRegressor
        recommended_distribution = "log"
        recommended_range = [1e-3, 10.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if p_lower == "p" and domain_type == "interval":
        # Minkowski power parameter (KNeighbors). Typically in [1, 2]
        recommended_distribution = "uniform"
        recommended_range = [1, 2]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 MLP-SPECIFIC (momentum, beta_1, beta_2, power_t, etc.)
    # -------------------------------------------------------------------------
    if p_lower == "momentum" and domain_type == "interval":
        # Typically in [0,1], so let's keep it uniform
        recommended_distribution = "uniform"
        recommended_range = [0.0, 0.99]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if p_lower in ("beta_1", "beta_2") and domain_type == "interval":
        # Usually in [0,1]. 
        recommended_distribution = "uniform"
        recommended_range = [0.8, 0.999]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if p_lower == "power_t" and domain_type == "interval":
        # For SGD. Typically in log scale if domain is large, or uniform if small.
        recommended_distribution = "log"
        recommended_range = [1e-3, 1.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    if p_lower == "max_fun" and domain_type == "interval":
        # MLP: rarely tuned, but if we do, let's keep uniform in [1000, 50000]
        recommended_distribution = "uniform"
        recommended_range = [1000, 50000]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 TWEEDIE / POWER
    # -------------------------------------------------------------------------
    if p_lower == "power" and domain_type == "interval":
        # For TweedieRegressor, typical in [0,2]. 
        recommended_distribution = "uniform"
        recommended_range = [0.0, 2.0]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 SMALL FRACTION / QUANTILE
    # -------------------------------------------------------------------------
    if p_lower == "quantile" and domain_type == "interval":
        # Typically [0.0,1.0], but let's restrict to [0.05, 0.95]
        recommended_distribution = "uniform"
        recommended_range = [0.05, 0.95]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    # -------------------------------------------------------------------------
    #                 N_COMPONENTS (PLS, PCA-likes)
    # -------------------------------------------------------------------------
    if "n_components" in p_lower and domain_type == "interval":
        # For PLS, CCA, etc. Usually from 1 up to some moderate value
        recommended_distribution = "uniform"
        recommended_range = [1, 20]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result
    
    # -------------------------------------------------------------------------
    #                 CV
    # -------------------------------------------------------------------------
    if "cv" in p_lower and domain_type == "interval":
        # For PLS, CCA, etc. Usually from 1 up to some moderate value
        recommended_distribution = "uniform"
        recommended_range = [2, 5]
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result
    # -------------------------------------------------------------------------
    #                 FALLBACK FOR ANY INTERVAL WE DID NOT COVER
    # -------------------------------------------------------------------------
    if domain_type == "interval":
        # If we reached here, we have an interval parameter we haven't specifically matched.
        # We try to see if it is in [0,1], we keep "uniform".
        # If it's [1, inf], we do [1, 50]. If it's [0, inf], fallback to [1e-6, 1e2].
        # Adjust as you see fit.
        recommended_distribution = "uniform"
        # If start is numeric and end is numeric, keep them if they look bounded
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if start >= 0 and end <= 1:
                # e.g. [0.0, 0.5], keep it
                recommended_range = [start, end]
            else:
                # Could be [1, 100], or something in general
                recommended_range = [start, min(end, 50) if not is_inf(end) else 50]
        else:
            # If we see 'inf', let's pick [1e-3, 1e2]
            recommended_range = [1e-3, 1e2]

    else:
        # If domain_type is something else, we keep "none" by default, or just keep domain
        recommended_distribution = "none"
        recommended_range = []
        result["recommended_distribution"] = recommended_distribution
        result["recommended_range"] = recommended_range
        return result

    return result

def create_cv_hyperparameters_model(
        df, 
        hyperparameters_preprocessing,
        hyperparameters_model_space,
        hyperparameters_specific_regressor_args,
):
    """
    Generate and execute cross-validation experiments with optimized hyperparameter combinations.

    This function dynamically generates all possible combinations of generic parameters and model-specific
    hyperparameters for a set of machine learning experiments. It then executes the experiments concurrently
    using the generated hyperparameter configurations.

    Parameters:
    ----------
    df : pd.DataFrame
            The raw dataframe containing the time-series data to be preprocessed and used for the cross-validation pipeline.

    hyperparameter_model_space : dict
            A dictionary defining the generic parameters and their possible values.
            These parameters are common across all models and include:
            - `ini_train` (list of str): List of start dates for training periods.
            - `fin_train` (list of str): List of end dates for training periods.
            - `fin_test` (list of str): List of end dates for testing periods.
            - `model_sklearn_name` (list of str): List of model names from scikit-learn to evaluate.
            - `n_lags` (list of int): Number of lag features to include in the models.
            - `n_leads` (list of int): Number of lead features to include in the models.
            - `X_name_features` (list of list of str): Feature subsets to include as predictors in the models.
            Other parameters can also be included as required.

    hyperparameters_specific_regressor_args : dict
            A dictionary mapping each `model_sklearn_name` to its corresponding hyperparameter domains.
            Each model's domain is defined as a dictionary of hyperparameter names and their possible values. 
            Example:
            {
            "ARDRegression": {
                    "max_iter": [200, 300],
                    "tol": [0.001, 0.01],
                    "alpha_1": [1e-06, 1e-05],
            },
            "Ridge": {
                    "alpha": [0.1, 1.0, 10.0],
                    "solver": ["auto", "svd"]
            }
            }

    Returns:
    -------
    None
        The function does not return any value. Instead, it generates the hyperparameter combinations and
        executes the experiments concurrently by passing the configurations to the `create_model_machine_learning_algorithm`
        function which manage all persistently with folder structure.

    Workflow:
    ---------
    #TODO: Add an explanation/workflow diagram to illustrate the steps in the function.

    Example Usage:
    --------------
    hyperparameter_model_space = {
        "ini_train": ["2023-04-18", "2023-05-01"],
        "fin_train": ["2023-04-25", "2023-05-08"],
        "fin_test": ["2023-04-27", "2023-05-10"],
        "model_sklearn_name": ["Ridge", "ARDRegression"],
        "n_lags": [1, 5],
        "n_leads": [5, 10],
        "X_name_features": [["feature1", "feature2"], ["feature3", "feature4"]],
    }

    model_specific_args = {
            "Ridge": {
            "alpha": [0.1, 1.0],
            "solver": ["auto", "svd"]
            },
            "ARDRegression": {
            "max_iter": [200],
            "tol": [0.001],
            }
    }

    create_cv_hyperparameters_model(hyperparameter_model_space, model_specific_args) 
    """
    def create_dataframe_combinations(df,parameter_space):
            """
            Generate preprocessed dataframes based on combinations of preprocessing parameters.

            This function dynamically generates all possible combinations of preprocessing parameters
            specified in the `parameter_space` dictionary, applies the preprocessing to the given dataframe,
            and returns a list of resulting preprocessed dataframes.

            Parameters:
            ----------
            df : pd.DataFrame
                    The raw dataframe containing the time-series data to be preprocessed.
            parameter_space : dict
                    A dictionary defining the preprocessing parameters and their possible values.
                    The keys in the dictionary represent parameter names, and the values are lists of possible
                    values for those parameters. The expected keys include:
                    - `df` (pd.DataFrame): The raw dataframe to be preprocessed. It should be included in the
                    `parameter_space` dictionary under the key `"df"`.
                    - `resample_freq` (list of str): Frequencies for resampling the time-series data.
                    - `aggregation_func` (list of str): Aggregation functions to apply during resampling.
                    - `interpolation_method` (list of str): Methods to use for interpolating missing values.
                    - `target_variable` (list of str): Names of the columns to set as the target variable.
                    - `pivot` (list of bool): Whether to apply a pivot operation.
                    - `pivot_index` (list of list of str): Columns to use as the index in the pivot table.
                    - `pivot_columns` (list of list of str): Columns to use as the columns in the pivot table.
                    - `pivot_values` (list of list of str): Columns to use as the values in the pivot table.
                    - `subset_cols` (list of list of str): Columns to subset the dataframe to.
                    - `target_column_name` (list of str): New names for the target variable column.

            Returns:
            -------
            list
                    A list of preprocessed dataframes, each generated using a unique combination of the parameters.

            Example Usage:
            --------------
            parameter_space = {
                    "df": raw_df, 
                    "resample_freq": ["60S", "30S"],
                    "aggregation_func": ["mean", "median"],
                    "interpolation_method": ["linear", "quadratic"],
                    "target_variable": ["00-eco2"],
                    "pivot": [True, False],
                    "pivot_index": [["timestamp", "id_device"]],
                    "pivot_columns": [["id_variable"]],
                    "pivot_values": [["value"]],
                    "subset_cols": [["timestamp", "id_device", "id_variable", "value"]],
                    "target_column_name": ["y"]
            }
            preprocessed_dataframes = create_dataframe_combinations(parameter_space)
            for df in preprocessed_dataframes:
                    print(df)
            """
            keys, values = zip(*parameter_space.items())

            return [
                    preprocess_and_standardize_dataframe(
                        df=df,  
                        resample_freq=params["resample_freq"],
                        aggregation_func=params["aggregation_func"],
                        interpolation_method=params["interpolation_method"],
                        target_variable=params.get("target_variable"),
                        outlier_cols=params.get("outlier_cols"),
                        pivot=params["pivot"],
                        pivot_index=params.get("pivot_index"),
                        pivot_columns=params.get("pivot_columns"),
                        pivot_values=params.get("pivot_values"),
                        subset_cols=params.get("subset_cols"),
                        target_column_name=params.get("target_column_name"),
                    )
                    for combination in product(*values)
                    for params in [dict(zip(keys, combination))]
                ]


    def generate_hyperparameters_machine_learning(parameter_space, model_specific_args, metadata_preprocessing):
        """
        Generate unique hyperparameter combinations for machine learning experiments.

        This function combines generic parameters and model-specific hyperparameters into dictionaries
        representing all possible configurations.

        Parameters:
        ----------
        parameter_space : dict
                Generic parameters shared across all models, such as training/testing periods, feature configurations,
                and model names (`model_sklearn_name`).

        model_specific_args : dict
                Hyperparameter domains for each model, mapped by `model_sklearn_name`.

        metadata_preprocessing : list
                List of metadata dictionaries corresponding to the preprocessing steps applied to each dataframe.

        Returns:
        -------
        list
                A list of dictionaries, where each dictionary contains a unique combination of:
                - Generic parameters (`parameter_space`).
                - Model-specific hyperparameters (`model_specific_args`).
                - Preprocessing metadata (`metadata_preprocessing`).
        """
        generic_keys, generic_values = zip(*((k, v) for k, v in parameter_space.items() if k != "model_sklearn_name"))
        model_names = parameter_space.get("model_sklearn_name", [])

        return [
            {
                **dict(zip(generic_keys, generic_comb)),
                "model_sklearn_name": model,
                "metadata_preprocessing": metadata,
                "machine_learning_model_args": dict(zip(model_args_keys, model_args_comb))
            }
            for generic_comb, metadata in zip(product(*generic_values), metadata_preprocessing)
            for model in model_names
            for model_args_keys, model_args_values in [(list(model_specific_args[model].keys()), list(model_specific_args[model].values()))]
            for model_args_comb in product(*model_args_values)
        ]
    
    dfs_preprocessed_and_metadata = create_dataframe_combinations(
        df=df, 
        parameter_space =hyperparameters_preprocessing
        )
    
    dfs_preprocessed = [df["df_resampled_interpolated"] for df in dfs_preprocessed_and_metadata]
    metadata_preprocessing = [df["metadata_preprocessing"] for df in dfs_preprocessed_and_metadata]
    
    hyperparameters_model_space["tidy_data"] = dfs_preprocessed

    hyperparameters = generate_hyperparameters_machine_learning(
    parameter_space=hyperparameters_model_space, 
    model_specific_args=hyperparameters_specific_regressor_args, 
    metadata_preprocessing=metadata_preprocessing
    )
    
    return execute_concurrently(
        create_model_machine_learning_algorithm,
        hyperparameters,
        executor_type='process',
        max_workers=os.cpu_count()  # Use all available CPUs
        )
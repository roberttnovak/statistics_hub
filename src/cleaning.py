import pandas as pd
from typing import List, Callable, Optional
import logging
from OwnLog import log_exceptions, log_function_args
import numpy as np 

@log_exceptions
@log_function_args
def prepare_dataframe_from_db(df: pd.DataFrame, 
                              cols_for_query: List[str], 
                              logger: Optional[logging.Logger] = None,
                              handle_timestamp: bool = True,
                              utc: bool = True) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame from a db structure. The expected DataFrame 
    structure from the db is as follows:
    
    | id_data | id_device | id_sensor | id_variable | timestamp           | value | unit | id_location |
    |---------|-----------|-----------|-------------|---------------------|-------|------|-------------|
    | 1       | DBEM003   | sWEA      | 00-temp     | 2023-04-18 09:31:00 | 18.57 | ºC   | NaN         |
    | ...     | ...       | ...       | ...         | ...                 | ...   | ...  | ...         |
    
    The preprocessing steps include:
    1. Dropping the 'id_data' column.
    2. Filtering the DataFrame based on specified variable identifiers 
       in the 'id_variable' column.
    3. Optionally handles the 'timestamp' column by converting it to datetime 
       and setting UTC if specified.
    
    This function is tailored for a DataFrame structure where each row represents 
    a data point, identified by a variable identifier and a data identifier.
    The preprocessing is essential to prepare the data for subsequent analysis tasks.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame, where each row corresponds to a data point, and contains 
        a 'id_data' column and a 'id_variable' column among others.
    cols_for_query : List[str]
        A list of variable identifiers to be retained in the DataFrame.
    logger : logging.Logger, optional
        A logger instance to log events during the execution of the function.
        If None (default), no logging will occur.
    handle_timestamp : bool, optional
        A flag to specify whether to handle the 'timestamp' column by converting it to 
        datetime and setting UTC if specified. Default is True.
    utc : bool, optional
        A flag to specify whether to set UTC for the 'timestamp' column if 
        handle_timestamp is True. Default is True.
    
    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame with 'id_data' column dropped, filtered 
        based on specified variable identifiers, and 'timestamp' column handled 
        if specified.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id_data': list(range(1, 11)),
    ...     'id_device': ['DBEM003'] * 10,
    ...     'id_sensor': ['sWEA'] * 10,
    ...     'id_variable': ['00-temp'] * 5 + ['01-hum'] * 5,
    ...     'timestamp': pd.date_range(start='2023-10-17 10:00:00', periods=10, freq='T'),
    ...     'value': [18.57 + i * 0.01 for i in range(10)],
    ...     'unit': ['ºC'] * 5 + ['%'] * 5,
    ...     'id_location': [None] * 10
    ... })
    >>> cols_for_query = ['00-temp']
    >>> preprocessed_df = prepare_dataframe_from_db(df, cols_for_query)
    >>> preprocessed_df
       id_device id_sensor id_variable           timestamp  value unit id_location
    0    DBEM003      sWEA     00-temp 2023-10-17 10:00:00  18.57   ºC        None
    1    DBEM003      sWEA     00-temp 2023-10-17 10:01:00  18.58   ºC        None
    2    DBEM003      sWEA     00-temp 2023-10-17 10:02:00  18.59   ºC        None
    3    DBEM003      sWEA     00-temp 2023-10-17 10:03:00  18.60   ºC        None
    4    DBEM003      sWEA     00-temp 2023-10-17 10:04:00  18.61   ºC        None

    In this example, the `prepare_dataframe_from_db` function processes the input DataFrame `df` to drop the 'id_data' 
    column, filter the DataFrame based on the specified variable identifier '00-temp', and handles the 'timestamp' column
    by converting it to datetime. The output DataFrame `preprocessed_df` retains only the rows with the specified variable 
    identifier and the relevant columns.
    """
    
    # Log the starting of the preprocessing process
    if logger:
        logger.info('Starting preprocessing of DataFrame.')
    
    # Ensure the necessary columns exist in the DataFrame.
    if 'id_data' not in df.columns:
        if logger:
            logger.error("Column 'id_data' not found in DataFrame.")
        raise ValueError("Column 'id_data' not found in DataFrame.")
    
    if 'id_variable' not in df.columns:
        if logger:
            logger.error("Column 'id_variable' not found in DataFrame.")
        raise ValueError("Column 'id_variable' not found in DataFrame.")
    
    for col in cols_for_query:
        if col not in df['id_variable'].values:
            if logger:
                logger.warning(f"Value {col} not found in column 'id_variable'.")
    
    # Drop 'id_data' column and create a copy to avoid modifying the original DataFrame.
    df_copy = df.drop(columns=['id_data']).copy()
    if logger:
        logger.info("'id_data' column dropped from DataFrame.")
    
    # Apply filter 
    preprocessed_df = df_copy[df_copy['id_variable'].isin(cols_for_query)].reset_index(drop=True)
    if logger:
        logger.info(f"DataFrame filtered based on specified variable identifiers: {cols_for_query}")
    
    # Handle 'timestamp' column if specified
    if handle_timestamp:
        if 'timestamp' not in preprocessed_df.columns:
            if logger:
                logger.error("Column 'timestamp' not found in DataFrame.")
            raise ValueError("Column 'timestamp' not found in DataFrame.")
        
        preprocessed_df.loc[:,"timestamp"] = pd.to_datetime(preprocessed_df["timestamp"], utc=utc)
        if logger:
            logger.info(f"'timestamp' column converted to datetime with UTC set to {utc}")
    
    # Log the completion of the preprocessing process
    if logger:
        logger.info('Preprocessing of DataFrame completed.')
    
    return preprocessed_df

def handle_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    # ToDo:
    # Placeholder for outlier handling logic.
    # Implement based on the specific needs of your data.
    return df

@log_exceptions
@log_function_args
def resample_data(df: pd.DataFrame, 
                  resample_freq: str = '60S', 
                  aggregation_func: Callable = np.mean,
                  logger: logging.Logger = None) -> pd.DataFrame:
    """
    Resamples the time series data in the DataFrame to a uniform frequency.
    
    The DataFrame is expected to have a 'timestamp' column, which is used as the index for 
    resampling. The data is grouped by 'id_device', 'id_sensor', and 'id_variable', and 
    the specified aggregation function is applied to compute the resampled values.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with time series data.
    resample_freq : str, optional
        The frequency for resampling the data (default is '60S' for 60 seconds).
    aggregation_func : Callable, optional
        The aggregation function to apply when resampling (default is mean).
    logger : logging.Logger, optional
        A logger instance to log events during the execution of the function.
        If None (default), no logging will occur.
    Returns
    -------
    pd.DataFrame
        The resampled DataFrame.

    Explanation
    -----------
    Consider a DataFrame with timestamps every 30 seconds. Resampling with a frequency of '60S' 
    (60 seconds) will aggregate these entries into 1-minute intervals. For instance, if the original 
    data has timestamps at 10:00:00, 10:00:30, and 10:01:00 with values 10, 15, and 20 respectively, 
    the resampling process will create two new rows: one for the interval 10:00:00 - 10:00:59, 
    and another for 10:01:00 - 10:01:59. Using the mean aggregation, the value for the first interval 
    will be the average of 10 and 15, and the value for the second interval will be 20.
    """

    if logger:
        logger.info(f'Starting resampling')

    df = df.copy()

    # Ensure the "timestamp" column is of datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set the "timestamp" column as the index
    df.set_index("timestamp", inplace=True)

    # Sort values by timestamp before grouping and resampling
    df.sort_values(by='timestamp', inplace=True)    

    df_resampled = df.groupby(['id_device', 'id_sensor', 'id_variable'])["value"] \
                     .resample(resample_freq) \
                     .apply(aggregation_func) \
                     .reset_index() \
                     .sort_values(by='timestamp')
    
    if logger:
        logger.info('Resampling completed.')    
    
    return df_resampled


@log_exceptions
@log_function_args
def interpolate_data(df: pd.DataFrame, 
                     method: str = 'linear', 
                     logger: logging.Logger = None) -> pd.DataFrame:
    """
    Interpolates missing values in the time series data using specified interpolation method.
    
    The DataFrame is expected to have a 'timestamp' column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with resampled time series data.
    method : str, optional
        The method of interpolation to use. Default is 'linear'. Other options include 
        'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc., as per 
        the options available in pandas.DataFrame.interpolate method.
    logger : logging.Logger, optional
        A logger instance to log events during the execution of the function.
        If None (default), no logging will occur.
    
    Returns
    -------
    pd.DataFrame
        The interpolated DataFrame.

    Explanation
    -----------
    Suppose the DataFrame has missing values at certain timestamps. The interpolation method fills 
    these gaps based on the specified method. For instance, if the DataFrame has values at 10:00:00 (10),
    10:01:00 (NaN), and 10:02:00 (20), a 'linear' interpolation will estimate the value at 10:01:00 as 
    the midpoint (15) between the two known values. Different interpolation methods use different techniques
    to estimate the missing values. For example, 'cubic' interpolation considers the curvature of the data 
    points, creating a smoother estimate than 'linear'.
    
    Examples
    --------
    >>> df_resampled = ...
    >>> df_interpolated = interpolate_data(df_resampled, method='cubic')
    """
    if logger:
        logger.info(f'Starting interpolation with method: {method}')

    df_interpolated = df.copy()
    df_interpolated.set_index('timestamp', inplace=True)
    df_interpolated = df_interpolated.groupby(['id_device', 'id_sensor', 'id_variable'])['value'] \
                                     .apply(lambda group: group.interpolate(method=method)) \
                                     .reset_index()
    
    if logger:
        logger.info('Interpolation completed.')
    
    return df_interpolated

@log_exceptions
@log_function_args
def process_time_series_data(df: pd.DataFrame, 
                             resample_freq: str = '60S', 
                             aggregation_func: Callable = np.mean,
                             method: str = 'linear',
                             outlier_cols: list = None,
                             logger: logging.Logger = None) -> pd.DataFrame:
    """
    Processes time series data by resampling to a uniform frequency and interpolating missing values.
    
    This function is a wrapper that calls resample_data and interpolate_data in sequence.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with time series data.
    resample_freq : str, optional
        The frequency for resampling the data (default is '60S' for 60 seconds).
    aggregation_func : Callable, optional
        The aggregation function to apply when resampling (default is mean).
    method : str, optional
        The method of interpolation to use. Default is 'linear'. Other options include 
        'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc., as per 
        the options available in pandas.DataFrame.interpolate method.
    outlier_cols : list, optional
        A list of column names in the DataFrame where outliers should be handled.
        If provided, the `handle_outliers` function will be called to handle outliers 
        in these specified columns. If None (default), no outlier handling will occur.
    logger : logging.Logger, optional
        A logger instance to log events during the execution of the function.
        If None (default), no logging will occur.
    Returns
    -------
    pd.DataFrame
        The processed DataFrame with resampled and interpolated data.
    """

    if outlier_cols:
        df = handle_outliers(df, outlier_cols)
        if logger:
            logger.info('Outliers handled.')

    df_resampled = resample_data(df=df, 
                                 resample_freq = resample_freq, 
                                 aggregation_func = aggregation_func, 
                                 logger= logger)
    
    df_processed = interpolate_data(df = df_resampled,
                                    method=method,
                                    logger=logger)
    
    return df_processed

def interpolate_mv_local_median(XData, BandwithD, BandwithH):
    """
    Interpolate missing values in a dataset using local median.

    This function uses a local median-based approach for interpolating missing
    values in a 2D array. It iteratively increases the search window until
    a non-NaN median is found for each missing value.

    Parameters
    ----------
    XData : np.array
        array with missing values (NaNs) to be interpolated.
    BandwithD : int
        Bandwidth in the vertical (row) direction for the local median calculation.
    BandwithH : int
        Bandwidth in the horizontal (column) direction for the local median calculation.

    Returns
    -------
    np.array
        2D array with missing values interpolated.

   Explanation
    -----------
    Consider the following example to understand how this function works:

    Example XData:
        Xdata = np.array([
            [1, 2, np.nan],
            [4, np.nan, 6],
            [7, 8, 9]
        ])
    
    Here, the function is called with:
        interpola_mv_local_median(XData, 1, 1)

    Step-by-Step Process:
        1. Identify Missing Values:
           - The function locates two NaN values at positions [0, 2] and [1, 1].

        2. Interpolate Each Missing Value:
           a. For NaN at [0, 2]:
              - The function looks in the 1x1 bandwidth around it.
              - It finds the values [2, NaN, 6] in this window.
              - The median of these values is 4 (since NaN is ignored).
              - Thus, NaN at [0, 2] is replaced with 4.

           b. For NaN at [1, 1]:
              - The function looks in the 1x1 bandwidth around it.
              - The window contains NaNs or no values in the first iteration.
              - The window is expanded, and the search continues until a valid median is found.
              - Once found, the NaN at [1, 1] is replaced with this median.

        3. Result:
           - The returned array has no NaN values, with each NaN replaced by an appropriate local median.

    Examples
    --------
    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
    >>> print(interpola_mv_local_median(data, 1, 1))
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])


    """
    # Copy the original array to avoid modifying the original data.
    X = np.copy(XData)
    # Get the dimensions of the array (n rows and p columns).
    n, p = X.shape

    # Identify indices of missing values (NaNs) in the array.
    missing_indices = np.argwhere(np.isnan(X))

    # Iterate over each missing value's indices.
    for i, h in missing_indices:
        k = 1  # Initialize the bandwidth multiplier.
        
        # Continue the loop as long as the current element is NaN and the search window
        # is within the bounds of the array.
        while np.isnan(X[i, h]) and k * BandwithD < n and k * BandwithH < p:
            # Calculate the minimum and maximum row indices for the local window.
            row_min = max(i - k * BandwithD, 0)
            row_max = min(i + k * BandwithD + 1, n)

            # Calculate the minimum and maximum column indices for the local window.
            col_min = max(h - k * BandwithH, 0)
            col_max = min(h + k * BandwithH + 1, p)

            # Flatten the data within the local window to a 1D array.
            IData = X[row_min:row_max, col_min:col_max].flatten()

            # Calculate the median of the local window, ignoring NaNs.
            local_median = np.nanmedian(IData)
            
            # If a non-NaN median is found, replace the NaN value with this median.
            if not np.isnan(local_median):
                X[i, h] = local_median

            # Increase the bandwidth multiplier to expand the search window in the next iteration.
            k += 1

    return X


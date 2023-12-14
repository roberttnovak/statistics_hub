import pandas as pd
import numpy as np

def summary_statistics_numerical(df, variables=None, groupby=None):
    """
    Generate a statistical summary of the data in a DataFrame for numerical variables 

    This function computes various statistical measures (count, quartiles, percentiles, median,
    mean, standard deviation, min, max) for specified variables, optionally grouped by specified categories.

    Parameters:
    - df (pd.DataFrame): DataFrame from which to generate the statistical summary.
    - variables (str or list of str, optional): Name(s) of the column(s) for which to compute statistics.
      If None (default), statistics are computed for all columns.
    - groupby (str or list of str, optional): Name(s) of the column(s) containing categories.
      If specified, the summary is generated separately for each category.

    Returns:
    - pd.DataFrame: A DataFrame containing the statistical summary for the specified variables and groups.

    Notes:
    - If 'variables' is a string, it is converted to a list for consistent processing.
    - If 'groupby' is not specified, a default 'all' category is used for summarizing.

    Future Developments:
    - Allow specification of custom summary functions as arguments.
    """

    # Copy the dataframe to avoid modifying the original
    df = df.copy()

    # Default grouping handling
    if groupby is None:
        df["group"] = "all"
        groupby = "group"
    else:
        if isinstance(groupby, str):
            groupby = [groupby]

    # Handling 'variables' argument
    if variables is None:
        variables = list(set(df.columns) - set(groupby))
    else:
        if isinstance(variables, str):
            variables = [variables]
        variables = list(set(variables) - set(groupby))

    # Prepare list to store summary dataframes
    dfs_summary = []

    # Iterate over variables and compute summary statistics
    for variable in variables:
        df_summary = df.groupby(groupby).agg(
            count=(variable, 'count'),
            Q1=(variable, lambda x: x.quantile(0.25)),
            Q3=(variable, lambda x: x.quantile(0.75)),
            percentile_90=(variable, lambda x: x.quantile(0.9)),
            percentile_99=(variable, lambda x: x.quantile(0.99)),
            median=(variable, 'median'),
            mean=(variable, 'mean'),
            std=(variable, 'std'),
            min=(variable, 'min'),
            max=(variable, 'max')
        ).reset_index()

        # Adding a column to identify the variable
        df_summary["variable_summary"] = variable
        dfs_summary.append(df_summary)

    # Concatenate all summary dataframes
    final_summary_df = pd.concat(dfs_summary)

    return final_summary_df

def summary_statistics_categorical(df, variables=None, groupby=None):
    """
    ... (El docstring permanece igual)
    """

    df = df.copy()

    if groupby is None:
        df["group"] = "all"
        groupby = ["group"]
    elif isinstance(groupby, str):
        groupby = [groupby]

    if variables is None:
        variables = df.select_dtypes(include=['object', 'category']).columns.tolist()
    elif isinstance(variables, str):
        variables = [variables]

    dfs_summary = []

    for variable in variables:
        df_summary = df.groupby(groupby + [variable]).size().reset_index(name='count')
        total = df_summary.groupby(groupby)['count'].sum()
        df_summary = df_summary.merge(total, on=groupby, suffixes=('', '_total'))
        df_summary['percentage'] = (df_summary['count'] / df_summary['count_total']) * 100

        df_summary["variable_summary"] = variable
        dfs_summary.append(df_summary)

    final_summary_df = pd.concat(dfs_summary)

    return final_summary_df


def summary_statistics(df, variables=None, groupby=None):
    """
    Generate a combined statistical summary for both numerical and categorical variables in a DataFrame.

    This function automatically detects the type of each column and applies the appropriate
    statistical summary: numerical summary for numeric columns and frequency count for string-type columns.
    Users can specify columns to include in the analysis using the 'variables' argument.

    Parameters:
    - df (pd.DataFrame): DataFrame from which to generate the summary.
    - variables (str or list of str, optional): Name(s) of the column(s) for analysis. If None, all columns are analyzed.
    - groupby (str or list of str, optional): Name(s) of the column(s) containing grouping categories.

    Returns:
    - pd.DataFrame: A DataFrame containing the combined summary for both numerical and categorical variables.

    Notes:
    - The function internally calls `summary_statistics_numerical` and `summary_statistics_categorical`.
    """

    # Handling 'variables' argument
    if variables is not None:
        if isinstance(variables, str):
            variables = [variables]
        df = df[variables + (groupby if groupby is not None else [])]

    # Split dataframe into numeric and categorical parts
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply appropriate summary function to each part
    numeric_summary = summary_statistics_numerical(df, variables=numeric_cols, groupby=groupby) if numeric_cols else pd.DataFrame()
    categorical_summary = summary_statistics_categorical(df, variables=categorical_cols, groupby=groupby) if categorical_cols else pd.DataFrame()

    # Combine the summaries
    combined_summary = pd.concat([numeric_summary, categorical_summary])

    return combined_summary

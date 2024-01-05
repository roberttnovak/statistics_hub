#ToDo: Move sklearn functions utils to this script and fix dependencies

import json
import os
from sklearn.utils import all_estimators

from sklearn_regressor_scraper import get_regressor_info


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
    regressor_params = get_all_regressors_with_its_parameters()

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

def get_explanation_all_parameters(regressor):
    all_regressors_with_its_parameters = get_all_regressors_with_its_parameters()
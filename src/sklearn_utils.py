#ToDo: Move sklearn functions utils to this script and fix dependencies

import json
import os
from sklearn.utils import all_estimators
import requests
from bs4 import BeautifulSoup



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



def extract_regressor_info(soup, parameter_subset=None):
    """
    Extracts regressor information and parameter details from the BeautifulSoup object. 
    It includes an optional filter for specific parameters.

    Parameters
    ----------
    soup : BeautifulSoup
        BeautifulSoup object containing the parsed HTML of the regressor documentation page.
    parameter_subset : list of str, optional
        A list of parameter names to specifically include in the extraction. 
        If None (default), information for all parameters is extracted.

    Returns
    -------
    dict
        A dictionary containing two keys: 'regressor_info' and 'parameters_info'.
        'regressor_info' is a string with the concatenated text of all relevant <p> tags before the first <dl>. This is: Description of regressor
        'parameters_info' is a list of dictionaries, each containing 'parameter', 'description', and 'value_default' for each parameter.
    """
    # Find decription of regressor 
    dd_tag_of_descriprion_regressor = soup.find_all('dl')[0].find('dd')
    p_tags_before_first_dl = []
    for sibling in dd_tag_of_descriprion_regressor.children:
        if sibling.name == 'dl':  
            break
        if sibling.name == 'p': 
            p_tags_before_first_dl.append(sibling)
    regressor_info = "\n".join([p.text for p in p_tags_before_first_dl])

    # Find parameters and its default values and descriptions
    dd_tags = soup.find_all('dd', class_='field-odd')
    dt_parameters_tags = [dt for dt in dd_tags[0].find_all('dt') if dt.find('strong')]
    # Extract parameter names and default values
    parameters, values_default = zip(*[
        [
            parameter_tag.strong.text, 
            parameter_tag.find('span', class_='classifier').text.split('default=')[1] 
            if parameter_tag.find('span', class_='classifier') and 'default=' in parameter_tag.find('span', class_='classifier').text 
            else None
        ]
        for parameter_tag in dt_parameters_tags
    ])
    
    # Extract descriptions of the parameters
    dd_parameters_tags = [
        dt.find_next_sibling() 
        for dt in dt_parameters_tags 
        if dt.find_next_sibling() and dt.find_next_sibling().name == 'dd'
    ]

    parameters_descriptions = [dd.text for dd in dd_parameters_tags]

    # Filter the information based on parameter_subset
    parameters_info = [
        {"parameter": parameter, "description": description, "value_default": value_default}
        for parameter, value_default, description in zip(parameters, values_default, parameters_descriptions)
        if parameter_subset is None or parameter in parameter_subset
    ]
    return {"regressor_info":regressor_info, "parameters_info":parameters_info}

def get_regressor_info(regressor_name, parameter_subset=None):
    """
    Retrieves and scrapes the scikit-learn documentation for a specified regressor to obtain detailed information, 
    optionally filtered by a subset of its parameters.

    Parameters
    ----------
    regressor_name : str
        Name of the regressor whose information is to be scraped from the scikit-learn documentation.
    parameter_subset : list of str, optional
        A list of parameter names to filter the information. If None (default), information for all parameters is retrieved.

    Returns
    -------
    dict or str
        If successful, returns a dictionary with the regressor name as the key and the extracted information as the value. 
        The value is a dictionary containing 'regressor_info' and 'parameters_info'.
        If the regressor name is not found or an HTTP error occurs, returns an error message string.
    """
    # Create a mapping of regressor names to their module paths
    mapping = {name: clazz.__module__ for name, clazz in all_estimators(type_filter='regressor')}
    
    # Retrieve the module path of the regressor and construct the URL
    module_path = mapping.get(regressor_name)
    if module_path:
        # Adjust the module path to match the structure of the documentation URL
        adjusted_module_path = module_path.split('.')[1]  # Take the primary module element
        base_url = f"https://scikit-learn.org/stable/modules/generated/sklearn.{adjusted_module_path}.{regressor_name}.html"
        response = requests.get(base_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            info = extract_regressor_info(soup, parameter_subset)
            return {regressor_name : info}
        else:
            print(f"Failed to retrieve documentation for {regressor_name}. Link: {base_url}")
            return None
    else:   
        print(f"Class name {regressor_name} not found in module mapping.")
        return None
    

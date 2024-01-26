"""
This script provides a suite of functions to interact with scikit-learn's regressors. 
It includes utilities for retrieving the list of all available regressors, obtaining their parameters, 
and scraping detailed documentation from scikit-learn's online resources. 
Scrapping is necessary to get description of each regressor and its parameters.

The key functionalities include:
- `get_all_regressors`: Retrieves a list of all regressor names from scikit-learn.
- `get_all_regressors_with_its_parameters`: Generates a dictionary mapping each regressor to its parameters.
- `extract_regressor_info`: Extracts detailed information of a specific regressor from its BeautifulSoup-parsed HTML documentation.
- `get_regressor_info`: Scrapes the scikit-learn documentation for a specific regressor to obtain detailed information.
- `get_valid_regressors_info`: Compiles detailed information about all scikit-learn regressors and their parameters into a comprehensive dictionary.

The script is designed to be modular and can be extended or modified to incorporate additional functionalities 
related to machine learning regressors in scikit-learn. It's an ideal tool for data scientists and machine learning practitioners 
who need quick and organized access to scikit-learn regressor details for analysis or automation purposes.

Dependencies:
- scikit-learn: Used for accessing machine learning regressors.
- BeautifulSoup: Utilized for parsing HTML content for web scraping.
- requests: Required for making HTTP requests to scikit-learn documentation pages.

Examples of Use:
- Fetching a list of regressors excluding specific ones.
- Getting parameters of all scikit-learn regressors.
- Extracting and viewing detailed information about a specific regressor and its parameters.

Note:
The script requires an internet connection to access scikit-learn's online documentation for scraping purposes
when use extract_regressor_info() and get_regressor_info() functions()
"""

import re
from sklearn.utils.discovery import all_estimators
import requests
from bs4 import BeautifulSoup

def get_all_regressors(exclude=None):
    """
    Retrieve all regressor names from scikit-learn, with an option to exclude specific ones.

    Parameters
    ----------
    exclude : list of str, optional
        A list of regressor names to exclude from the results. If None (default), all regressors are included.

    Returns
    -------
    list
        A list of regressor names available in scikit-learn, excluding any specified in the 'exclude' parameter.
    
    Example
    -------
    >>> get_all_regressors(exclude=['ARDRegression', 'AdaBoostRegressor'])
    ['LinearRegression', 'Ridge', ...]
    """
    # Obtain all regressors from scikit-learn
    regressors = all_estimators(type_filter='regressor')

    # Filter out excluded regressors
    if exclude is not None:
        regressors = [regressor for regressor in regressors if regressor[0] not in exclude]

    # Extract and return the names of the regressors
    regressor_names = [name for name, _ in regressors]
    
    return regressor_names


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
            re.sub(r"[\"'“”‘’ ]","",parameter_tag.find('span', class_='classifier').text.split('default=')[1]) # Some defaults values are string and sometimes is obtained as, for example, ’linear’
            if parameter_tag.find('span', class_='classifier') and 'default=' in parameter_tag.find('span', class_='classifier').text 
            else None # Some parameters dont have default values
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
            info.update({"url_scrapped":base_url})
            return {regressor_name : info}
        else:
            print(f"Failed to retrieve documentation for {regressor_name}. Link: {base_url}")
            return None
    else:   
        print(f"Class name {regressor_name} not found in module mapping.")
        return None

#ToDo: Rename get_all_regressor_info
def get_valid_regressors_info(all_regressors_with_its_parameters = None):
    """
    Retrieves detailed information for all scikit-learn regressors and their parameters.
    
    This function iterates over all available scikit-learn regressors and extracts detailed information 
    for each, including descriptions, parameter details, and default values. It returns a nested dictionary 
    where each top-level key is a regressor name and its value is a dictionary containing detailed information 
    about that regressor and its parameters.

    Parameters
    ----------
    all_regressors_with_its_parameters : dict, optional
        A dictionary with regressor names as keys and lists of their parameters as values.
        If None (default), the function internally calls `get_all_regressors_with_its_parameters`
        to obtain this dictionary.

    Returns
    -------
    dict
        A nested dictionary with the following structure:
        {
            'Regressor1': {
                'regressor_info': 'Description of Regressor1...',
                'parameters_info': [
                    {
                        'parameter': 'param1',
                        'description': 'Description of param1...',
                        'value_default': 'default value of param1...'
                    },
                    ...
                ]
            },
            ...
        }
        Each regressor's entry contains 'regressor_info' (a string with a description of the regressor)
        and 'parameters_info' (a list of dictionaries, each for one parameter of the regressor).

    Example
    -------
    >>> all_regressors_info = get_valid_regressors_info()
    >>> print(all_regressors_info['LinearRegression']['regressor_info'])
    'LinearRegression regressor description...'
    >>> print(all_regressors_info['LinearRegression']['parameters_info'][0])
    {'parameter': 'fit_intercept', 'description': 'Whether to calculate the intercept for this model...', 'value_default': 'True'}
    """
    if all_regressors_with_its_parameters is None:
        all_regressors_with_its_parameters = get_all_regressors_with_its_parameters()
    all_regressors_info = {}
    for regressor, parameters in all_regressors_with_its_parameters.items(): 
        regressor_info = get_regressor_info(regressor, parameters)
        all_regressors_info.update(regressor_info)
    return all_regressors_info

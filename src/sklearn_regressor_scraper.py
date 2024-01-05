"""
Regressor Information Scraper for Scikit-Learn

This script provides tools to scrape and extract detailed information about regressors available
in the scikit-learn library. It uses web scraping to obtain descriptions and details of the 
parameters of given regressor classes directly from the scikit-learn official documentation.


Usage
-----
The script can be used to quickly access detailed information about any regressor in scikit-learn by simply providing its 
class name. This is particularly useful for data scientists and developers who need to reference regressor details 
without manually searching through documentation.

Example
-------
To get information about the 'RandomForestRegressor', you can call the function like this:

    info = get_regressor_info('RandomForestRegressor')
    print(info)

Note: Any changes in the website's structure may affect the script's functionality.
"""

import requests
from bs4 import BeautifulSoup
from sklearn.utils import all_estimators


def extract_regressor_info(soup, parameter_subset=None):
    """
    Extracts regressor information from the BeautifulSoup object with an optional filter for specific parameters.

    Parameters
    ----------
    soup : BeautifulSoup
        BeautifulSoup object containing the parsed HTML.
    parameter_subset : list of str, optional
        A list of parameter names to filter the information. If None (default), information for all parameters is returned.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the extracted information about a parameter and its description.
        Each dictionary has keys: 'parameter', 'description', 'value_default'.
    """
    dl_tags = soup.find_all('dd', class_='field-odd')
    dt_parameters_tags = [dt for dt in dl_tags[0].find_all('dt') if dt.find('strong')]
    
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

    descriptions = [dd.text for dd in dd_parameters_tags]

    # Filter the information based on parameter_subset
    info = [
        {"parameter": parameter, "description": description, "value_default": value_default}
        for parameter, value_default, description in zip(parameters, values_default, descriptions)
        if parameter_subset is None or parameter in parameter_subset
    ]
    return info


def get_regressor_info(regressor_name, parameter_subset=None):
    """
    Scrape the scikit-learn documentation to obtain detailed information about a regressor, with an optional parameter subset filter.

    Parameters
    ----------
    regressor_name : str
        The name of the regressor for which information is to be scraped.
    parameter_subset : list of str, optional
        A list of parameter names to filter the information. If None (default), information for all parameters is returned.

    Returns
    -------
    dict or str
        A dictionary containing the description and parameters of the regressor. If a specific parameter subset is requested,
        returns information only for those parameters. If the parameter is not found, a message is returned indicating no information is available.
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
            regressor_info = extract_regressor_info(soup, parameter_subset)
            return {regressor_name : regressor_info}
        else:
            print(f"Failed to retrieve documentation for {regressor_name}. Link: {base_url}")
            return None
    else:   
        print(f"Class name {regressor_name} not found in module mapping.")
        return None

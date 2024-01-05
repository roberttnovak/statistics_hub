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

import functools
import json
import logging
import os
from pathlib import Path
import pymysql
import requests
import concurrent.futures
import inspect
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time
from typing import Any, Dict


# nos conectamos a la BD pasada como argumento
def connect(_server, _user, _password, _database):
    try:
        db = pymysql.connect(host=_server, user=_user, password=_password, database=_database)
        return db
    except pymysql.err.OperationalError as e:
        # e[0]
            # (1007, "Can't create database 'test_piloto'; database exists")
            # (1045, "Access denied for user 'jaime'@'localhost' (using password: YES)")
            # (1049, "Unknown database 'test_piloto'")
            # (2003, "Can't connect to MySQL server on 'localhost2' ([Errno -3] Temporary failure in name resolution)")
        print("Exeception occured in coonect:{}".format(e))
    except Exception as e:
        print("Unknown Exeception occured:{} Please contact with admin.".format(e))


# nos desconectamos de la BD
def disconnect(_database):
    _database.close()

# def insert_measure_ssh(_id_sensor, _sensor_type, _measure, _unit, _timestamp, _server, _user, _password, _database):
#     with SSHTunnelForwarder((ssh_host, ssh_port),ssh_username=ssh_user,ssh_password=ssh_password,remote_bind_address=(_server, 3306)) as tunnel: 
#         _db = pymysql.connect(host='127.0.0.1', user=_user,passwd=_password, db=_database,port=tunnel.local_bind_port)
#         _cursor = _db.cursor()
#         sql = """Insert INTO data (id_sensor,variable,value,unit,timestamp) VALUES ('%s', '%s', %s, '%s', '%s')""" %(_id_sensor,_sensor_type,_measure,_unit,_timestamp)
#         try:
#             _cursor.execute(sql)
#             _db.commit()
#         except Exception as e:
#             # Rollback in case there is any error
#             print(e)
#             _db.rollback()
#         _db.close()

# insertamos una medida
def insert_measure(_db, _cursor, _id_sensor, _sensor_type, _measure, _unit, _timestamp):
    sqlQuery = """Insert INTO data (
                id_sensor,
                variable,
                value,
                unit,
                timestamp)
                VALUES ('%s', '%s', %s, '%s', '%s')""" %(
                _id_sensor,
                _sensor_type,
                _measure,
                _unit,
                _timestamp)

    try:
        _cursor.execute(sqlQuery)
        _db.commit()
    except Exception as e:
        # Rollback in case there is any error
        print(e)
        _db.rollback()


def get_sensor(_db, _cursor, _id_sensor):
    sqlQuery = """Select * from sensor where id_sensor = '%s'""" %(_id_sensor)

    try:
        _cursor.execute(sqlQuery)
        return _cursor.fetchone()
    except Exception as e:
        # Rollback in case there is any error
        print(e)
        _db.rollback()


def get_data_api(_objectid,_channelid,_num):
    try:
        a = requests.get('http://gurapi-emergency.guapeton.deep-insight.es/v1/data/data_last?projectid=DBEMERGENCY&acqid=unknown&objectid='+_objectid+'&channelid='+_channelid+'&num='+_num)
        if a.status_code != 200:
            ret = False
            a.close()
        return a.json()

    except Exception as e:

        print(e)
        ret = False


def get_data_api_time(_objectid,_channelid,_init,_end):
    try:
        a = requests.get('http://gurapi-emergency.guapeton.deep-insight.es/v1/data/data_bydate?projectid=DBEMERGENCY&acqid=unknown&objectid='+_objectid+'&channelid='+_channelid+'&init='+_init+'&end='+_end)
        # print('http://gurapi-emergency.guapeton.deep-insight.es/v1/data/data_bydate?projectid=DBEMERGENCY&acqid=unknown&objectid='+_objectid+'&channelid='+_channelid+'&init='+_init+'&end='+_end)
        if a.status_code != 200:
            ret = False
            a.close()
        return a.json()

    except Exception as e:

        print(e)
        ret = False

# r = get_data_api('sWEADBEM001-0','02-pres','1')
# # print(r)
# sensors = ['DBEM001', 'DBEM002', 'DBEM003', 'DBEM004', 'DBEM005', 'DBEM006', 'DBEM007','DBEM008']
# sensors_sWEA = ['sWEADBEM001-0', 'sWEADBEM002-0', 'sWEADBEM003-0', 'sWEADBEM004-0', 'sWEADBEM005-0', 'sWEADBEM006-0', 'sWEADBEM007-0', 'sWEADBEM008-0']
# channels_sWEA = ['00-temp', '01-hum', '02-pres', '03-siaq', '04-diaq', 'Q0-asiaq', 'Q1-adiaq']
# cal_sWEA = [0.01,0.01,1,1,1,1,1]
# server = 'localhost'
# user = 'smartcampus'
# password = 'RJ47PIANO!a'
# database = 'smartcampus_hduino'
# mysql_client = connect(_server= server, _user=user, _password=password, _database= database)
# mysql_cursor = mysql_client.cursor()
# def save_mysql(_sensor,_variable,_data,_unit, _time, _cal):  
#     try: 
#         insert_measure(mysql_client,mysql_cursor,_sensor,_variable,int(_data)*_cal, _unit, _time)
#         print("hola")
#     except Exception as e:
#         print(e)  

# r = get_data_api_time('sWEADBEM001-0','00-temp','2021-05-26 08:00:00.000000', '2021-05-26 14:48:00.000000')
# print(r.keys())

# for s in range(0,len(sensors_sWEA)):
#     for i in range(0,len(r['data'])):
#                 save_mysql(sensors[s],'temperatura', r['data'][i], 'ºC', r['time'][i], cal_sWEA[0])
        

def execute_concurrently(func, args_list):
    """
    Executes a function concurrently with different sets of arguments.
    
    Parameters:
    func (callable): A callable object (e.g., function) to be executed.
    args_list (list): A list of dictionaries, where each dictionary contains 
                      the keyword arguments for the function.
    
    Returns:
    list: A list of results from the function.
    
    Example:
    >>> def add(x, y):
    ...     return x + y
    ...
    >>> args_list = [{'x': 5, 'y': 3}, {'x': 10, 'y': 7}]
    >>> execute_concurrently(add, args_list)
    [8, 17]
    """
    
    # Initialize an empty list to store the results
    results = []
    
    # Create a ThreadPoolExecutor to manage the concurrent execution of the function
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        # Create a list to store the Future objects returned by executor.submit,
        # and map each Future to its corresponding arguments
        futures = {executor.submit(func, **args): args for args in args_list}
        
        # Iterate through the Future objects as they complete
        for future in concurrent.futures.as_completed(futures):
            args = futures[future]
            try:
                # Get the result of the completed function and print a success message
                result = future.result()
                print(f"Task with args {args} processed successfully.")
                results.append(result)
            except Exception as exc:
                # If an exception occurs during execution, print an error message
                print(f"Task with args {args} generated an exception: {exc}")
                results.append(exc)
    
    # Return the list of results
    return results


def list_directories_by_depth(path, max_depth=3, list_only_last_level=True):
    """
    Lists all directories and subdirectories up to a specified depth.
    
    Parameters:
    path (str): The path of the directory to start from.
    max_depth (int): The maximum depth to explore. Default is 3.
    list_only_last_level (bool): If True, only the directories at max_depth will be listed.
                                 If False, all directories from the starting path to max_depth will be listed.
                                 Default is True.
    
    Returns:
    list: A list of directories and subdirectories up to the specified depth, 
          or a list of directories at the specified depth if list_only_last_level is True.
    
    Example:
    --------
    Assume the directory structure is as follows:
    - /path/to/dir
        - subdir1
            - subsubdir1
            - subsubdir2
        - subdir2
            - subsubdir3
            - subsubdir4
    
    >>> list_directories_by_depth('/path/to/dir', max_depth=2, list_only_last_level=True)
    ['/path/to/dir/subdir1/subsubdir1', '/path/to/dir/subdir1/subsubdir2', '/path/to/dir/subdir2/subsubdir3', '/path/to/dir/subdir2/subsubdir4']
    
    >>> list_directories_by_depth('/path/to/dir', max_depth=2, list_only_last_level=False)
    ['/path/to/dir', '/path/to/dir/subdir1', '/path/to/dir/subdir1/subsubdir1', '/path/to/dir/subdir1/subsubdir2', '/path/to/dir/subdir2', '/path/to/dir/subdir2/subsubdir3', '/path/to/dir/subdir2/subsubdir4']
    """
    # Convert the path to a pathlib.Path object for easier path manipulation
    path = Path(path)
    # Get the absolute path to ensure accurate depth calculations
    path = path.resolve()
    
    # Initialize an empty list to store the result
    result = []
    
    # Define the root depth based on the number of directory separators in the root path
    root_depth = len(path.parts) - 1  # Subtract 1 as the root itself is not counted in the depth
    
    # Use os.walk to efficiently traverse the directory structure
    for dirpath, dirnames, filenames in os.walk(str(path)):
        # Convert the current directory path to a pathlib.Path object
        dirpath = Path(dirpath)
        
        # Calculate the depth of the current directory relative to the root path
        depth = len(dirpath.parts) - root_depth - 1  # Subtract 1 as the root itself is not counted in the depth
        
        # If the current depth is less than the max depth...
        if depth < max_depth:
            # ... and if list_only_last_level is False, add the current directory to the result
            if not list_only_last_level:
                result.append(str(dirpath))
        # If the current depth is equal to the max depth...
        elif depth == max_depth:
            # ... add the current directory to the result
            result.append(str(dirpath))
            # ... and stop os.walk from traversing deeper by clearing the dirnames list
            del dirnames[:]
    
    return result


def get_all_args(func):
    """
    Retrieves all the arguments and their default values of a given function.
    
    This function inspects a given function to extract its arguments and their 
    default values, if any. This can be useful for logging, debugging, or 
    other introspective tasks within a program.
    
    Parameters:
    -----------
    func : Callable
        The function whose arguments need to be extracted.
        
    Returns:
    --------
    dict
        A dictionary where keys are the argument names and values are the 
        default values of the arguments. If an argument doesn't have a default 
        value, its value in the dictionary will be None.
        
    Examples:
    ---------
    >>> def example_function(arg1, arg2=5):
    ...     pass
    ...
    >>> args_dict = get_all_args(example_function)
    >>> print(args_dict)
    {'arg1': None, 'arg2': 5}
    
    Raises:
    -------
    TypeError:
        If the provided `func` parameter is not a callable object.
    """
    if not callable(func):
        raise TypeError(f"The provided object {func} is not callable.")
        
    sig = inspect.signature(func)
    return {k: v.default if v.default is not inspect.Parameter.empty else None 
            for k, v in sig.parameters.items()}

def load_json(folder_path: str, json_filename: str) -> dict:
    """
    Load contents from a JSON file.
    
    This function reads a JSON file from the specified folder path and file name, 
    and returns the contents of the JSON file as a dictionary.
    
    Parameters:
    folder_path (str): The path to the folder containing the JSON file.
    json_filename (str): The name of the JSON file (without extension).
    
    Returns:
    dict: A dictionary containing the contents of the JSON file.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    json.JSONDecodeError: If the file is not a valid JSON.
    
    Example:
    >>> json_contents = load_json("/path/to/json/folder", "json_file_name")
    """
    
    # Construct the full path to the JSON file
    json_filepath = Path(folder_path) / f"{json_filename}.json"
    
    # Ensure the JSON file exists
    if not json_filepath.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_filepath}")
    
    # Load and return the contents of the JSON file
    with open(json_filepath, 'r') as file:
        json_contents = json.load(file)
        
    return json_contents

def modify_json_values(file_path: str, changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a JSON file from the given path, modify its content according to the provided changes, 
    and return the modified JSON object.

    Parameters:
    - file_path (str): The path to the JSON file to be modified.
    - changes (dict): A dictionary containing the changes to be made where keys represent the
                      keys in the JSON file and values represent the new values to be set.

    Returns:
    - dict: The modified JSON object.

    Examples:
    >>> modify_json_values('data.json', {'key1': 'new_value', 'key2': 'another_value'})
    {'key1': 'new_value', 'key2': 'another_value', ...remaining content of the JSON...}
    """
    # Load the existing content from the JSON file
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Apply the changes to the JSON data
    for key, value in changes.items():
        # Assuming the keys are at the root level of the JSON structure
        # For nested updates, you would need a more complex approach
        if key in data:
            data[key] = value
        else:
            raise KeyError(f"The key {key} does not exist in the JSON data.")

    # Write the modified JSON data back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)





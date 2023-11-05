import json
import os
import sshtunnel
import pymysql
import pandas as pd
from sqlalchemy import create_engine
import requests
import datetime
import logging

from OwnLog import log_exceptions, log_function_args

def test_database_connection(
        ssh_host, ssh_port, ssh_user, ssh_password,
        db_server, db_user, db_password, database
    ):
    """
    Tests the SSH and database connections using the provided credentials.
    
    Parameters:
    - ssh_host (str): The hostname or IP address of the SSH server.
    - ssh_port (int): The port number of the SSH server.
    - ssh_user (str): The username to use for the SSH connection.
    - ssh_password (str): The password to use for the SSH connection.
    - db_server (str): The hostname or IP address of the database server.
    - db_user (str): The username to use for the database connection.
    - db_password (str): The password to use for the database connection.
    - database (str): The name of the database to connect to.
    
    Returns:
    - bool: True if both the SSH and database connections are successful, False otherwise.
    """
    try:
        with sshtunnel.SSHTunnelForwarder(
                (ssh_host, ssh_port),
                ssh_username=ssh_user,
                ssh_password=ssh_password,
                remote_bind_address=(db_server, 3306)
        ) as tunnel:
            conn = pymysql.connect(
                host='127.0.0.1', 
                user=db_user, 
                passwd=db_password, 
                db=database, 
                port=tunnel.local_bind_port
            )
            conn.close()
            return True
    except Exception as e:
        print(f"Connection to database failed: {e}")
        return False



def get_data_ssh(
    query,
    ssh_host,
    ssh_port,
    ssh_user,
    ssh_password,
    db_server,
    db_user,
    db_password,
    database
    ):
    with sshtunnel.SSHTunnelForwarder((ssh_host, ssh_port),
                                      ssh_username=ssh_user,
                                      ssh_password=ssh_password,
                                      remote_bind_address=(db_server, 3306)) as tunnel:
        _db = pymysql.connect(host='127.0.0.1',
                              user=db_user,
                              passwd=db_password,
                              db=database,
                              port=tunnel.local_bind_port)
        _cursor = _db.cursor()
        try:
            data = pd.read_sql(query, _db)
            return data
        except Exception as e:
            # Rollback in case there is any error
            print("Exception get_data_ssh")
            print(e)
            _db.rollback()
        _db.close()


def write_data_ssh(
    dataframe,
    table_name,
    ssh_host,
    ssh_port,
    ssh_user,
    ssh_password,
    db_server,
    db_user,
    db_password,
    database,
    if_exists='fail'
    ):
    with sshtunnel.SSHTunnelForwarder((ssh_host, ssh_port),
                                      ssh_username=ssh_user,
                                      ssh_password=ssh_password,
                                      remote_bind_address=(db_server, 3306)) as tunnel:
        _db = pymysql.connect(host='127.0.0.1', user=db_user, passwd=db_password, db=database, port=tunnel.local_bind_port)
        try:
            connection_string = f"mysql+pymysql://{db_user}:{db_password}@127.0.0.1:{tunnel.local_bind_port}/{database}"
            engine = create_engine(connection_string)

            dataframe.to_sql(table_name, engine, if_exists=if_exists, index=False)

        except Exception as e:
            print(e)
            _db.rollback()

        _db.close()

def update_value_ssh(
    table_name,
    column_to_update,
    new_value,
    identifier_columns,
    identifier_values,
    ssh_host,
    ssh_port,
    ssh_user,
    ssh_password,
    db_server,
    db_user,
    db_password,
    database,
):
    with sshtunnel.SSHTunnelForwarder((ssh_host, ssh_port),
                                      ssh_username=ssh_user,
                                      ssh_password=ssh_password,
                                      remote_bind_address=(db_server, 3306)) as tunnel:
        try:
            connection = pymysql.connect(host='127.0.0.1',
                                         user=db_user,
                                         password=db_password,
                                         port=tunnel.local_bind_port,
                                         database=database)
            with connection.cursor() as cursor:
                where_conditions = " AND ".join([f"{col} = %({col})s" for col in identifier_columns])
                update_query = f"""UPDATE {table_name}
                                   SET {column_to_update} = %({column_to_update})s
                                   WHERE {where_conditions}"""
                query_params = {**{column_to_update: new_value}, **dict(zip(identifier_columns, identifier_values))}
                cursor.execute(update_query, query_params)
                connection.commit()
        except Exception as e:
            print(e)
        finally:
            connection.close()


def get_data_api_time(_objectid,_channelid,_init,_end):
    """
    '%Y-%m-%d %H:%M:%S.%f'
    """
    try:
        headers = {'Accept': 'application/json'}
        a = requests.get('http://gurapi-emergency.guapeton.deep-insight.es/v1/data/data_bydate?projectid=DBEMERGENCY&acqid=unknown&objectid='+_objectid+'&channelid='+_channelid+'&init='+_init+'&end='+_end, headers = headers)
        print('http://gurapi-emergency.guapeton.deep-insight.es/v1/data/data_bydate?projectid=DBEMERGENCY&acqid=unknown&objectid='+_objectid+'&channelid='+_channelid+'&init='+_init+'&end='+_end)
        print(a.status_code)
        print(a.content)
        if a.status_code != 200:
            a.close()
        return a.json()
    except Exception as e:
        print(f"exception get_data_api_time: {e}")

@log_exceptions
@log_function_args
def data_importer(automatic_importation: bool, 
                  creds_path: str, 
                  path_instants_data_saved: str, 
                  database: str = None, 
                  query: str = None, 
                  save_importation=True,
                  file_name: str = None,
                  logger: bool = False) -> pd.DataFrame:
    """
    Import data from a SQL database or a local directory based on the given flag.
    
    This function provides a flexible way to import data either from a SQL database 
    or from a local directory containing CSV files. If importing from a database, 
    a credentials file and a SQL query are required. If importing from a local directory, 
    the function will look for the most recent CSV file in the specified directory.
    
    Parameters:
    ------------
    automatic_importation : bool
        If True, data will be imported from a SQL database.
        If False, data will be imported from a local directory.

    creds_path : str
        Path to the credentials file (JSON format) needed for accessing the SQL database.

    path_instants_save_data : str
        Path to the local directory from where data will be imported if automatic_importation is False.

    database : str, optional
        Name of the SQL database from where data will be imported. Not needed if automatic_importation is False.

    query : str, optional
        SQL query to select data from the SQL database. Not needed if automatic_importation is False.

    save_importation : bool, optional
        If True, and if data is imported from a database, the imported data will be saved 
        to a CSV file in the specified local directory. Default is True.

    file_name : str, optional
        Name of a specific file to import from the local directory. If None (default),
        the function will import the most recent file in the directory.
        Not needed if automatic_importation is True.
    logger : logging.Logger, optional
        The logger to use for logging messages. If None (default), no logging will be performed.
    Raises:
    --------
    ValueError:
        If automatic_importation is True but either database or query is None.
    
    Returns:
    ------------
    df : DataFrame
        Imported data as a pandas DataFrame.
    
    Examples:
    --------
    Importing data from a SQL database:
    >>> data_importer(
            automatic_importation=True,
            creds_path='credentials.json',
            path_instants_save_data='data/',
            database='my_database',
            query='SELECT * FROM my_table'
        )
    
    Importing data from a local directory:
    >>> data_importer(
            automatic_importation=False,
            creds_path='credentials.json',
            path_instants_save_data='data/'
        )  
    """

    # Start logging if logger is provided
    if logger:
        logger.info('Starting data import process.')

    # Check provided arguments for automatic importation
    if automatic_importation:

        if logger:
            logger.info('Importing data from SQL database.')

        if database is None or query is None:
            raise ValueError("Both database and query must be provided when automatic_importation is True")
        
        # Read credentials for SQL connection
        with open(creds_path, 'r') as json_file:
            creds_sql = json.load(json_file)

        # Fetch data from the SQL database
        df_imported = get_data_ssh(query, database=database, **creds_sql)

    else:
        
        # Determine the file to import based on the provided file_name
        if file_name:
            if logger:
                logger.info(f'Importing data from specified file: {file_name}')
            file_path = f"{path_instants_data_saved}/{file_name}"
        else:
            if logger:
                logger.info(f'Importing data from most recent file')
            # Get the name of the latest file in the specified directory
            dfs_saved = sorted(os.listdir(path_instants_data_saved))
            last_df = dfs_saved[-1]
            file_path = f"{path_instants_data_saved}/{last_df}"

        # Read data from the specified or most recent CSV file in the directory
        df_imported = pd.read_csv(file_path)
        
    # Automatically save the imported data from the database, if requested
    if save_importation and automatic_importation:
        if logger:
            logger.info(f'Data imported from database saved to {save_path}')
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"{path_instants_data_saved}/{name}.csv"
        df_imported.to_csv(save_path, index=False)
    
    # Create a copy of the imported DataFrame to prevent in-place modification
    df = df_imported.copy()

    if logger:
        logger.info('Data import process completed.')
    
    return df

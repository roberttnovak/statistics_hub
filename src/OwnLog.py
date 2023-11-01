import os
import logging
import inspect
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time
import functools


LOG_DIR = "logs/"

class ExecutionTimeFilter(logging.Filter):
    """
    A filter to add execution time to the log records.
    
    This filter calculates the execution time by comparing the current time
    with the time at which it was last called, and adds this information to
    the log records.
    
    Examples:
    --------
    >>> import logging
    >>> logger = logging.getLogger()
    >>> execution_time_filter = ExecutionTimeFilter()
    >>> logger.addFilter(execution_time_filter)
    >>> logger.info("Test log")
    """
    def __init__(self):
        self.start_time = time.time()  # Initialize start_time here

    def filter(self, record):
        record.execution_time = time.time() - self.start_time
        self.start_time = time.time()  # Update start_time here
        return True

class OwnLogger:
    """
    A class to create and manage a unified logger for recording events during the execution of a script and its functions.
    
    Attributes:
        script_name (str): The name of the script from which this class is instantiated.
        level (int): The logging level.
        log_rotation (str): The type of log rotation to use.
        max_bytes (int): The maximum log file size for size-based rotation, in bytes.
        backup_count (int): The number of backup log files to keep.
        when (str): The time when log rotation should occur for time-based rotation.
        _script_logger (logging.Logger): The configured logger for the script.
        
    Examples:
    --------
    >>> own_logger = OwnLogger()
    >>> logger = own_logger.get_logger()
    >>> logger.info("This is an info message from script")
    """

    def __init__(self, level=logging.INFO, log_rotation='size', max_bytes=10485760,
                 backup_count=10, when='midnight'):
        """
        Initializes a ScriptFunctionLogger instance.
        
        This constructor initializes a logger for the script based on the provided 
        or default configurations.
        
        Parameters
        ----------
        level : int, optional
            The logging level. Defaults to logging.INFO.
        log_rotation : str, optional
            The type of log rotation to use: 'size' for size-based rotation,
            or 'time' for time-based rotation. Defaults to 'size'.
        max_bytes : int, optional
            The maximum log file size for size-based rotation, in bytes.
            Defaults to 10MB.
        backup_count : int, optional
            The number of backup log files to keep. Defaults to 10.
        when : str, optional
            The time when log rotation should occur for time-based rotation.
            Defaults to 'midnight'.
            
        Examples
        --------
        >>> script_function_logger = ScriptFunctionLogger(level=logging.DEBUG, log_rotation='time', when='W6')
        """
        self.script_name = os.path.splitext(os.path.basename(inspect.getmodule(inspect.stack()[1][0]).__file__))[0]
        self.level = level
        self.log_rotation = log_rotation
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.when = when
        
        self._script_logger = self._create_logger(self.script_name)
    
    def _create_logger(self, logger_name):
        """
        Creates and configures a logger with the specified name.
        
        This is a helper method that encapsulates the logger creation logic.
        
        Parameters
        ----------
        logger_name : str
            The name of the logger.
            
        Returns
        -------
        logging.Logger
            A configured logger.
            
        Examples
        --------
        # This method is intended to be used internally,
        # but if needed it can be used as follows:
        >>> logger = script_function_logger._create_logger('custom_logger')
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
            
        if not logger.handlers:
            if self.log_rotation == 'size':
                handler = RotatingFileHandler(os.path.join(LOG_DIR, f'{logger_name}.log'), maxBytes=self.max_bytes, backupCount=self.backup_count)
            elif self.log_rotation == 'time':
                handler = TimedRotatingFileHandler(os.path.join(LOG_DIR, f'{logger_name}.log'), when=self.when, backupCount=self.backup_count)
            
            formatter = logging.Formatter(
                f'--------------------------------------------------\n'
                f'Timestamp:       %(asctime)s\n'
                f'Function Name:   %(funcName)s\n'
                f'Log Level:       %(levelname)s\n'
                f'Execution Time:  %(execution_time)s seconds\n'
                f'Message:         %(message)s\n'
                f'Thread Name:     %(threadName)s\n'
                f'Thread ID:       %(thread)d\n'
                f'Process Name:    %(processName)s\n'
                f'Process ID:      %(process)d\n'
            )
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            execution_time_filter = ExecutionTimeFilter()
            execution_time_filter.start_time = time.time()
            logger.addFilter(execution_time_filter)
            
        return logger
    
    def get_logger(self):
        """
        Gets the unified logger for the script.
        
        This method returns a logger configured for the script, 
        which can be used across different functions.
        
        Returns
        -------
        logging.Logger
            A configured logger.
            
        Examples
        --------
        >>> own_logger = OwnLogger()
        >>> script_logger = own_logger.get_logger()
        """
        return self._script_logger

def log_exceptions(func):
    """
    A decorator to log exceptions that occur in a function.
    
    This decorator wraps a function such that if an exception is raised during the execution of the 
    function, the exception is logged using the provided logger, and then re-raised.
    
    Parameters:
    ----------
    func : Callable
        The function to be decorated.
    
    Returns:
    -------
    Callable
        The decorated function with exception logging enabled.
    
    Examples:
    --------
    >>> @log_exceptions
    >>> def my_function(arg1, arg2, logger):
    >>>     return arg1 / arg2  # Throws ZeroDivisionError if arg2 is 0
    >>> 
    >>> script_logger = own_logger.get_logger()
    >>> my_function(10, 0, logger=script_logger)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = kwargs.get('logger')
            if logger:
                logger.exception(f"An error occurred in {func.__name__}: {e}")
            raise  # re-raise the exception after logging it
    return wrapper

def log_function_args(func):
    """
    Decorator to log the arguments of a function.
    
    This decorator wraps a function such that when the function is called, the names and values of 
    the arguments passed to the function are logged using the provided logger.
    
    Parameters:
    ----------
    func : Callable
        The function to be decorated.
    
    Returns:
    -------
    Callable
        The decorated function with argument logging enabled.
    
    Examples:
    --------
    >>> @log_function_args
    >>> def my_function(arg1, arg2, logger):
    >>>     pass  # Your code here
    >>> 
    >>> script_logger = own_logger.get_logger()
    >>> my_function("value1", "value2", logger=script_logger)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = kwargs.get('logger')
        if logger:
            arg_values = ', '.join([f"{a}" for a in args] + [f"{k}={v}" for k, v in kwargs.items()])
            logger.info(f"Calling {func.__name__} with arguments: {arg_values}")
        return func(*args, **kwargs)
    return wrapper



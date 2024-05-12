import os
import warnings
import json
import pandas as pd
import joblib 
import pickle

from own_utils import list_directories_by_depth
import plotly.io as pio

# ToDo: Create more functions for validating inputs like validate_string_input
# ToDo: Cuadrar save_predictions con save_preprocessed_data
# ToDo: Meter remove_object y también la funiconalidad en la clase de las flags crear/borrar flags
#       de procesado 

# supported extension to save/load objects
SUPPORTED_EXTENSIONS = ['joblib', 'pickle','json', 'csv']

class PersistenceManager:
    """
    A class to manage the saving and loading of machine learning models,
    scalers, preprocessed data, metadata, predictions and others
    """

    def __init__(self, base_path: str = None, folder_name_model: str = None, folder_name_range_train: str = None, folder_name_time_execution: str = None, folder_datasets: str = None):
        """
        Initializes the PersistenceManager with the provided base path, model name, training range, 
        execution time, and an optional datasets folder. The path for saving and loading objects 
        is constructed based on these parameters.

        Parameters:
        -----------
        base_path : str
            The base directory path where the model, associated data, and datasets will be saved.
        folder_name_model : str
            The name or identifier of the machine learning model to be managed.
        folder_name_range_train : str
            The range of data used for training the model, represented as a string.
        folder_name_time_execution : str
            The time taken for executing some process, possibly training the model, represented as a string.
        folder_datasets : str, optional (default: None)
            The folder where datasets are stored. If specified, self.path is set to base_path/folder_datasets.

        Examples:
        ---------
        >>> pm = PersistenceManager(
        ...     base_path=os.path.join("..", "models"),
        ...     folder_name_model='SVR',
        ...     folder_name_range_train='initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0',
        ...     folder_name_time_execution='execution-time-2023_09_27_13_24_30',
        ... )

        >>> pm = PersistenceManager(
        ...     base_path=os.path.join("..", "models"),
        ...     folder_datasets='data'
        ... )
        """
        self.base_path = os.path.normpath(base_path)
        self.folder_name_model = folder_name_model
        self.folder_name_range_train = folder_name_range_train
        self.folder_name_time_execution = folder_name_time_execution
        self.folder_datasets = folder_datasets
        
        # Use folder_datasets if it's not None, else construct the path with other components
        if folder_datasets is not None:
            self.path = os.path.normpath(os.path.join(base_path, folder_datasets)) if base_path is not None else folder_datasets
        else:
            # Construct the path excluding None values from other components
            path_components = [base_path, folder_name_model, folder_name_range_train, folder_name_time_execution]
            filtered_path_components = [component for component in path_components if component is not None]
            self.path = os.path.normpath(os.path.join(*filtered_path_components))

    def get_available_models(self, folder_name_predictions="predictions", include_predictions_only=False):
        """
        Scans the base directory and returns a list of available model names. If 'include_predictions_only' is True,
        it only returns the models which have a 'predictions' directory at any sublevel within their directory structure.

        Parameters:
        -----------
        folder_name_predictions : str, optional (default: "predictions")
            The name of the folder where predictions are stored.
        include_predictions_only : bool, optional (default: False)
            If True, only returns model names that have a 'predictions' directory at any sublevel.

        Returns:
        --------
        list
            A list of model names available in the base directory, optionally filtered to include only those 
            with a specified predictions directory at any sublevel.

        Examples:
        ---------
        >>> pm = PersistenceManager(...)
        >>> pm.get_available_models()
        ['model1', 'model2', ...]

        >>> pm.get_available_models(include_predictions_only=True)
        ['model1', ...] # Only models with predictions directory at any sublevel
        """
        try:
            models = set()
            for root, dirs, files in os.walk(self.base_path):
                for dir in dirs:
                    if include_predictions_only and dir == folder_name_predictions:
                        model = root.replace(self.base_path, "").split(os.sep)[1]
                        models.add(model)
                    elif not include_predictions_only:
                        model_path = os.path.normpath(os.path.join(self.base_path, dir))
                        if os.path.isdir(model_path):
                            models.add(dir)
            return list(models)
        except FileNotFoundError:
            return []


    def get_training_ranges_for_model(self, model_name):
        """
        Given a model name, scans the corresponding directory and returns a list of training ranges available for the model.

        Parameters:
        -----------
        model_name : str
            The name of the model to scan for training ranges.

        Returns:
        --------
        list
            A list of training ranges available for the specified model.

        Examples:
        ---------
        >>> pm = PersistenceManager(...)
        >>> pm.get_training_ranges_for_model('model1')
        ['range1', 'range2', ...]
        """
        model_path = os.path.normpath(os.path.join(self.base_path, model_name))
        try:
            return [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
        except FileNotFoundError:
            return []

    def get_execution_times_for_model_and_range(self, model_name, training_range):
        """
        Given a model name and a training range, scans the corresponding directory and returns a list of execution times available.

        Parameters:
        -----------
        model_name : str
            The name of the model.
        training_range : str
            The training range of the model.

        Returns:
        --------
        list
            A list of execution times available for the specified model and training range.

        Examples:
        ---------
        >>> pm = PersistenceManager(...)
        >>> pm.get_execution_times_for_model_and_range('model1', 'range1')
        ['execution1', 'execution2', ...]
        """
        range_path = os.path.normpath(os.path.join(self.base_path, model_name, training_range))
        try:
            return [name for name in os.listdir(range_path) if os.path.isdir(os.path.join(range_path, name))]
        except FileNotFoundError:
            return []

    def get_models_hierarchy_dict(self):
        """
        Lists all available models along with their training ranges and execution times. 
        This method organizes the information in a nested dictionary structure.

        Returns:
        --------
        dict
            A nested dictionary where each key is a model name, each value is a dictionary 
            where keys are training ranges, and each value is a list of execution times.

        Examples:
        ---------
        >>> pm = PersistenceManager(base_path='path/to/models')
        >>> model_details = pm.list_model_details()
        {
            'model1': {
                'range1': ['execution1', 'execution2', ...],
                'range2': ['execution1', 'execution2', ...],
                ...
            },
            'model2': {
                'range1': ['execution1', 'execution2', ...],
                ...
            },
            ...
        }
        """
        models_list = self.get_available_models()
        models_with_details = {}

        for model in models_list:
            ranges = self.get_training_ranges_for_model(model)
            models_with_details[model] = {}

            for range_ in ranges:
                execution_times = self.get_execution_times_for_model_and_range(model, range_)
                models_with_details[model][range_] = execution_times

        return models_with_details
    
    def get_models_hierarchy_list(self):
        """
        """
        return [
            (model, range_, execution)
            for model in self.get_available_models()
            for range_ in self.get_training_ranges_for_model(model)
            for execution in self.get_execution_times_for_model_and_range(model, range_)
        ]
    
    def get_models_info_as_dict(self, folder_name_predictions="predictions", include_predictions_only=False):
        """
        Recursively explores the directory structure from the base path to the model,
        training range, and execution times, compiling a list of dictionaries with information
        about each model, training range, and execution time. If 'include_predictions_only' is True,
        only returns information for paths within the 'folder_name_predictions' directory for each execution, if it exists.

        Parameters:
        -----------
        folder_name_predictions : str, optional (default: "predictions")
            The name of the folder where predictions are stored.
        include_predictions_only : bool, optional (default: False)
            If True, only return information for paths within the 'folder_name_predictions' directory for each execution.

        Returns:
        --------
        list
            A list of dictionaries for each model, training range, and execution time combination,
            optionally filtered to include only those within the 'folder_name_predictions' directory.

        Examples:
        ---------
        >>> pm = PersistenceManager(...)
        >>> models_info = pm.get_models_info_as_dict()
        [{'folder_name_model': 'model1', 'folder_name_range_train': 'range1', 'folder_name_time_execution': 'execution1'}, ...]

        >>> prediction_info = pm.get_models_info_as_dict(include_predictions_only=True)
        [{'folder_name_model': 'model1', 'folder_name_range_train': 'range1', 'folder_name_time_execution': 'predictions'}, ...]
        """
        model_path = self.base_path if self.folder_name_model is None else os.path.normpath(os.path.join(self.base_path, self.folder_name_model))

        return [
            {   
                "base_path": self.base_path, 
                "folder_name_model": model,
                "folder_name_range_train": range,
                "folder_name_time_execution": execution 
            }
            for model in self.get_available_models()
            for range in self.get_training_ranges_for_model(model)
            for execution in self.get_execution_times_for_model_and_range(model, range) if not include_predictions_only or os.path.exists(os.path.normpath(os.path.join(model_path, model, range, execution, folder_name_predictions)))
        ]
    
        
    def list_all_models(self):
        """
        Lists all available models by scanning the base directory.

        Returns:
        --------
        list
            A list of all available models.
        """
        return list_directories_by_depth(self.base_path, max_depth=1, list_only_last_level=True)

    def list_all_training_ranges(self):
        """
        Lists all training ranges for each model.

        Returns:
        --------
        dict
            A dictionary with model names as keys and a list of training ranges as values.
        """
        models = self.list_all_models()
        training_ranges = {}
        for model in models:
            model_path = os.path.join(self.base_path, model)
            training_ranges[model] = list_directories_by_depth(model_path, max_depth=1, list_only_last_level=True)
        return training_ranges

    def list_all_execution_times(self):
        """
        Lists all execution times for each model and training range.

        Returns:
        --------
        dict
            A nested dictionary with model names as keys, training ranges as sub-keys, 
            and a list of execution times as values.
        """
        training_ranges = self.list_all_training_ranges()
        execution_times = {}
        for model, ranges in training_ranges.items():
            execution_times[model] = {}
            for range_ in ranges:
                range_path = os.path.join(self.base_path, model, range_)
                execution_times[model][range_] = list_directories_by_depth(range_path, max_depth=1, list_only_last_level=True)
        return execution_times
    
    def list_evaluations(self, sub_folder="evaluations"):
        """
        Lists all evaluation files in a specified sub-folder within the 
        model directory structure.

        Parameters:
        -----------
        sub_folder : str, optional (default: "evaluations")
            The sub-folder within the model directory structure where 
            evaluation files are stored.

        Returns:
        --------
        list
            A list of file names found in the specified evaluation sub-folder.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> evaluation_files = pm.list_evaluations()
        ['evaluation1.csv', 'evaluation2.csv', ...]

        Notes:
        ------
        This method assumes that evaluation files are stored in a specific 
        sub-folder within the model's directory structure.
        """

        # Build the full path to the evaluations sub-folder
        evaluations_path = self.build_path(sub_folder=sub_folder)
        
        # Check if the evaluations path exists
        if not os.path.exists(evaluations_path):
            warnings.warn(f"Evaluations path '{evaluations_path}' does not exist.")
            return []

        # List all files in the evaluations sub-folder
        evaluation_files = [f for f in os.listdir(evaluations_path) if os.path.isfile(os.path.join(evaluations_path, f))]

        return evaluation_files
    
    def list_datasets(self):
        """
        Lists all dataset files within the folder_datasets directory.

        Returns:
        --------
        list
            A list of dataset file names available in the folder_datasets directory.

        Examples:
        ---------
        >>> pm = PersistenceManager(base_path='path/to/base', folder_datasets='datasets')
        >>> dataset_files = pm.list_datasets()
        ['dataset1.csv', 'dataset2.csv', ...]

        Notes:
        ------
        This method assumes that dataset files are stored in the folder_datasets directory.
        """
        # Ensure the folder_datasets attribute is set
        if self.folder_datasets is None:
            raise ValueError("The folder_datasets attribute is not set.")

        # Build the path to the datasets directory
        datasets_path = os.path.join(self.base_path, self.folder_datasets)

        # Check if the datasets path exists
        if not os.path.exists(datasets_path):
            raise FileNotFoundError(f"The specified datasets directory '{datasets_path}' does not exist.")

        # List all files in the datasets directory
        return [f for f in os.listdir(datasets_path) if os.path.isfile(os.path.join(datasets_path, f))]
    
    def list_datasets_with_structure(self):
        """
        Lists all dataset files within the folder_datasets directory, including those in subdirectories,
        and returns a nested dictionary representing the folder structure and files.

        Returns:
        --------
        dict
            A nested dictionary where each key is a folder and each value is recursively another dictionary
            for subfolders or an empty dictionary for files, representing the hierarchy of folders and datasets.

        Examples:
        ---------
        >>> pm = PersistenceManager(base_path='path/to/base', folder_datasets='datasets')
        >>> dataset_files = pm.list_datasets_with_structure()
        {
            'folder1': {'dataset1.csv': {}, 'dataset2.csv': {}},
            'folder2': {
                'folder3': {'dataset1.csv': {}, 'dataset2.csv': {}}
            }
        }

        Notes:
        ------
        This method assumes that dataset files are stored in the folder_datasets directory and possibly its subdirectories.
        The method is useful for displaying the directory and file structure in a file explorer interface, maintaining
        the hierarchy of directories and files.
        """
        if self.folder_datasets is None:
            raise ValueError("The folder_datasets attribute is not set.")

        datasets_path = os.path.join(self.base_path, self.folder_datasets)
        if not os.path.exists(datasets_path):
            raise FileNotFoundError(f"The specified datasets directory '{datasets_path}' does not exist.")

        dataset_list = []
        for root, _, files in os.walk(datasets_path):
            relative_path = os.path.relpath(root, datasets_path)
            path_components = tuple(relative_path.split(os.sep)) if relative_path != "." else ()
            for file in files:
                dataset_list.append(path_components + (file,))

        # Build the tree from the list of path components
        return self._build_tree(dataset_list)

    def _build_tree(self, paths):
        tree = {}
        for path in paths:
            current = tree
            for part in path[:-1]:  # Traverse through all parts except the last one
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[path[-1]] = {}  # The last part is the file
        return tree

    def list_all_files(self):
        """
        Lists all files within the base_path directory, including files in subdirectories.

        Returns:
        --------
        list
            A list of file paths relative to the base_path.

        Examples:
        ---------
        >>> pm = PersistenceManager(base_path='path/to/base')
        >>> files = pm.list_all_files()
        ['dir1/file1.ext', 'dir2/subdir/file2.ext', ...]

        Notes:
        ------
        This method does not require the full specification of model-related attributes
        and focuses only on the base_path provided during the initialization of the instance.
        """
        # Internal function to recursively walk through directory tree
        def walk_directory(directory):
            file_list = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    # Construct file path relative to the base_path
                    relative_path = os.path.relpath(os.path.join(root, file), self.base_path)
                    file_list.append(relative_path)
            return file_list

        # Call internal function with base_path
        return walk_directory(self.base_path)


    def validate_string_input(self,*args):
        """
        Validates the input arguments to ensure they are non-empty strings.
        Raises a ValueError if any of the arguments is not a non-empty string.

        Parameters:
        -----------
        *args : str
            The input arguments to validate.

        Raises:
        -------
        ValueError:
            If any of the arguments is not a non-empty string.

        Examples:
        ---------
        >>> validate_string_input("model_name", "train_range", "execution_time")
        # No exception is raised

        >>> validate_string_input("model_name", "", "execution_time")
        # Raises ValueError: All folder names must be non-empty strings
        """
        for arg in args:
            if not arg:
                raise ValueError("All folder names must be non-empty strings")

    def ensure_path(self, path, create=False):
        """
        Ensures a path exists. If the directory does not exist,
        it either creates it (if 'create' is True) or raises a NotADirectoryError 
        (if 'create' is False).
        
        Parameters:
        -----------
        path : str
            The directory path.
        create : bool, optional (default: False)
            Determines the behavior when the directory does not exist:
            - If True, the directory will be created.
            - If False, a NotADirectoryError will be raised.
                
        Returns:
        --------
        bool
            True if the path exists (or was created), False otherwise.
            
        Raises:
        -------
        NotADirectoryError:
            If the directory does not exist and 'create' option is set to False.
        """
        path_exists = os.path.exists(path)
        if not path_exists:
            if create:
                os.makedirs(path)
                path_exists = True  # Update path_exists to True as the path has now been created
            else:
                raise NotADirectoryError(f"Directory '{path}' does not exist and 'create' option is set to False.")
        return path_exists
    
    def build_path(self, sub_folder=None, file_name=None, extension=None):
        """
        Constructs a path string based on the provided sub-folder(s) and file name.
        This method uses the base path initialized in the PersistenceManager instance
        and appends the specified sub-folder(s) and file name to it, if provided.
        It normalizes the path to be system-independent.

        Parameters:
        -----------
        sub_folder : str or list of str, optional (default: None)
            The name(s) of the sub-folder(s) to append to the base path. If a list is
            provided, it represents nested sub-folders. If None, the base path
            remains unchanged.
        file_name : str, optional (default: None)
            The name of the file to append to the path. If an extension is also provided,
            it will be appended to the file name.
        extension : str, optional (default: None)
            The file extension to append to the file name. This parameter is ignored if
            file_name is None.

        Returns:
        --------
        str
            The constructed and normalized path string.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> path = pm.build_path(sub_folder=['preprocessed_data', 'scalers'], file_name='scaler', extension='joblib')
        # path will be normalized, e.g., 'model_name/train_range/execution_time/preprocessed_data/scalers/scaler.joblib'

        Notes:
        ------
        This method does not check if the constructed path exists or is valid. It merely
        constructs and normalizes a string based on the provided input and the base path.
        """
        # Start with the base path initialized in the PersistenceManager instance
        path = self.path
        
        # Append sub_folder(s) to the base path
        if sub_folder:
            if isinstance(sub_folder, list):
                sub_folder_path = os.path.join(*sub_folder)
            else:
                sub_folder_path = sub_folder
            path = os.path.join(path, sub_folder_path)
        
        # Append file_name and extension to the path
        if file_name:
            file_name = f"{file_name}.{extension}" if extension else file_name
            path = os.path.join(path, file_name)
        
        # Normalize the path to be system-independent
        return os.path.normpath(path)

    
    def save_object(self, obj, file_name, overwrite=False, extension='joblib', sub_folder=None):
        """
        Saves an object to disk. The method supports extensions defined in PersistenceManager.SUPPORTED_EXTENSIONS
        depending on the provided extension.

        Parameters:
        -----------
        obj : object
            The object to save. This should be a serializable object.
        file_name : str
            The name of the file (without extension) where the object will be saved.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        extension : str, optional (default: 'joblib')
            The extension method to use when saving defined in PersistenceManager.SUPPORTED_EXTENSIONS
        sub_folder : str, optional (default: None)
            Sub-folder under the specified path where the object will be saved.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If an unsupported extension is provided.
        FileNotFoundError:
            If the specified directory does not exist.
        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_object(obj=some_object, file_name='object_name', overwrite=True, extension='joblib', sub_folder = 'sub_folder')
        """

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension '{extension}'. Supported extensions are {', '.join(SUPPORTED_EXTENSIONS)}.")


        file_path = self.build_path(sub_folder = sub_folder, file_name=file_name, extension=extension)
        self.ensure_path(os.path.dirname(file_path),create = True)

        # Check if the file already exists and the overwrite flag
        if os.path.exists(file_path) and not overwrite:
            warnings.warn(f"{file_path} exists. The object will not be overwritten")
            return
        
        # Save the object to disk using the specified serialization method
        if extension == 'joblib': 
            with open(file_path, 'wb') as f:
                joblib.dump(obj, f)
        elif extension == 'pickle': 
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        elif extension == 'json':
            with open(file_path,'w') as f:
                json.dump(obj,f)
        elif extension == 'csv':
            if not isinstance(obj, pd.DataFrame):
                raise ValueError(f"Object must be a pandas DataFrame for CSV serialization, got {type(obj)}")
            obj.to_csv(file_path, sep=";", index = False)            
        else:
            raise ValueError(f"Unsupported extension '{extension}'.")
        
    def load_csv_as_raw_string(self, folder_path, filename, num_rows=None):
        """
        Loads a CSV file as a raw string from the specified directory, with an option to limit
        the number of rows read.

        This function is useful when the raw contents of a CSV file are needed,
        rather than parsing it into a DataFrame. It allows reading a limited number
        of rows from the CSV file.

        Parameters:
        -----------
        folder_path : str
            The folder path where the CSV file resides.
        filename : str
            The name of the CSV file. The '.csv' extension is optional and will be added if not present.
        num_rows : int, optional
            The number of rows to read from the CSV file. If None (default), all rows are read.

        Returns:
        --------
        str
            The raw contents of the specified number of rows from the CSV file as a string.

        Raises:
        -------
        FileNotFoundError:
            If the specified CSV file does not exist.

        Examples:
        ---------
        >>> pm = PersistenceManager(...)
        >>> raw_csv_string = pm.load_csv_as_raw_string('path/to/directory', 'data.csv', num_rows=10)
        """

        # Append '.csv' if not present in the filename
        if not filename.endswith('.csv'):
            filename += '.csv'

        # Construct the full path for the CSV file
        csv_file_path = os.path.normpath(os.path.join(self.base_path, folder_path, filename))

        # Check if the CSV file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"The specified CSV file {csv_file_path} does not exist.")

        # Read the CSV file as a raw string
        try:
            with open(csv_file_path, 'r') as file:
                if num_rows is not None:
                    raw_string = ''.join([next(file) for _ in range(num_rows)])
                else:
                    raw_string = file.read()
        except StopIteration:
            # Handle the case where the file has fewer rows than num_rows
            with open(csv_file_path, 'r') as file:
                raw_string = file.read()

        return raw_string



    def load_object(self, folder_path, filename, extension, csv_params=None):
        """
        Loads an object from disk based on the provided folder path, filename, and extension.
        The method supports multiple file formats defined in the constant SUPPORTED_EXTENSIONS.

        Parameters:
        -----------
        folder_path : str
            The folder path where the object file resides.
        filename : str
            The name of the file from where the object will be loaded.
        extension : str
            The file extension indicating the format of the object file.
        csv_params : dict, optional
            Additional parameters to read csv files. Same as pd.read_csv parameters

        Returns:
        --------
        object
            The deserialized object loaded from the disk.

        Raises:
        -------
        ValueError:
            If an unsupported file extension is provided.
        FileNotFoundError:
            If the specified file does not exist.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> obj = pm.load_object(folder_path='path/to', filename='file', extension='joblib')
        """

        # Validate the file extension against the list of supported extensions
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension '{extension}'. Supported extensions are {', '.join(SUPPORTED_EXTENSIONS)}.")

        # Combine folder path, filename, and extension to form the full path and normalize it
        full_path = os.path.normpath(os.path.join(folder_path, f"{filename}.{extension}"))

        # Check if the specified file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The specified file {full_path} does not exist.")

        # Load the object based on the given extension
        if extension == 'joblib':
            with open(full_path, 'rb') as f:
                obj = joblib.load(f)
        elif extension == 'pickle': 
            with open(full_path, 'rb') as f:
                obj = pickle.load(f)
        elif extension == 'json':
            with open(full_path, 'r') as f:
                obj = json.load(f)
        elif extension == 'csv':
            obj = pd.read_csv(full_path, **csv_params) if csv_params else pd.read_csv(full_path)
        else:
            # This else block is technically redundant due to the earlier check but kept for clarity.
            raise ValueError(f"Unsupported extension '{extension}'. Supported extensions are {', '.join(SUPPORTED_EXTENSIONS)}.")

        return obj

    def create_flag(self, flag_name, content='', sub_folder=None):
        """
        Creates a flag file with the specified name and content in a specified sub-folder
        or at the root level of the model directory structure. This flag is useful to indicate
        if some process is done or not and do or not other simultanous actions. 

        Parameters:
        -----------
        flag_name : str
            The name of the flag file. This name will be used to create the file name along with the '.txt' extension.
        content : str, optional (default: '')
            The content to write to the flag file.
        sub_folder : str or list of str, optional (default: None)
            The name(s) of the sub-folder(s) to append to the base path. If a list is
            provided, it represents nested sub-folders. If None, the flag file will be
            created at the root level of the model directory structure.

        Returns:
        --------
        None

        Raises:
        -------
        FileNotFoundError:
            If the specified directory does not exist and could not be created.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.create_flag('training-done')
        >>> pm.create_flag('evaluation-done', sub_folder='evaluations')
        """
        # Construct the path for the flag file
        flag_file_path = self.build_path(sub_folder=sub_folder, file_name=flag_name, extension='txt')
        # Ensure the directory exists, create it if it doesn't
        self.ensure_path(os.path.dirname(flag_file_path), create=True)
        # Write the content to the flag file
        with open(flag_file_path, 'w') as f:
            f.write(content)

    def remove_flag(self, flag_name, sub_folder=None):
        """
        Removes a flag file with the specified name from a specified sub-folder
        or from the root level of the model directory structure.

        Parameters:
        -----------
        flag_name : str
            The name of the flag file. This name will be used to create the file name along with the '.txt' extension.
        sub_folder : str or list of str, optional (default: None)
            The name(s) of the sub-folder(s) to append to the base path. If a list is
            provided, it represents nested sub-folders. If None, the flag file will be
            removed from the root level of the model directory structure.

        Returns:
        --------
        None

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.remove_flag('training_completed', sub_folder='flags')
        """
        # Construct the path for the flag file
        flag_file_path = self.build_path(sub_folder=sub_folder, file_name=flag_name, extension='txt')
        # Check if the flag file exists, if not issue a warning
        if not os.path.exists(flag_file_path):
            warnings.warn(f"The specified flag file {flag_file_path} does not exist.")
        else:
            # Remove the flag file
            os.remove(flag_file_path)

    def flag_exists(self, flag_name, sub_folder=None):
        """
        Checks if a flag file with the specified name exists in a specified sub-folder
        or at the root level of the model directory structure.

        Parameters:
        -----------
        flag_name : str
            The name of the flag file. This name will be used to create the file name along with the '.txt' extension.
        sub_folder : str or list of str, optional (default: None)
            The name(s) of the sub-folder(s) to append to the base path. If a list is
            provided, it represents nested sub-folders. If None, the flag file will be
            checked at the root level of the model directory structure.

        Returns:
        --------
        bool
            True if the flag file exists, False otherwise.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.flag_exists('training_completed', sub_folder='flags')
        True
        """
        # Construct the path for the flag file
        flag_file_path = self.build_path(sub_folder=sub_folder, file_name=flag_name, extension='txt')

        # Check if the flag file exists and return the result
        return os.path.exists(flag_file_path)


    def save_model(self, model, name="model", overwrite=False, extension="joblib"):
        """
        Saves a machine learning model to disk using joblib. The model is saved 
        in a directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure will be as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>.

        If the directory does not exist, it will be created. If a file with the 
        same name already exists in the directory, a warning will be issued 
        unless the overwrite flag is set to True.

        Parameters:
        -----------
        model : object
            The machine learning model to save. This should be a trained model 
            object compatible with joblib.
        name : str, optional (default: "model")
            The name of the model file. This name will be used to create the 
            file name along with the '.joblib' extension.
        overwrite : bool, optional (default: False)
            Determines the behavior when a file with the same name already 
            exists in the directory:
            - If True, the existing file will be overwritten.
            - If False, a warning will be issued and the file will not be 
              overwritten.
        extension : str, optional (default: 'joblib')
            The serialization method to use. 

        Returns:
        --------
        None

        Raises:
        -------
        NotADirectoryError:
            If the directory does not exist and the 'create' option in 
            ensure_path is set to False.

        Examples:
        ---------
        >>> pm = PersistenceManager('SVR', 'train_range', 'execution_time')
        >>> pm.save_model(trained_model, name='svr_model', overwrite=True)

        Notes:
        ------
        This method uses the `save_object` method to save the model object to disk.
        Ensure that the model object is compatible with joblib.

        """
        # Save the model object to disk using the save_object method
        self.save_object(obj=model, file_name=name, overwrite=overwrite, extension=extension)



    def load_model(self, name="model", extension="joblib"):
        """
        Loads a machine learning model from disk using joblib. The model is expected
        to be saved in a directory structure based on the attributes initialized in the
        PersistenceManager instance. The directory structure should be as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>.

        Parameters:
        -----------
        name : str, optional (default: "model")
            The name of the model file (without extension). This name should match
            the name used when saving the model using `save_model`.
        extension : str, optional (default: "joblib")
            The file extension of the model file. This should match the extension
            used when saving the model.

        Returns:
        --------
        object
            The loaded machine learning model.

        Raises:
        -------
        ValueError:
            If an unsupported extension is provided.
        FileNotFoundError:
            If the specified model file does not exist.

        Examples:
        ---------
        >>> pm = PersistenceManager('SVR', 'train_range', 'execution_time')
        >>> loaded_model = pm.load_model(name='model')

        Notes:
        ------
        This method uses the `load_object` method to load the model object from disk. 
        Ensure that the model file is compatible with joblib and the specified name 
        and extension match the file on disk.

        """
        # The path where the model is saved is constructed based on the 
        # initialized attributes in the PersistenceManager instance
        path = self.build_path()
        
        # Load the model object from disk using the load_object method
        model = self.load_object(folder_path=path, filename = name, extension=extension)
        
        return model

    def save_scaler(self, scaler, folder_name_preprocessed_data="preprocessed-data-to-use-in-model",
                     name='scaler', overwrite=False, extension="joblib"):
        """
        Saves a scaler object to disk using joblib. The scaler is saved in a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_preprocessed_data>.

        Parameters:
        -----------
        scaler : object
            The scaler object to save. This should be a scikit-learn scaler 
            object or any object that can be serialized with joblib.
        folder_name_preprocessed_data : str, optional (default: "preprocessed-data-to-use-in-model")
            The folder name where the preprocessed data and scaler will be saved.
        name : str, optional (default: 'scaler')
            The name of the file (without extension). This name should match
            the name used when loading the scaler using `load_scaler`.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        extension : str, optional (default: joblib)
            The serialization method to use. 

        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the specified folder name for preprocessed data is not a non-empty string.
        NotADirectoryError:
            If the directory does not exist and the 'create' option in `ensure_path` is set to False.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_scaler(scaler_object, 'preprocessed_data_folder')

        Notes:
        ------
        This method uses the `save_object` method to save the scaler object to disk.
        Ensure that the scaler object is compatible with joblib and the specified name is unique
        to avoid overwriting existing files unless intended.

        """
        # Save the model object to disk using the save_object method
        self.save_object(obj=scaler, sub_folder=folder_name_preprocessed_data, file_name=name, overwrite=overwrite, extension=extension)

        
    def load_scaler(self, name="scaler", folder_name_preprocessed_data="preprocessed-data-to-use-in-model", extension="joblib"):
        """
        Loads a scaler object from disk using joblib. The scaler is loaded from a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_preprocessed_data>.

        Parameters:
        -----------
        name : str, optional (default: 'scaler')
            The name of the file (without extension). This name should match
            the name used when saving the scaler using `save_scaler`.
        folder_name_preprocessed_data : str, optional (default: "preprocessed-data-to-use-in-model")
            The folder name where the preprocessed data and scaler are saved.
        extension : str, optional (default: 'joblib')
            The file extension of the scaler file.

        Returns:
        --------
        object
            The loaded scaler object.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> scaler = scaler = pm.load_scaler(name='custom_scaler_name', extension='joblib')

        Notes:
        ------
        This method uses the `load_object` method to load the scaler object from disk. Ensure that
        the specified folder name, file name, and extension match the values used
        when saving the scaler.

        Raises:
        -------
        FileNotFoundError:
            If the specified file does not exist.
        """

        # Build the full path for the scaler file using the `build_path` method
        # This will create a path string based on the provided sub-folder name, file name, and extension
        scaler_file_path = self.build_path(sub_folder=folder_name_preprocessed_data)
        
        # Load the scaler object from disk using the `load_object` method
        # This will handle the file loading
        scaler = self.load_object(folder_path = scaler_file_path, filename=name, extension=extension)

        return scaler


    def save_metadata(self, metadata, name='metadata', overwrite=False):
        """
        Saves metadata to disk using json. The metadata is saved in a directory 
        structure based on the attributes initialized in the PersistenceManager 
        instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>

        Parameters:
        -----------
        metadata : dict
            The metadata to save. This should be a dictionary containing key-value 
            pairs describing the model's parameters or other relevant information.
        folder_name_metadata : str, optional (default: "metadata")
            The folder name where the metadata will be saved.
        name : str, optional (default: 'metadata')
            The name of the file (without extension). This name should match
            the name used when loading the metadata using `load_metadata`.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the specified folder name for metadata is not a non-empty string.
        NotADirectoryError:
            If the directory does not exist and the 'create' option in `ensure_path` 
            is set to False.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_metadata({'param1': value1, 'param2': value2})

        Notes:
        ------
        This method uses json to save the metadata to disk. Ensure that
        the metadata is a dictionary and the specified name is unique
        to avoid overwriting existing files unless intended.

        """
        # Save the metadata object to disk using the save_object method
        self.save_object(obj=metadata, file_name=name, overwrite=overwrite, extension='json')

    
    def load_metadata(self, name="metadata", extension = 'json'):
        """
        Loads metadata from disk using json. The metadata is loaded from a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>

        Parameters:
        -----------
        name : str, optional (default: 'metadata')
            The name of the file (without extension). This name should match
            the name used when saving the metadata using `save_metadata`.
        extension : str, optional (default: 'json')
            The file extension of the metadata file. This should match the extension
            used when saving the metadata using `save_metadata`.

        Returns:
        --------
        dict
            The loaded metadata dictionary.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> metadata = pm.load_metadata(name='custom_metadata_name')

        Notes:
        ------
        This method uses json to load the metadata from disk. Ensure that
        the specified folder name, file name, and extension match the values used
        when saving the metadata.

        Raises:
        -------
        FileNotFoundError:
            If the specified file does not exist.
        """

        # Build the full path for the metadata file using the `build_path` method
        # This will create a path string based on the provided sub-folder name and file name
        metadata_file_path = self.build_path()
        
        # Load the metadata object from disk using the `load_object` method
        # This will handle the file loading
        metadata = self.load_object(folder_path = metadata_file_path, filename=name, extension = extension)

        return metadata


    def save_preprocessed_data(self, preprocessed_data, name, 
                               folder_name_preprocessed_data="preprocessed-data-to-use-in-model", 
                               overwrite=False, extension = 'csv'):
        """
        Saves preprocessed data to disk in CSV format. The data is saved in a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_preprocessed_data>.

        Parameters:
        -----------
        preprocessed_data : pd.DataFrame
            The preprocessed data to save.
        name : str
            The name of the file (without extension)
        folder_name_preprocessed_data : str, optional (default: "preprocessed-data-to-use-in-model")
            The folder name where the preprocessed data will be saved.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        extension : str, optional (default: 'csv')
            The file extension to use when saving the preprocessed data.

        Returns:
        --------
        None

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_preprocessed_data(preprocessed_data_df, name='custom_preprocessed_data_name')

        """

        self.save_object(obj = preprocessed_data, sub_folder = folder_name_preprocessed_data, file_name=name, overwrite=overwrite, extension=extension)

    def load_preprocessed_data(self, name, folder_name_preprocessed_data="preprocessed-data-to-use-in-model",
                               extension='csv', datetime_columns=[], utc=True):
        """
        Loads preprocessed data from disk in CSV format. The data is loaded from a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_preprocessed_data>.

        Parameters:
        -----------
        name : str
            The name of the file (without extension).
        folder_name_preprocessed_data : str, optional (default: "preprocessed-data-to-use-in-model")
            The folder name where the preprocessed data is saved.
        extension : str, optional (default: 'csv')
            The file extension of the preprocessed data file.
        datetime_columns : list, optional
            List of columns to be converted to datetime.
        utc : bool, optional
            Whether to convert datetime columns to UTC. Default is True.

        Returns:
        --------
        pd.DataFrame
            The loaded preprocessed data.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> preprocessed_data_df = pm.load_preprocessed_data(name='custom_preprocessed_data_name')

        """
        
        # Load the preprocessed data object from disk using the load_object method
        #ToDo: Hacer el build_path
        path = os.path.join(self.path,folder_name_preprocessed_data)
        preprocessed_data = self.load_object(folder_path=path, filename = name, extension=extension)

        # Check and convert datetime columns
        if utc:
            for col in datetime_columns:
                preprocessed_data[col] = pd.to_datetime(preprocessed_data[col]).dt.tz_convert('UTC')

        return preprocessed_data


    def save_predictions(self, predictions, name, folder_name_predictions="predictions", 
                         overwrite=False, extension = 'csv'):
        """
        Saves predictions data to disk in CSV format. The data is saved in a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_predictions>.

        Parameters:
        -----------
        predictions : pd.DataFrame
            The predictions data to save.
        name : str
            The name of the file (without extension).        
        folder_name_predictions : str, optional (default: "predictions")
            The folder name where the predictions data will be saved.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        extension : str, optional (default: 'csv')
            The file extension to use when saving the predictions data.

        Returns:
        --------
        None

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_predictions(predictions_df, 'custom_predictions_name')

        """
        self.save_object(obj = predictions, file_name=name, sub_folder = folder_name_predictions, overwrite=overwrite, extension=extension)

    def load_predictions(self, name, folder_name_predictions="predictions",
                         extension='csv', datetime_columns=[], utc=True):
        """
        Loads predictions data from disk in CSV format. The data is loaded from a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_predictions>.

        Parameters:
        -----------
        name : str, optional
            The name of the file (without extension).
        folder_name_predictions : str, optional (default: "predictions")
            The folder name where the predictions data is saved.
        extension : str, optional (default: 'csv')
            The file extension of the predictions data file.
        datetime_columns : list, optional
            List of columns to be converted to datetime.
        utc : bool, optional
            Whether to convert datetime columns to UTC. Default is True.

        Returns:
        --------
        pd.DataFrame
            The loaded predictions data.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> predictions_df = pm.load_predictions(name='custom_predictions_name')

        """
        # Load the preprocessed data object from disk using the load_object method
        #ToDo: Hacer el build_path
        path = os.path.join(self.path, folder_name_predictions)
        predictions_df = self.load_object(folder_path=path, filename = name, extension=extension)

        # Convert utc columns
        if utc:
            for col in datetime_columns:
                predictions_df[col] = pd.to_datetime(predictions_df[col]).dt.tz_convert('UTC')
        
        return predictions_df

    def save_evaluation_data(self, evaluation_data, name, 
                               folder_name_evaluation="evaluations", 
                               overwrite=False, extension = 'csv'):
        """
        Saves evaluation data of a model to disk in CSV format. The data is saved in a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_evaluation_data>.

        Parameters:
        -----------
        evaluation_data : pd.DataFrame
            The evaluation data of the model to save.
        name : str
            The name of the file (without extension)
        folder_name_evaluation : str, optional (default: "preprocessed-data-to-use-in-model")
            The folder name where the preprocessed data will be saved.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        extension : str, optional (default: 'csv')
            The file extension to use when saving the preprocessed data.

        Returns:
        --------
        None

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_preprocessed_data(evaluation_data_df, name='custom_preprocessed_data_name')

        """
        self.save_object(obj = evaluation_data, sub_folder = folder_name_evaluation, file_name=name, overwrite=overwrite, extension=extension)

    def load_evaluation_data(self, name, folder_name_evaluation_data="evaluations",
                               extension='csv', datetime_columns=[], utc=True):
        """
        Loads evaluated data from disk in CSV format. The data is loaded from a 
        directory structure based on the attributes initialized in the 
        PersistenceManager instance. The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_evaluation_data>.

        Parameters:
        -----------
        name : str
            The name of the file (without extension).
        folder_name_evaluation_data : str, optional (default: "evaluations")
            The folder name where the preprocessed data is saved.
        extension : str, optional (default: 'csv')
            The file extension of the preprocessed data file.
        datetime_columns : list, optional
            List of columns to be converted to datetime.
        utc : bool, optional
            Whether to convert datetime columns to UTC. Default is True.

        Returns:
        --------
        pd.DataFrame
            The loaded preprocessed data.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> evaluation_data_df = pm.load_evaluation_data(name='custom_evaluation_data_name')

        """
        
        # Load the preprocessed data object from disk using the load_object method
        #ToDo: Hacer el build_path
        path = os.path.join(self.path,folder_name_evaluation_data)
        evaluation_data = self.load_object(folder_path=path, filename = name, extension=extension)

        # Check and convert datetime columns
        if utc:
            for col in datetime_columns:
                evaluation_data[col] = pd.to_datetime(evaluation_data[col]).dt.tz_convert('UTC')

        return evaluation_data
    
    def save_dataset(self, df, file_name, overwrite=False, extension='csv'):
        """
        Saves a dataset (pandas DataFrame) using the save_object method.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to save.
        file_name : str
            The name of the file (without extension) where the dataset will be saved.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        extension : str, optional (default: 'csv')
            The file extension/format to use for saving the dataset.

        Returns:
        --------
        None

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_dataset(df=dataframe, file_name='dataset_name')
        """
        if extension not in ['csv', 'json']:  # Add more supported formats if needed
            raise ValueError(f"Unsupported extension '{extension}' for datasets.")

        # Use the save_object method to save the dataset
        self.save_object(df, file_name, overwrite=overwrite, extension=extension, sub_folder=self.folder_datasets)

    def load_dataset(self, file_name, extension='csv', csv_params = None):
        """
        Loads a dataset into a pandas DataFrame using the load_object method.

        Parameters:
        -----------
        file_name : str
            The name of the file (without extension) to load the dataset from.
        extension : str, optional (default: 'csv')
            The file extension/format of the dataset.
        csv_params : dict, optional
            params of csv file. Same as parameters in pd.read_csv

        Returns:
        --------
        pandas.DataFrame
            The loaded dataset as a pandas DataFrame.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> df = pm.load_dataset(file_name='dataset_name')
        """
        if extension not in ['csv', 'json']:  # Add more supported formats if needed
            raise ValueError(f"Unsupported extension '{extension}' for datasets.")

        # Build the folder path for datasets
        # folder_path = os.normpath(os.path.join(self.path, self.folder_datasets)) if self.folder_datasets else self.path

        # Use the load_object method to load the dataset
        return self.load_object(self.path, file_name, extension, csv_params = csv_params)


    
    def save_plotly_visualization(self, visualization, name, folder_name_visualizations="visualizations", 
                                overwrite=False, format='html'):
        """
        Saves a Plotly visualization to disk in the specified format. The visualization is saved 
        in a directory structure based on the attributes initialized in the PersistenceManager instance. 
        The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_visualizations>.

        Parameters:
        -----------
        visualization : plotly.graph_objs._figure.Figure
            The Plotly visualization object to save.
        name : str
            The name of the file (without extension).
        folder_name_visualizations : str, optional (default: "visualizations")
            The folder name where the visualization will be saved.
        overwrite : bool, optional (default: False)
            Whether to overwrite the file if it already exists.
        format : str, optional (default: 'html')
            The format in which to save the visualization ('html' or 'json').

        Returns:
        --------
        None

        Raises:
        -------
        ValueError:
            If the specified format is not supported.
        NotADirectoryError:
            If the directory does not exist and the 'create' option in `ensure_path` 
            is set to False.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> pm.save_plotly_visualization(visualization_figure, 'visualization_name')
        """

        supported_formats = ['html', 'json']
        if format not in supported_formats:
            raise ValueError(f"Unsupported format '{format}'. Supported formats are {', '.join(supported_formats)}.")

        # Build the path where the visualization will be saved
        visualization_path = self.build_path(sub_folder=folder_name_visualizations, file_name=name, extension=format)
        
        # Ensure the directory exists, create it if it doesn't
        self.ensure_path(os.path.dirname(visualization_path), create=True)

        # Save the visualization based on the chosen format
        if format == 'html':
            visualization.write_html(visualization_path)
        elif format == 'json':
            with open(visualization_path, 'w') as f:
                json.dump(visualization.to_plotly_json(), f)

    def load_plotly_visualization(self, name, folder_name_visualizations="visualizations", format='html'):
        """
        Loads a Plotly visualization from disk in the specified format. The visualization is expected 
        to be saved in a directory structure based on the attributes initialized in the PersistenceManager instance. 
        The directory structure is as follows:
        <folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>/<folder_name_visualizations>.

        Parameters:
        -----------
        name : str
            The name of the file (without extension).
        folder_name_visualizations : str, optional (default: "visualizations")
            The folder name where the visualization is saved.
        format : str, optional (default: 'html')
            The format in which the visualization was saved ('html' or 'json').

        Returns:
        --------
        plotly.graph_objs._figure.Figure
            The loaded Plotly visualization object.

        Raises:
        -------
        ValueError:
            If the specified format is not supported.
        FileNotFoundError:
            If the specified visualization file does not exist.

        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> visualization_figure = pm.load_plotly_visualization('visualization_name', format='html')
        """

        supported_formats = ['html', 'json']
        if format not in supported_formats:
            raise ValueError(f"Unsupported format '{format}'. Supported formats are {', '.join(supported_formats)}.")

        # Build the path where the visualization is saved
        visualization_path = self.build_path(sub_folder=folder_name_visualizations, file_name=name, extension=format)

        # Check if the specified file exists
        if not os.path.exists(visualization_path):
            raise FileNotFoundError(f"The specified visualization file {visualization_path} does not exist.")

        # Load the visualization based on the chosen format
        if format == 'html':
            return pio.read_html(visualization_path)
        elif format == 'json':
            with open(visualization_path, 'r') as f:
                visualization_json = json.load(f)
                return pio.from_json(visualization_json)


    # def persist_model_to_disk_structure(
    #     self,
    #     path_to_save_model,
    #     folder_name_model,
    #     folder_name_range_train,
    #     folder_name_time_execution,
    #     model,
    #     metadata,
    #     scaler,
    #     folder_name_preprocessed_data,
    #     preprocessed_data=None,
    #     additional_persistable_objects_to_save=None
    # ):
    #     """
    #     Saves the model structure and associated objects to disk.

    #     Parameters:
    #     -----------
    #     path_to_save_model : str
    #         Base path to save the model.
    #     folder_name_model : str
    #         Name of the folder that will contain the model.
    #     folder_name_range_train : str
    #         Name of the folder representing the training range.
    #     folder_name_time_execution : str
    #         Name of the folder representing the execution time.
    #     model : object
    #         Model object to save.
    #     metadata : dict
    #         Metadata associated with the model.
    #     scaler : object
    #         Data scaling object.
    #     folder_name_preprocessed_data : str
    #         Name of the folder for the preprocessed data.
    #     preprocessed_data : dict, optional
    #         Dictionary with preprocessed data to save. Keys are file names and values are the corresponding data.
    #     additional_persistable_objects_to_save : list, optional
    #         List of additional PersistableItem objects to save.

    #     File and Folder Structure:
    #     -------------------------
    #         path_to_save_model/
    #         ├── folder_name_model/
    #         │   ├── folder_name_range_train/
    #         │   │   ├── folder_name_time_execution/
    #         │   │   │   ├── model (file)
    #         │   │   │   ├── metadata.json (file)
    #         │   │   │   └── folder_name_preprocessed_data/
    #         │   │   │       ├── scaler (file)
    #         │   │   │       ├── preprocessed data (if provided)
    #         │   │   │       └── additional files (if any)
    #         │   │   │   ├── training-done.txt (file)  # New line

    #     Description:
    #     ------------
    #     The function constructs the necessary paths to save the model structure and associated data
    #     at the specified location. It handles directory creation and saving of relevant objects
    #     (model, metadata, scaler, and preprocessed data). Preprocessed data and additional objects
    #     are saved only if provided.
        
    #     A file named `training-done.txt` is created in the `folder_name_time_execution` directory 
    #     as a flag to indicate that the model training has been completed successfully. This can be 
    #     useful for other processes or systems that may be monitoring the model training process, 
    #     to know when the training and saving tasks have been completed.
    #     """
    #     # Set default values for optional parameters
    #     preprocessed_data = preprocessed_data or {}
    #     additional_persistable_objects_to_save = additional_persistable_objects_to_save or []

    #     # Create the full path to save the model
    #     full_path_model = os.path.join(
    #         path_to_save_model, folder_name_model,
    #         folder_name_range_train, folder_name_time_execution
    #     )

    #     # Create the full path to save the preprocessed data
    #     full_path_preprocessed_data_of_model = os.path.join(
    #         full_path_model, folder_name_preprocessed_data
    #     )

    #     # Call existing methods to save model, metadata, and scaler
    #     self.save_model(model)
    #     self.save_metadata(metadata)
    #     self.save_scaler(scaler)

    #     # Save each preprocessed data item
    #     for name, data in preprocessed_data.items():
    #         self.save_preprocessed_data(data, full_path_preprocessed_data_of_model, name)

    #     # Save any additional persistable objects
    #     for obj in additional_persistable_objects_to_save:
    #         obj.save()

    #     # Create training-done.txt to indicate training completion
    #     with open(os.path.join(full_path_model, 'training-done.txt'), 'w') as file:
    #         file.write('Training completed successfully.\n')


def persist_model_to_disk_structure(
    path_to_save_model,
    folder_name_model,
    folder_name_range_train,
    folder_name_time_execution,
    model,
    metadata,
    scaler,
    folder_name_preprocessed_data=None,
    preprocessed_data=None,
    additional_persistable_objects_to_save=None
):
    """
    Saves the model structure and associated objects to disk.

    Parameters:
    -----------
    path_to_save_model : str
        Base path to save the model.
    folder_name_model : str
        Name of the folder that will contain the model.
    folder_name_range_train : str
        Name of the folder representing the training range.
    folder_name_time_execution : str
        Name of the folder representing the execution time.
    model : object
        Model object to save.
    metadata : dict
        Metadata associated with the model.
    scaler : object
        Data scaling object.
    folder_name_preprocessed_data : str, optional
        Name of the folder for the preprocessed data. If not provided, no preprocessed data will be saved.
    preprocessed_data : dict, optional
        Dictionary with preprocessed data to save. Keys are file names and values are the corresponding data.
    additional_persistable_objects_to_save : list, optional
        List of additional PersistableItem objects to save.

    File and Folder Structure:
    -------------------------
        path_to_save_model/
        ├── folder_name_model/
        │   ├── folder_name_range_train/
        │   │   ├── folder_name_time_execution/
        │   │   │   ├── model (file)
        │   │   │   ├── metadata.json (file)
        │   │   │   └── folder_name_preprocessed_data/
        │   │   │       ├── scaler (file)
        │   │   │       ├── preprocessed data (if provided)
        │   │   │       └── additional files (if any)
        │   │   │   ├── training-done.txt (file)  # New line

    Description:
    ------------
    The function constructs the necessary paths to save the model structure and associated data
    at the specified location. It handles directory creation and saving of relevant objects
    (model, metadata, scaler, and preprocessed data). Preprocessed data and additional objects
    are saved only if provided.
    
    A file named `training-done.txt` is created in the `folder_name_time_execution` directory 
    as a flag to indicate that the model training has been completed successfully. This can be 
    useful for other processes or systems that may be monitoring the model training process, 
    to know when the training and saving tasks have been completed.
    
    Error handling is performed to ensure that any issues encountered during the persistence 
    process are captured and logged, providing easier debugging and ensuring the integrity of 
    the saved data.
    """
    # Set default values for optional parameters
    preprocessed_data = preprocessed_data or {}
    additional_persistable_objects_to_save = additional_persistable_objects_to_save or []

    pm = PersistenceManager(
        base_path = path_to_save_model,
        folder_name_model = folder_name_model, 
        folder_name_range_train = folder_name_range_train, 
        folder_name_time_execution = folder_name_time_execution
    )

    pm.save_model(model)
    pm.save_metadata(metadata)
    pm.save_scaler(scaler)

    pm.create_flag("training-done")

    # Save each preprocessed data item
    if folder_name_preprocessed_data:
        for name, data in preprocessed_data.items():
            pm.save_preprocessed_data(preprocessed_data=data, folder_name_preprocessed_data=folder_name_preprocessed_data, name=name)

    pm.create_flag("preprocessing-done")

    # Save any additional persistable objects
    #ToDo update this part of function
    # for obj in additional_persistable_objects_to_save:
    #     obj.save()

    # # ToDo: Next part of code is obsolete. Review with testing if can I remove or not 
    # Create training-done.txt to indicate training completion
    # with open(os.path.join(pm.path, 'training-done.txt'), 'w') as file:
    #     file.write('Training completed successfully.\n')

import os
import warnings
import json
import pandas as pd
import joblib 
import pickle

from own_utils import list_directories_by_depth

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

    def __init__(self, base_path: str = None ,folder_name_model : str = None, folder_name_range_train: str= None, folder_name_time_execution: str= None):
        """
        Initializes the PersistenceManager with the provided base path, model name, training range, 
        and execution time, which are used to construct a path for saving and loading objects.
        The constructed path will have the structure: 
        <base_path>/<folder_name_model>/<folder_name_range_train>/<folder_name_time_execution>.

        Parameters:
        -----------
        base_path : str
            The base directory path where the model and associated data will be saved.
        folder_name_model : str
            The name or identifier of the machine learning model to be managed.
        folder_name_range_train : str
            The range of data used for training the model, represented as a string.
        folder_name_time_execution : str
            The time taken for executing some process, possibly training the model, represented as a string.

        Examples:
        ---------
        >>> pm = PersistenceManager(
        ...     base_path=os.path.join("..","models"),
        ...     folder_name_model='SVR',
        ...     folder_name_range_train='initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0',
        ...     folder_name_time_execution='execution-time-2023_09_27_13_24_30'
        ... )
        """
        #ToDo: Hacer que puedan ponerse vacío. 
        # Validation of inputs
        # self.validate_string_input(folder_name_model, folder_name_range_train, folder_name_time_execution)
        # Attributes initialization
        self.base_path = base_path
        self.folder_name_model = folder_name_model
        self.folder_name_range_train = folder_name_range_train
        self.folder_name_time_execution = folder_name_time_execution
        
        # Construct the path excluding None values
        path_components = [base_path, folder_name_model, folder_name_range_train, folder_name_time_execution]
        filtered_path_components = [component for component in path_components if component is not None]
        self.path = os.path.join(*filtered_path_components)

    def get_available_models(self):
        """
        Scans the base directory and returns a list of available model names.

        Returns:
        --------
        list
            A list of model names available in the base directory.

        Examples:
        ---------
        >>> pm = PersistenceManager(...)
        >>> pm.get_available_models()
        ['model1', 'model2', ...]
        """
        try:
            return [name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))]
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
        model_path = os.path.join(self.base_path, model_name)
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
        range_path = os.path.join(self.base_path, model_name, training_range)
        try:
            return [name for name in os.listdir(range_path) if os.path.isdir(os.path.join(range_path, name))]
        except FileNotFoundError:
            return []
        
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
            The constructed path string.
            
        Examples:
        ---------
        >>> pm = PersistenceManager('model_name', 'train_range', 'execution_time')
        >>> path = pm.build_path(sub_folder=['preprocessed_data', 'scalers'], file_name='scaler', extension='joblib')
        # path will be: 'model_name/train_range/execution_time/preprocessed_data/scalers/scaler.joblib'
        
        Notes:
        ------
        This method does not check if the constructed path exists or is valid. It merely
        constructs a string based on the provided input and the base path.

        """
        # Start with the base path initialized in the PersistenceManager instance
        path = self.path
        
        # If a sub_folder is provided, append it to the base path
        # If sub_folder is a list of strings, join them to form a nested sub-folder path
        if sub_folder:
            if isinstance(sub_folder, list):
                sub_folder_path = os.path.join(*sub_folder)
            else:
                sub_folder_path = sub_folder
            path = os.path.join(path, sub_folder_path)
        
        # If a file_name is provided, append it to the path
        # If an extension is also provided, append it to the file_name
        file_name = f"{file_name}.{extension}" if extension else file_name
        if file_name:
            path = os.path.join(path, file_name)
        
        # Return the constructed path string
        return path

    
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


    def load_object(self, folder_path, filename, extension):
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
        
        # Combine folder path, filename, and extension to form the full path
        full_path = os.path.join(folder_path, f"{filename}.{extension}")
        
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
            obj = pd.read_csv(full_path, sep = ";")
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
        metadata_file_path = self.build_path(file_name=name)
        
        # Load the metadata object from disk using the `load_object` method
        # This will handle the file loading
        metadata = self.load_object(folder_path = metadata_file_path, extension = extension)

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

    def load_evaluation_data(self, name, folder_name_evaluation_data="evaluation",
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
        folder_name_evaluation_data : str, optional (default: "evaluation")
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

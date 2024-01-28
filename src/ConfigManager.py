import json
from pathlib import Path
from typing import Any, Dict, List, Union

class ConfigManager:
    def __init__(self, config_folder_path: str):
        """
        Initializes the ConfigManager with the path to the folder containing configuration files.

        Parameters:
        config_folder_path (str): The path to the folder containing configuration files. It is expected
        to be str because Path() from pathlib is used

        """
        self.config_folder_path = Path(config_folder_path)

    def load_config(self, config_filename: str, subfolder: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        Load a configuration file from a specified subfolder or subfolder path.

        Parameters:
        config_filename (str): The name of the configuration file (without extension).
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path where the configuration file is located. Defaults to None.

        Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
        
        Raises:
        FileNotFoundError: If the configuration file is not found.
        """
        config_filepath = self._get_config_filepath(config_filename, subfolder)
        if not config_filepath.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_filepath}")
        with open(config_filepath, 'r') as file:
            config = json.load(file)
        return config

    def save_config(self, config_filename: str, config: Dict[str, Any], subfolder: Union[str, List[str]] = None, create: bool = False):
        """
        Save a configuration file to a specified subfolder.

        Parameters:
        config_filename (str): The name of the configuration file (without extension).
        config (Dict[str, Any]): The configuration to save as a dictionary.
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path where the configuration file will be saved. Defaults to None.
        create (bool, optional): A flag to indicate whether to create the subfolder if it does not exist. Defaults to False.

        Raises:
        FileNotFoundError: If the subfolder does not exist and the create flag is set to False.
        """
        config_filepath = self._get_config_filepath(config_filename, subfolder)
        if not config_filepath.parent.is_dir():  # Check if the subfolder exists
            if create:
                config_filepath.parent.mkdir(parents=True)  # Create subfolder and any missing parent folders
            else:
                raise FileNotFoundError(f"Subfolder does not exist: {config_filepath.parent}")

        with open(config_filepath, 'w') as file:
            json.dump(config, file, indent=4)  # Save the configuration to file


    def list_configs(self, subfolder: Union[str, List[str]] = None) -> List[str]:
        """
        List all configuration files in a specified subfolder.

        Parameters:
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path to look for configuration files. Defaults to None.

        Returns:
        List[str]: A list of configuration file names (without extension).
        """
        folder_path = self._get_folder_path(subfolder)
        return [f.stem for f in folder_path.glob("*.json")]

    def update_config(self, config_filename: str, new_values: Dict[str, Any], subfolder: Union[str, List[str]] = None):
        """
        Update specific keys in a configuration file in a specified subfolder, supporting nested updates.

        This method allows updating both top-level and nested configuration settings by passing a dictionary
        with keys representing the paths to the settings. Nested paths should be indicated by dictionaries.

        Parameters:
        - config_filename (str): The name of the configuration file (without extension).
        - new_values (Dict[str, Any]): A dictionary of keys and values to update. For nested updates, the value should be a dictionary.
        - subfolder (Union[str, List[str]], optional): The subfolder or subfolder path where the configuration file is located. Defaults to None.

        Examples:
        - To update a top-level setting: `{"predictor": "RandomForestRegressor"}`
        - To update a nested setting: `{"time_serie_args": {"name_time_column": "time"}}`

        The method will maintain the existing structure of the configuration, only updating the keys provided in `new_values`.
        """
        config = self.load_config(config_filename, subfolder)  # Load the existing config

        def update_dict(d, u):
            """
            Recursively update dictionary `d` with values from `u`, supporting nested dictionaries.
            """
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = self._safe_convert_type(v, type(d.get(k, v)))
            return d

        updated_config = update_dict(config, new_values)
        self.save_config(config_filename, updated_config, subfolder)  # Save the updated config back to file

    def _safe_convert_type(self, value: str, original_type: type) -> Any:
        """
        Safely convert the string value to the specified type if possible, avoiding the use of eval.

        Parameters:
        - value (str): The string value to convert.
        - original_type (type): The type to convert the value to.

        Returns:
        - Any: The converted value.
        """
        if value.lower() in ('none', 'null', ''):
            return None
        try:
            if original_type == bool:
                return value.lower() in ('true', '1', 't', 'y', 'yes')
            elif original_type == list:
                return json.loads(value.replace("'", '"'))  # Converts string to list, ensuring double quotes for JSON compatibility
            elif original_type in [int, float]:
                return original_type(value)
            else:  # fallback to string
                return value
        except (ValueError, json.JSONDecodeError):
            # Return the original value if conversion fails
            return value

    def update_all_configs(self, new_values: Dict[str, Any], subfolder: Union[str, List[str]] = None):
        """
        Update specific keys in all configuration files in a specified subfolder.

        Parameters:
        new_values (Dict[str, Any]): A dictionary of keys and values to update.
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path to look for configuration files. Defaults to None.
        """
        for config_filename in self.list_configs(subfolder):  # Iterate over all config files in the specified subfolder
            self.update_config(config_filename, new_values, subfolder)  # Update each config file with the new values

    def _get_config_filepath(self, config_filename: str, subfolder: Union[str, List[str]] = None) -> Path:
        """
        Get the full path to a configuration file, optionally in a specified subfolder or subfolder path.

        Parameters:
        config_filename (str): The name of the configuration file (without extension).
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path where the configuration file is located. Defaults to None.

        Returns:
        Path: The full path to the configuration file.
        """
        folder_path = self._get_folder_path(subfolder)
        return folder_path / f"{config_filename}.json"
    
    def _get_folder_path(self, subfolder: Union[str, List[str]] = None) -> Path:
        """
        Get the folder path, optionally to a specified subfolder or subfolder path.
        Raises an error if the subfolder does not exist.

        Parameters:
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path to look for configuration files. Defaults to None.

        Returns:
        Path: The path to the folder or subfolder.

        Raises:
        FileNotFoundError: If the subfolder does not exist.
        """
        folder_path = self.config_folder_path.joinpath(Path(subfolder)) if subfolder else self.config_folder_path
        if not folder_path.is_dir():  # Verify if the subfolder exists
            raise FileNotFoundError(f"Subfolder does not exist: {folder_path}")
        return folder_path

# # Usage:
# config_manager = ConfigManager("../config")

# # Load a specific config from a subfolder
# config = config_manager.load_config("KNeighborsRegressor", subfolder="models_parameters")

# # List all configs in a subfolder
# all_configs = config_manager.list_configs(subfolder="models_parameters")

# # Update a specific config in a subfolder
# config_manager.update_config("KNeighborsRegressor", {"new_key": "new_value"}, subfolder="models_parameters")

# # Update all configs in a subfolder
# config_manager.update_all_configs({"global_key": "global_value"}, subfolder="models_parameters")

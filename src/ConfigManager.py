import json
from pathlib import Path
from typing import Any, Dict, List, Union

class ConfigManager:
    def __init__(self, config_folder_path: str):
        """
        Initializes the ConfigManager with the path to the folder containing configuration files.

        Parameters:
        config_folder_path (str): The path to the folder containing configuration files.

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
        Update specific keys in a configuration file in a specified subfolder.

        Parameters:
        config_filename (str): The name of the configuration file (without extension).
        new_values (Dict[str, Any]): A dictionary of keys and values to update.
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path where the configuration file is located. Defaults to None.
        """
        config = self.load_config(config_filename, subfolder)  # Load the existing config
        config.update(new_values)  # Update the config with new values
        self.save_config(config_filename, config, subfolder)  # Save the updated config back to file

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

        Parameters:
        subfolder (Union[str, List[str]], optional): The subfolder or subfolder path to look for configuration files. Defaults to None.

        Returns:
        Path: The path to the folder or subfolder.
        """
        if subfolder:
            return self.config_folder_path.joinpath(Path(subfolder))
        return self.config_folder_path

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

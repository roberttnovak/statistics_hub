import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pathlib import Path
import pytest
from own_utils import execute_concurrently, extract_nested_dict_values, get_deepest_keys_values, list_directories_by_depth

# Funciones de ejemplo para pruebas
def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def power(base, exponent):
    return base ** exponent

def division(x, y):
    return x / y

def print_message(message):
    print(message)
    
def trapezoid_area(b1, b2, h):
    return ((b1 + b2) * h) / 2

class TestExecuteConcurrently:
    @pytest.mark.parametrize(
        "func,args_list,expected_result",
        [
            (add, [{'x': 5, 'y': 3}, {'x': 10, 'y': 7},{'x': 5, 'y': 5}], [8, 17, 10]),
            (trapezoid_area, [{'b1': 5, 'b2': 3, 'h': 4}, {'b1': 10, 'b2': 7, 'h': 5}, {'b1': 6, 'b2': 4, 'h': 7}], [16.0, 42.5, 35.0])
        ]
    )
    def test_valid_input(self, func, args_list, expected_result):
        """Test valid inputs."""
        results = execute_concurrently(func, args_list)
        assert sorted(results) == sorted(expected_result)

    @pytest.mark.parametrize(
        "func,args_list,successful_results,exception_type",
        [
            (division, [{"x": 4, "y": 2},{"x": 10, "y": 0},{"x": 10, "y": 2}], [2, 5], ZeroDivisionError)
        ]
    )
    def test_exception(self, func, args_list, successful_results, exception_type):
        """Test exceptions."""
        results = execute_concurrently(func, args_list)
        # Dividir los resultados en resultados exitosos y excepciones
        successful_results_out = [result for result in results if not isinstance(result, Exception)]
        exceptions = [result for result in results if isinstance(result, Exception)]
        assert sorted(successful_results_out) == sorted(successful_results)  # Comprobar los resultados exitosos
        assert len(exceptions) == 1  # Comprobar que hay una excepción
        assert isinstance(exceptions[0], exception_type)  # Comprobar que la excepción es del tipo correcto

    @pytest.mark.parametrize(
        "func,args_list,expected_result",
        [
            (print_message, [{"message": "Hello, World!"}], [None])
        ]
    )
    def test_none_return(self, func, args_list, expected_result):
        """Test functions with no return value."""
        results = execute_concurrently(func, args_list)
        assert results == expected_result
        
class TestListDirectoriesByDepth:
    @pytest.fixture
    def setup_directory_structure(self, tmp_path):
        """
        Set up a test directory structure at a temporary location.

        This function creates a directory structure for testing. It utilizes pytest's tmp_path fixture
        to create temporary directories. The structure has 3 "subdir" directories,
        each containing 2 "subsubdir" directories, which in turn each contain 2 "subsubsubdir" directories.
        ├── subdir1
        │   ├── subsubdir1
        │   │   ├── subsubsubdir1
        │   │   └── subsubsubdir2
        │   └── subsubdir2
        │       ├── subsubsubdir1
        │       └── subsubsubdir2
        ├── subdir2
        │   ├── subsubdir1
        │   │   ├── subsubsubdir1
        │   │   └── subsubsubdir2
        │   └── subsubdir2
        │       ├── subsubsubdir1
        │       └── subsubsubdir2
        └── subdir3
            ├── subsubdir1
            │   ├── subsubsubdir1
            │   └── subsubsubdir2
            └── subsubdir2
                ├── subsubsubdir1
                └── subsubsubdir2
        

        Parameters:
            self: The test class instance.
            tmp_path: Temporary directory path provided by pytest.

        Returns:
            tmp_path: The path to the root of the temporary directory.
        """
        for i in range(1, 4):
            subdir = tmp_path / f"subdir{i}"
            subdir.mkdir()
            for j in range(1, 3):
                subsubdir = subdir / f"subsubdir{j}"
                subsubdir.mkdir()
                for k in range(1, 3):
                    subsubsubdir = subsubdir / f"subsubsubdir{k}"
                    subsubsubdir.mkdir()

        return tmp_path

    @pytest.mark.parametrize(
        "max_depth,list_only_last_level,expected_result",
        [
            # Pruebas para max_depth=2
            (2, True, [
                Path('subdir1/subsubdir1'),
                Path('subdir1/subsubdir2'),
                Path('subdir2/subsubdir1'),
                Path('subdir2/subsubdir2'),
                Path('subdir3/subsubdir1'),
                Path('subdir3/subsubdir2')
            ]),
            (2, False, [
                Path(''),
                Path('subdir1'),
                Path('subdir1/subsubdir1'),
                Path('subdir1/subsubdir2'),
                Path('subdir2'),
                Path('subdir2/subsubdir1'),
                Path('subdir2/subsubdir2'),
                Path('subdir3'),
                Path('subdir3/subsubdir1'),
                Path('subdir3/subsubdir2')
            ]),
            # Pruebas para max_depth=3
            (3, True, [
                Path('subdir1/subsubdir1/subsubsubdir1'),
                Path('subdir1/subsubdir1/subsubsubdir2'),
                Path('subdir1/subsubdir2/subsubsubdir1'),
                Path('subdir1/subsubdir2/subsubsubdir2'),
                Path('subdir2/subsubdir1/subsubsubdir1'),
                Path('subdir2/subsubdir1/subsubsubdir2'),
                Path('subdir2/subsubdir2/subsubsubdir1'),
                Path('subdir2/subsubdir2/subsubsubdir2'),
                Path('subdir3/subsubdir1/subsubsubdir1'),
                Path('subdir3/subsubdir1/subsubsubdir2'),
                Path('subdir3/subsubdir2/subsubsubdir1'),
                Path('subdir3/subsubdir2/subsubsubdir2')
            ]),
            (3, False, [
                Path(''),
                Path('subdir1'),
                Path('subdir1/subsubdir1'),
                Path('subdir1/subsubdir1/subsubsubdir1'),
                Path('subdir1/subsubdir1/subsubsubdir2'),
                Path('subdir1/subsubdir2'),
                Path('subdir1/subsubdir2/subsubsubdir1'),
                Path('subdir1/subsubdir2/subsubsubdir2'),
                Path('subdir2'),
                Path('subdir2/subsubdir1'),
                Path('subdir2/subsubdir1/subsubsubdir1'),
                Path('subdir2/subsubdir1/subsubsubdir2'),
                Path('subdir2/subsubdir2'),
                Path('subdir2/subsubdir2/subsubsubdir1'),
                Path('subdir2/subsubdir2/subsubsubdir2'),
                Path('subdir3'),
                Path('subdir3/subsubdir1'),
                Path('subdir3/subsubdir1/subsubsubdir1'),
                Path('subdir3/subsubdir1/subsubsubdir2'),
                Path('subdir3/subsubdir2'),
                Path('subdir3/subsubdir2/subsubsubdir1'),
                Path('subdir3/subsubdir2/subsubsubdir2')
            ])
        ]
    )
    def test_list_directories_by_depth(
        self, setup_directory_structure, max_depth, list_only_last_level, expected_result
    ):
        """Test listing directories by depth."""
        path = setup_directory_structure
        result = list_directories_by_depth(path, max_depth, list_only_last_level)
        
        # Convertir las rutas resultantes a objetos Path, luego a rutas relativas (como strings).
        result = [Path(p).relative_to(path) for p in result]
        
        # Convertir las rutas a strings usando as_posix() para la comparación.
        result = [p.as_posix() for p in result]
        expected_result = [p.as_posix() for p in expected_result]
        
        print(f"Result: {result}")  # Agregar esta línea para imprimir el resultado

        assert sorted(result) == sorted(expected_result), f"Expected {sorted(expected_result)}, but got {sorted(result)}"

class TestGetDeepestKeysValues:
    @pytest.mark.parametrize(
        "input_dict, expected_output",
        [
            # Simple test case
            ({"a": {"b": {"c": 1}}}, {"c": 1}),
            # Cases with several nesting levels
            (
                {"level1": {"level2": {"level3a": "value1", "level3b": "value2"}, "level2b": "value3"}}, 
                {"level3a": "value1", "level3b": "value2", "level2b": "value3"}
            ),
            (
                {
                    'level1a': {
                        'level2a': {
                            'level3a': 'value3a',
                            'level3b': 'value3b'
                        },
                        'level2b': 'value2b'
                    },
                    'level1b' : 'value1b'
                },
                {'level1b': 'value1b', 'level3a': 'value3a', 'level3b': 'value3b', 'level2b': 'value2b'}
            ),
            # Case with mix of levels and types
            ({"x": 1, "y": {"a": 2, "b": 3}, "z": {"p": {"q": 4}}}, {"x":1, "a": 2, "b": 3, "q": 4}),
            # Case with empty dictionary
            ({}, {}),
            # Case with levels but no values in the deepest level
            ({"a": {"b": {}}}, {}),
            # Real use case: Get args classiffied by some group
            (
                {
                    'data_importer_args' :
                        {
                            'data_importer_automatic_importation': False, 
                            'data_importer_database': None, 
                            'data_importer_query': None, 
                            'data_importer_save_importation': True, 
                            'data_importer_file_name': None
                        }, 
                    'prepare_dataframe_from_db_cols_for_query': ['00-eco2', '00-temp', '01-hum', '01-tvoc', '02-pres', '03-siaq', '04-diaq'], 
                    'preprocess_time_series_data_args':
                        {
                            'preprocess_time_series_data_resample_freq': '60S', 
                            'preprocess_time_series_data_aggregation_func': 'mean', 
                            'preprocess_time_series_data_method': 'linear', 
                            'preprocess_time_series_data_outlier_cols': None
                        },
                    'split_train_test_args': {
                        'ini_train': '2023-04-18 00:00:00+00:00',
                        'fin_train': '2023-04-25 00:00:00+00:00',
                        'fin_test': '2023-04-26 00:00:00+00:00'
                    },
                    'time_series_args': {
                        'name_time_column': 'timestamp',
                        'name_id_sensor_column': 'id_device',
                        'id_device': 'DBEM003', 
                        'names_objective_variable': 'y', 
                        'X_name_features': None,
                        'Y_name_features': 'y',
                        'n_lags': 10,
                        'n_predictions': 20,
                        'lag_columns': None,
                        'lead_columns': 'y',
                        'num_obs_to_predict': None
                    }, 
                    'predictor': 'KNeighborsRegressor',
                    'save_args':
                    {
                        'scale_in_preprocessing': True, 
                        'save_preprocessing': True, 
                        'folder_name_model': None, 
                        'folder_name_time_execution': None 
                    }
                },
                {
                    'data_importer_automatic_importation': False, 
                    'data_importer_database': None, 
                    'data_importer_query': None, 
                    'data_importer_save_importation': True, 
                    'data_importer_file_name': None, 
                    'prepare_dataframe_from_db_cols_for_query': ['00-eco2', '00-temp', '01-hum', '01-tvoc', '02-pres', '03-siaq', '04-diaq'], 
                    'preprocess_time_series_data_resample_freq': '60S', 
                    'preprocess_time_series_data_aggregation_func': 'mean', 
                    'preprocess_time_series_data_method': 'linear', 
                    'preprocess_time_series_data_outlier_cols': None,
                    'ini_train': '2023-04-18 00:00:00+00:00',
                    'fin_train': '2023-04-25 00:00:00+00:00',
                    'fin_test': '2023-04-26 00:00:00+00:00',
                    'name_time_column': 'timestamp',
                    'name_id_sensor_column': 'id_device',
                    'id_device': 'DBEM003', 
                    'names_objective_variable': 'y', 
                    'X_name_features': None,
                    'Y_name_features': 'y',
                    'n_lags': 10,
                    'n_predictions': 20,
                    'lag_columns': None,
                    'lead_columns': 'y',
                    'num_obs_to_predict': None, 
                    'predictor': 'KNeighborsRegressor',
                    'scale_in_preprocessing': True, 
                    'save_preprocessing': True, 
                    'folder_name_model': None, 
                    'folder_name_time_execution': None 
                }
            )
        ]
    )
    def test_get_deepest_keys_values(self, input_dict, expected_output):
        """Test that get_deepest_keys_values returns the correct deepest keys and values."""
        assert get_deepest_keys_values(input_dict) == expected_output

class TestExtractNestedDictValues:
    @pytest.mark.parametrize(
        "nested_dict, path_mapping, expected_output",
        [
            # Docstring example
            (
                {
                    'doc': {
                        'params': {
                            'rx_time': 1705479916.622762,
                            'radio': {
                                'freq': 868.3,
                                'modulation': {'type': 'LORA'}
                            }
                        },
                        'meta': {'network': 'ad4bc0f8d0cb49928b65d7d7219ef52c'}
                    }
                },
                {
                    "time": ("doc", "params", "rx_time"),
                    "frequency": ("doc", "params", "radio", "freq"),
                    "modulation_type": ("doc", "params", "radio", "modulation", "type"),
                    "network_id": ("doc", "meta", "network"),
                    "non_existing_field": ("doc", "params", "non_existing")
                },
                {
                    'time': 1705479916.622762,
                    'frequency': 868.3,
                    'modulation_type': 'LORA',
                    'network_id': 'ad4bc0f8d0cb49928b65d7d7219ef52c',
                    'non_existing_field': None
                }
            ),
            # Example with deeper levels
            (
                {
                    'level1': {
                        'level2': {
                            'level3': {
                                'key': 'value',
                                'list': [1, 2, 3]
                            }
                        }
                    }
                },
                {
                    'deep_value': ('level1', 'level2', 'level3', 'key'),
                    'deep_list': ('level1', 'level2', 'level3', 'list'),
                    'nonexistent': ('level1', 'level2', 'nonexistent')
                },
                {
                    'deep_value': 'value',
                    'deep_list': [1, 2, 3],
                    'nonexistent': None
                }
            ),
            # Different type of data
            (
                {
                    'data': {
                        'integer': 123,
                        'float': 456.789,
                        'string': 'hello',
                        'boolean': True,
                        'dict': {'nested_key': 'nested_value'}
                    }
                },
                {
                    'int_val': ('data', 'integer'),
                    'float_val': ('data', 'float'),
                    'str_val': ('data', 'string'),
                    'bool_val': ('data', 'boolean'),
                    'nested_val': ('data', 'dict', 'nested_key'),
                    'missing_val': ('data', 'missing')
                },
                {
                    'int_val': 123,
                    'float_val': 456.789,
                    'str_val': 'hello',
                    'bool_val': True,
                    'nested_val': 'nested_value',
                    'missing_val': None
                }
            )
            # ...
        ]
    )
    def test_extract_nested_dict_values(self, nested_dict, path_mapping, expected_output):
        """Test that extract_nested_dict_values correctly extracts values from nested dictionaries."""
        assert extract_nested_dict_values(nested_dict, path_mapping) == expected_output

class TestCreateLogger:
    #ToDo :  Hacer los tests para create_logger
    pass

if __name__ == "__main__":
    pytest.main()

import os
from pathlib import Path
import pytest
from src.own_utils import execute_concurrently, list_directories_by_depth

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


class TestCreateLogger:
    #ToDo :  Hacer los tests para create_logger
    pass

if __name__ == "__main__":
    pytest.main()

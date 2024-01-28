import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import statistics
import pytest
import datetime
import pandas as pd 
from src.predictions import (format_datetime_string, name_folder_train_range, get_train_test, 
                             create_lag_lead_features)

class TestFormatDateTimeString:
    @pytest.mark.parametrize(
        "datetime_str,expected_result",
        [
            ('2023-04-18 00:00:00+00:00', '2023_4_18_0_0_0-UTC0'),
            ('2023-04-18 00:00:00', '2023_4_18_0_0_0'),
        ]
    )
    def test_valid_input(self, datetime_str, expected_result):
        """Test valid datetime string input."""
        assert format_datetime_string(datetime_str) == expected_result

    def test_invalid_input(self):
        """Test invalid datetime string input."""
        with pytest.raises(ValueError):
            format_datetime_string('invalid_datetime_str')

class TestNameFolderTrainRange:
    @pytest.mark.parametrize(
        "ini_train_str, fin_train_str, expected",
        [
            ('2023-04-18 00:00:00+00:00', '2023-11-18 00:00:00+00:00', 'initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_11_18_0_0_0-UTC0'),
            ('2023-04-18 00:00:00', '2023-11-18 00:00:00', 'initrain-2023_4_18_0_0_0___fintrain-2023_11_18_0_0_0'),
            ('2023-04-18 00:00:00', '2023-11-18 00:00:00+00:00', 'initrain-2023_4_18_0_0_0___fintrain-2023_11_18_0_0_0-UTC0'),
            ('2023-04-18 00:00:00+00:00', '2023-11-18 00:00:00', 'initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_11_18_0_0_0'),
        ]
    )
    def test_valid_input(self, ini_train_str, fin_train_str, expected):
        """Test valid training datetime string input."""
        assert name_folder_train_range(ini_train_str, fin_train_str) == expected

class TestCombination:
    @pytest.mark.parametrize(
        "ini_train_str, fin_train_str",
        [
            ('2023-04-18 00:00:00+00:00', '2023-11-18 00:00:00+00:00'),
            ('2023-04-18 00:00:00', '2023-11-18 00:00:00'),
            ('2023-04-18 00:00:00', '2023-11-18 00:00:00+00:00'),
            ('2023-04-18 00:00:00+00:00', '2023-11-18 00:00:00'),
        ]
    )
    def test_combine_format_datetime_string_name_folder_train_range(self, ini_train_str, fin_train_str):
        """Test combination of format_datetime_string and name_folder_train_range."""
        folder_name = name_folder_train_range(ini_train_str, fin_train_str)
        formatted_ini_train = format_datetime_string(ini_train_str)
        formatted_fin_train = format_datetime_string(fin_train_str)
        expected = f"initrain-{formatted_ini_train}___fintrain-{formatted_fin_train}"
        assert folder_name == expected
 
class TestGetTrainTest:
    """Test suite for the get_train_test function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame to use in the tests."""
        data = {
            "timestamp": pd.date_range(start="2020-01-01", periods=10),
            "value": range(10)
        }
        return pd.DataFrame(data)

    @pytest.mark.parametrize(
        "ini_train, fin_train, fin_test, expected_train, expected_test",
        [
            (
                "2020-01-01", "2020-01-05", "2020-01-10",
                pd.DataFrame({
                    "timestamp": pd.date_range(start="2020-01-01", periods=5),
                    "value": range(5)
                }),
                pd.DataFrame({
                    "timestamp": pd.date_range(start="2020-01-06", periods=5),
                    "value": range(5, 10)
                })
            ),  # Normal case
            # ... (otros casos normales si los tienes)
        ]
    )
    def test_get_train_test(
        self,
        sample_dataframe,
        ini_train,
        fin_train,
        fin_test,
        expected_train,
        expected_test
    ):
        """Parametrized tests for get_train_test."""
        result = get_train_test(sample_dataframe, ini_train, fin_train, fin_test)
        pd.testing.assert_frame_equal(result['df_train'], expected_train, check_dtype=False)
        pd.testing.assert_frame_equal(result['df_test'], expected_test, check_dtype=False)

    @pytest.mark.parametrize(
        "ini_train, fin_train, fin_test",
        [
            ("2020-01-01", "2020-01-01", "2020-01-10"),  # Training with a single day
            ("2020-01-01", "2020-01-02", "2020-01-02")   # Empty test
        ]
    )
    def test_invalid_date_range(self, sample_dataframe, ini_train, fin_train, fin_test):
        """Error handling test for invalid date ranges."""
        with pytest.raises(ValueError, match="Date range is invalid: Ensure ini_train < fin_train < fin_test"):
            get_train_test(sample_dataframe, ini_train, fin_train, fin_test)




class TestCreateLagLeadFeatures:
    """Test suite for the create_lag_lead_features function."""

    @pytest.fixture(scope="class")
    def sample_dataframe(self):
        """Sample DataFrame to use in the tests.

        Returns:
            pd.DataFrame: A sample DataFrame.
        """
        return pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})

    @pytest.mark.parametrize(
        "n_lags, n_leads, lag_columns, lead_columns, fillna_method, expected_df",
        [
            # Caso original: 1 lag para 'A', 1 lead para 'B', llenar con la media
            (
                1, 1, ['A'], ['B'], 'mean',
                pd.DataFrame({
                    'A': [1, 2, 3, 4, 5],
                    'B': [5, 4, 3, 2, 1],
                    'lag_A_1': [statistics.mean([1,2,3,4]), 1, 2, 3, 4],
                    'lead_B_1': [4, 3, 2, 1, statistics.mean([1,2,3,4])]
                })
            ),
            # Caso donde no se generan lags ni leads
            (
                0, 0, None, None, 'mean',
                pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
            ),
            # 2 lags para 'A' y 'B', sin leads, llenar con cero
            (
                2, 0, ['A', 'B'], None, 'zero',
                pd.DataFrame({
                    'A': [1, 2, 3, 4, 5],
                    'B': [5, 4, 3, 2, 1],
                    'lag_A_1': [0, 1, 2, 3, 4],
                    'lag_A_2': [0, 0, 1, 2, 3],
                    'lag_B_1': [0, 5, 4, 3, 2],
                    'lag_B_2': [0, 0, 5, 4, 3]
                })
            ),
            # 2 leads para 'A', sin lags, llenar con la mediana
            (
                0, 2, None, ['A'], 'median',
                pd.DataFrame({
                    'A': [1, 2, 3, 4, 5],
                    'B': [5, 4, 3, 2, 1],
                    'lead_A_1': [2, 3, 4, 5, statistics.median([2, 3, 4, 5])],
                    'lead_A_2': [3, 4, 5, statistics.median([3, 4, 5]), statistics.median([3, 4, 5])]
                })
            ),

        # 1 lag y 1 lead para ambas columnas, llenar con la media
        (
            1, 1, ['A', 'B'], ['A', 'B'], 'mean',
            pd.DataFrame({
                'A': [1, 2, 3, 4, 5],
                'B': [5, 4, 3, 2, 1],
                'lag_A_1': [statistics.mean([1, 2, 3, 4]), 1, 2, 3, 4],
                'lag_B_1': [statistics.mean([5, 4, 3, 2]), 5, 4, 3, 2],
                'lead_A_1': [2, 3, 4, 5, statistics.mean([2, 3, 4, 5])],
                'lead_B_1': [4, 3, 2, 1, statistics.mean([4, 3, 2, 1])]
            })
        ),
        
        # Añade más casos de prueba según sea necesario
        ]
    )
    # ...

    def test_create_lag_lead_features(
        self,
        sample_dataframe,
        n_lags,
        n_leads,
        lag_columns,
        lead_columns,
        fillna_method,
        expected_df
    ):
        """Parametrized tests for create_lag_lead_features."""
        result_df = create_lag_lead_features(
            sample_dataframe, 
            n_lags=n_lags, 
            n_leads=n_leads, 
            lag_columns=lag_columns, 
            lead_columns=lead_columns, 
            fillna_method=fillna_method
        )
        pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

    @pytest.mark.parametrize(
        "n_lags, n_leads, lag_columns, lead_columns, fillna_method",
        [
            (1, 1, ['A'], ['B'], 'unsupported_method'),  # Método fillna no soportado
            (-1, 1, ['A'], ['B'], 'mean'),  # n_lags negativo
            (1, -1, ['A'], ['B'], 'mean'),  # n_leads negativo
            (1, 1, ['Z'], ['B'], 'mean'),  # Columna inexistente para lag
            (1, 1, ['A'], ['Z'], 'mean'),  # Columna inexistente para lead
            # Añade más casos de prueba para entradas no válidas
        ]
    )
    def test_invalid_inputs(
        self,
        sample_dataframe,
        n_lags,
        n_leads,
        lag_columns,
        lead_columns,
        fillna_method
    ):
        """Error handling test for invalid inputs."""
        with pytest.raises(ValueError):
            create_lag_lead_features(
                sample_dataframe, 
                n_lags=n_lags, 
                n_leads=n_leads, 
                lag_columns=lag_columns, 
                lead_columns=lead_columns, 
                fillna_method=fillna_method
            )
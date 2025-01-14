"""
Hyperparameter Domains for Scikit-learn Regressors

This script provides a comprehensive dictionary containing the domains or 
possible values of hyperparameters for various scikit-learn regressors. The 
domains have been generated using ChatGPT, a Large Language Model (LLM), due 
to the inherent difficulty of extracting this information directly from the 
scikit-learn documentation.

The main objective of this script is to leverage the broad knowledge of an LLM 
to automate the process of obtaining the range of values for each hyperparameter. 
These domains are particularly useful for hyperparameter tuning during Cross 
Validation (CV) or other model selection processes. By having structured 
domains, it becomes easier to design automated tuning strategies that cover 
relevant search spaces efficiently, without relying on manual parameter tuning.

### Structure of the JSON

The generated JSON is a nested dictionary where each top-level key represents a 
regressor from scikit-learn, and each regressor has a set of hyperparameters. 
Each hyperparameter is represented as a dictionary containing the following keys:

- **"domain"**: The possible values or range for the hyperparameter.
  - For intervals, the values are specified as a two-element list.
  - For categorical values, the values are specified as a list of possible options.

- **"type"**: The type of domain provided for the hyperparameter.
  - **"interval"**: The domain is a continuous range between two values. 
    - For example: `"domain": [0, "inf"]` means the value can be any number 
      between 0 and infinity.
  - **"categorical"**: The domain is a discrete set of possible values.
    - For example: `"domain": [True, False]` means the hyperparameter can take 
      only the values `True` or `False`.

### Special Considerations

- **Infinity Values**: The script uses strings to represent infinity due to 
  JSON limitations. The following strings are used:
  - **"inf"**: Positive infinity.
  - **"-inf"**: Negative infinity.
  - Example: `"domain": [0, "inf"]` represents a range from 0 to infinity.

- **Random State and Seeds**: Hyperparameters related to random states or seeds 
  are usually specified with an interval starting from 0 to positive infinity:
  - Example: `"domain": [0, "inf"]`.

- **Null Values**: When a hyperparameter can accept a `None` value, it is 
  represented as `null` in the JSON, following the JSON standard.

Generated on: 2024-02-15
Version: Scikit-learn 1.3.2
"""

# Hyperparameter domains for regressors in scikit-learn
from ConfigManager import ConfigManager


all_regressors_with_its_parameters_and_domains = {
    "ARDRegression": {
        "alpha_1": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "alpha_2": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "compute_score": {
            "domain": [True, False],
            "type": "categorical"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "lambda_1": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "lambda_2": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "threshold_lambda": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "tol": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "AdaBoostRegressor": {
        "base_estimator": {
            "domain": [None],
            "type": "categorical"
        },
        "estimator": {
            "domain": [None],
            "type": "categorical"
        },
        "learning_rate": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "loss": {
            "domain": ["linear", "square", "exponential"],
            "type": "categorical"
        },
        "n_estimators": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "BaggingRegressor": {
        "base_estimator": {
            "domain": [None],
            "type": "categorical"
        },
        "bootstrap": {
            "domain": [True, False],
            "type": "categorical"
        },
        "bootstrap_features": {
            "domain": [True, False],
            "type": "categorical"
        },
        "estimator": {
            "domain": [None],
            "type": "categorical"
        },
        "max_features": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "max_samples": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "n_estimators": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1],
            "type": "categorical"
        },
        "oob_score": {
            "domain": [True, False],
            "type": "categorical"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "BayesianRidge": {
        "alpha_1": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "alpha_2": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "alpha_init": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "compute_score": {
            "domain": [True, False],
            "type": "categorical"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "lambda_1": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "lambda_2": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "lambda_init": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "tol": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "CCA": {
        "copy": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_components": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "scale": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "DecisionTreeRegressor": {
        "ccp_alpha": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "criterion": {
            "domain": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "type": "categorical"
        },
        "max_depth": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_features": {
            "domain": ["auto", "sqrt", "log2", None],
            "type": "categorical"
        },
        "max_leaf_nodes": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_impurity_decrease": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "min_samples_leaf": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "min_samples_split": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_weight_fraction_leaf": {
            "domain": [0.0, 0.5],
            "type": "interval"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "splitter": {
            "domain": ["best", "random"],
            "type": "categorical"
        }
    },
    "DummyRegressor": {
        "constant": {
            "domain": [None, "any_numeric_value"],
            "type": "categorical_or_numeric"
        },
        "quantile": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "strategy": {
            "domain": ["mean", "median", "constant"],
            "type": "categorical"
        }
    },
    "ElasticNet": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "l1_ratio": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "ElasticNetCV": {
        "alphas": {
            "domain": [None, "array_like_of_floats"],
            "type": "categorical_or_array"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross_validation_generator"],
            "type": "categorical"
        },
        "eps": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "l1_ratio": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_alphas": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, "int"],
            "type": "categorical_or_interval"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "ExtraTreeRegressor": {
        "ccp_alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "criterion": {
            "domain": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "type": "categorical"
        },
        "max_depth": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_features": {
            "domain": ["auto", "sqrt", "log2", None],
            "type": "categorical"
        },
        "max_leaf_nodes": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_impurity_decrease": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "min_samples_leaf": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "min_samples_split": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_weight_fraction_leaf": {
            "domain": [0.0, 0.5],
            "type": "interval"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "splitter": {
            "domain": ["best", "random"],
            "type": "categorical"
        }
    },
    "ExtraTreesRegressor": {
        "bootstrap": {
            "domain": [True, False],
            "type": "categorical"
        },
        "ccp_alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "criterion": {
            "domain": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "type": "categorical"
        },
        "max_depth": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_features": {
            "domain": ["auto", "sqrt", "log2", None],
            "type": "categorical"
        },
        "max_leaf_nodes": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "max_samples": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "min_impurity_decrease": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "min_samples_leaf": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "min_samples_split": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_weight_fraction_leaf": {
            "domain": [0.0, 0.5],
            "type": "interval"
        },
        "n_estimators": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, "int"],
            "type": "categorical_or_interval"
        },
        "oob_score": {
            "domain": [True, False],
            "type": "categorical"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "GammaRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "solver": {
            "domain": ["lbfgs", "newton-cholesky"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "GaussianProcessRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X_train": {
            "domain": [True, False],
            "type": "categorical"
        },
        "kernel": {
            "domain": [None, "callable", "RBF", "Matern", "DotProduct"],
            "type": "categorical_or_callable"
        },
        "n_restarts_optimizer": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "n_targets": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "normalize_y": {
            "domain": [True, False],
            "type": "categorical"
        },
        "optimizer": {
            "domain": ["fmin_l_bfgs_b", "callable", None],
            "type": "categorical_or_callable"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "GradientBoostingRegressor": {
        "alpha": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "ccp_alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "criterion": {
            "domain": ["friedman_mse", "squared_error"],
            "type": "categorical"
        },
        "init": {
            "domain": [None, "callable", "zero"],
            "type": "categorical_or_callable"
        },
        "learning_rate": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "loss": {
            "domain": ["squared_error", "absolute_error", "huber", "quantile"],
            "type": "categorical"
        },
        "max_depth": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_features": {
            "domain": ["auto", "sqrt", "log2", None],
            "type": "categorical"
        },
        "max_leaf_nodes": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "min_impurity_decrease": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "min_samples_leaf": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "min_samples_split": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_weight_fraction_leaf": {
            "domain": [0.0, 0.5],
            "type": "interval"
        },
        "n_estimators": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_iter_no_change": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "subsample": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "validation_fraction": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "HistGradientBoostingRegressor": {
        "categorical_features": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "early_stopping": {
            "domain": [True, False],
            "type": "categorical"
        },
        "interaction_cst": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "l2_regularization": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "learning_rate": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "loss": {
            "domain": ["squared_error", "absolute_error", "poisson"],
            "type": "categorical"
        },
        "max_bins": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "max_depth": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_leaf_nodes": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "min_samples_leaf": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "monotonic_cst": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "n_iter_no_change": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "quantile": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "random_state": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "scoring": {
            "domain": [None, "callable"],
            "type": "categorical_or_callable"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "validation_fraction": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "HuberRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "epsilon": {
            "domain": [1.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "IsotonicRegression": {
        "increasing": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "out_of_bounds": {
            "domain": ["nan", "clip", "raise"],
            "type": "categorical"
        },
        "y_max": {
            "domain": [None, "-inf", "inf"],
            "type": "categorical_or_interval"
        },
        "y_min": {
            "domain": [None, "-inf", "inf"],
            "type": "categorical_or_interval"
        }
    },
    "KNeighborsRegressor": {
        "algorithm": {
            "domain": ["auto", "ball_tree", "kd_tree", "brute"],
            "type": "categorical"
        },
        "leaf_size": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "metric": {
            "domain": ["minkowski", "euclidean", "manhattan", "chebyshev"],
            "type": "categorical"
        },
        "metric_params": {
            "domain": [None, "dict"],
            "type": "categorical_or_dict"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "n_neighbors": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "p": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "weights": {
            "domain": ["uniform", "distance", "callable"],
            "type": "categorical_or_callable"
        }
    },
    "KernelRidge": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "coef0": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "degree": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "gamma": {
            "domain": [None, 0.0, "inf"],
            "type": "categorical_or_interval"
        },
        "kernel": {
            "domain": ["linear", "poly", "rbf", "sigmoid", "precomputed", "callable"],
            "type": "categorical_or_callable"
        },
        "kernel_params": {
            "domain": [None, "dict"],
            "type": "categorical_or_dict"
        }
    },
    "Lars": {
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_path": {
            "domain": [True, False],
            "type": "categorical"
        },
        "jitter": {
            "domain": [None, 0.0, "inf"],
            "type": "categorical_or_interval"
        },
        "n_nonzero_coefs": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "normalize": {
            "domain": [True, False, "deprecated"],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "LarsCV": {
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_n_alphas": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "normalize": {
            "domain": [True, False, "deprecated"],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "Lasso": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "LassoCV": {
        "alphas": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_alphas": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "LassoLars": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_path": {
            "domain": [True, False],
            "type": "categorical"
        },
        "jitter": {
            "domain": [None, 0.0, "inf"],
            "type": "categorical_or_interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "normalize": {
            "domain": [True, False, "deprecated"],
            "type": "categorical"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "LassoLarsCV": {
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_n_alphas": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "normalize": {
            "domain": [True, False, "deprecated"],
            "type": "categorical"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "LassoLarsIC": {
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "criterion": {
            "domain": ["aic", "bic"],
            "type": "categorical"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "noise_variance": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "normalize": {
            "domain": [True, False, "deprecated"],
            "type": "categorical"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "LinearRegression": {
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "LinearSVR": {
        "C": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "dual": {
            "domain": [True, False],
            "type": "categorical"
        },
        "epsilon": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "intercept_scaling": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "loss": {
            "domain": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "MLPRegressor": {
        "activation": {
            "domain": ["identity", "logistic", "tanh", "relu"],
            "type": "categorical"
        },
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "batch_size": {
            "domain": ["auto", 1, "inf"],
            "type": "categorical_or_interval"
        },
        "beta_1": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "beta_2": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "early_stopping": {
            "domain": [True, False],
            "type": "categorical"
        },
        "epsilon": {
            "domain": [1e-8, "inf"],
            "type": "interval"
        },
        "hidden_layer_sizes": {
            "domain": ["tuple of ints"],
            "type": "array"
        },
        "learning_rate": {
            "domain": ["constant", "invscaling", "adaptive"],
            "type": "categorical"
        },
        "learning_rate_init": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "max_fun": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "momentum": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "n_iter_no_change": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "nesterovs_momentum": {
            "domain": [True, False],
            "type": "categorical"
        },
        "power_t": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "shuffle": {
            "domain": [True, False],
            "type": "categorical"
        },
        "solver": {
            "domain": ["lbfgs", "sgd", "adam"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "validation_fraction": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "MultiTaskElasticNet": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "l1_ratio": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "MultiTaskElasticNetCV": {
        "alphas": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "l1_ratio": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_alphas": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "MultiTaskLasso": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "MultiTaskLassoCV": {
        "alphas": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "eps": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_alphas": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "selection": {
            "domain": ["cyclic", "random"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "NuSVR": {
        "C": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "cache_size": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "coef0": {
            "domain": [-1.0, "inf"],
            "type": "interval"
        },
        "degree": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "gamma": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "kernel": {
            "domain": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [-1, "inf"],
            "type": "interval"
        },
        "nu": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "shrinking": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "OrthogonalMatchingPursuit": {
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "n_nonzero_coefs": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "normalize": {
            "domain": [True, False],
            "type": "categorical"
        },
        "precompute": {
            "domain": [True, False, "auto"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        }
    },
    "OrthogonalMatchingPursuitCV": {
        "copy": {
            "domain": [True, False],
            "type": "categorical"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "normalize": {
            "domain": [True, False],
            "type": "categorical"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "PLSCanonical": {
        "algorithm": {
            "domain": ["nipals", "svd"],
            "type": "categorical"
        },
        "copy": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_components": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "scale": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        }
    },
    "PLSRegression": {
        "copy": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_components": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "scale": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        }
    },
    "PassiveAggressiveRegressor": {
        "C": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "average": {
            "domain": [True, False, "int"],
            "type": "categorical_or_interval"
        },
        "early_stopping": {
            "domain": [True, False],
            "type": "categorical"
        },
        "epsilon": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "loss": {
            "domain": ["hinge", "squared_hinge"],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_iter_no_change": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "shuffle": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "validation_fraction": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "PoissonRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "solver": {
            "domain": ["auto", "lbfgs"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "QuantileRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "quantile": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "solver": {
            "domain": ["highs-ds", "highs-ipm", "highs"],
            "type": "categorical"
        },
        "solver_options": {
            "domain": [None, "dict"],
            "type": "categorical_or_dict"
        }
    },
    "RANSACRegressor": {
        "estimator": {
            "domain": [None, "BaseEstimator"],
            "type": "categorical_or_instance"
        },
        "is_data_valid": {
            "domain": [None, "callable"],
            "type": "categorical_or_callable"
        },
        "is_model_valid": {
            "domain": [None, "callable"],
            "type": "categorical_or_callable"
        },
        "loss": {
            "domain": ["absolute_error", "squared_error"],
            "type": "categorical"
        },
        "max_skips": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "max_trials": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "min_samples": {
            "domain": [None, "int", "float"],
            "type": "categorical_or_interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "residual_threshold": {
            "domain": [None, 0.0, "inf"],
            "type": "categorical_or_interval"
        },
        "stop_n_inliers": {
            "domain": [None, "int"],
            "type": "categorical_or_interval"
        },
        "stop_probability": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "stop_score": {
            "domain": [None, "float"],
            "type": "categorical_or_interval"
        }
    },
    "RadiusNeighborsRegressor": {
        "algorithm": {
            "domain": ["auto", "ball_tree", "kd_tree", "brute"],
            "type": "categorical"
        },
        "leaf_size": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "metric": {
            "domain": ["minkowski", "euclidean", "manhattan", "chebyshev", "haversine"],
            "type": "categorical"
        },
        "metric_params": {
            "domain": [None, "dict"],
            "type": "categorical_or_dict"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "p": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "radius": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "weights": {
            "domain": ["uniform", "distance"],
            "type": "categorical"
        }
    },
    "RandomForestRegressor": {
        "bootstrap": {
            "domain": [True, False],
            "type": "categorical"
        },
        "ccp_alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "criterion": {
            "domain": ["squared_error", "absolute_error", "poisson"],
            "type": "categorical"
        },
        "max_depth": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "max_features": {
            "domain": ["auto", "sqrt", "log2", None],
            "type": "categorical"
        },
        "max_leaf_nodes": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "max_samples": {
            "domain": [None, 0.0, 1.0],
            "type": "categorical_or_interval"
        },
        "min_impurity_decrease": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "min_samples_leaf": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "min_samples_split": {
            "domain": [2, "inf"],
            "type": "interval"
        },
        "min_weight_fraction_leaf": {
            "domain": [0.0, 0.5],
            "type": "interval"
        },
        "n_estimators": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "oob_score": {
            "domain": [True, False],
            "type": "categorical"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "Ridge": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [None, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "positive": {
            "domain": [True, False],
            "type": "categorical"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "solver": {
            "domain": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        }
    },
    "RidgeCV": {
        "alpha_per_target": {
            "domain": [True, False],
            "type": "categorical"
        },
        "alphas": {
            "domain": [None, "array-like"],
            "type": "categorical_or_array"
        },
        "cv": {
            "domain": [None, "int", "cross-validation generator", "iterable"],
            "type": "categorical_or_callable"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "gcv_mode": {
            "domain": ["auto", "svd", "eigen"],
            "type": "categorical"
        },
        "scoring": {
            "domain": [None, "str", "callable"],
            "type": "categorical_or_callable"
        },
        "store_cv_values": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "SGDRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "average": {
            "domain": [True, False, "int"],
            "type": "categorical_or_interval"
        },
        "early_stopping": {
            "domain": [True, False],
            "type": "categorical"
        },
        "epsilon": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "eta0": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "l1_ratio": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "learning_rate": {
            "domain": ["constant", "optimal", "invscaling", "adaptive"],
            "type": "categorical"
        },
        "loss": {
            "domain": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_iter_no_change": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "penalty": {
            "domain": ["l2", "l1", "elasticnet"],
            "type": "categorical"
        },
        "power_t": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "shuffle": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "validation_fraction": {
            "domain": [0.0, 1.0],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    },
    "SVR": {
        "C": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "cache_size": {
            "domain": [1.0, "inf"],
            "type": "interval"
        },
        "coef0": {
            "domain": [-1.0, "inf"],
            "type": "interval"
        },
        "degree": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "epsilon": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "gamma": {
            "domain": ["scale", "auto", "float"],
            "type": "categorical_or_interval"
        },
        "kernel": {
            "domain": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [-1, "inf"],
            "type": "interval"
        },
        "shrinking": {
            "domain": [True, False],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "TheilSenRegressor": {
        "copy_X": {
            "domain": [True, False],
            "type": "categorical"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "max_subpopulation": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "n_jobs": {
            "domain": [None, -1, 1, "inf"],
            "type": "categorical_or_interval"
        },
        "n_subsamples": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "random_state": {
            "domain": [None, 0, "inf"],
            "type": "categorical_or_interval"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        }
    },
    "TransformedTargetRegressor": {
        "check_inverse": {
            "domain": [True, False],
            "type": "categorical"
        },
        "func": {
            "domain": [None, "callable"],
            "type": "categorical_or_callable"
        },
        "inverse_func": {
            "domain": [None, "callable"],
            "type": "categorical_or_callable"
        },
        "regressor": {
            "domain": ["estimator instance"],
            "type": "categorical"
        },
        "transformer": {
            "domain": [None, "estimator instance"],
            "type": "categorical"
        }
    },
    "TweedieRegressor": {
        "alpha": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "fit_intercept": {
            "domain": [True, False],
            "type": "categorical"
        },
        "link": {
            "domain": ["auto", "identity", "log"],
            "type": "categorical"
        },
        "max_iter": {
            "domain": [1, "inf"],
            "type": "interval"
        },
        "power": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "solver": {
            "domain": ["lbfgs", "newton-cholesky"],
            "type": "categorical"
        },
        "tol": {
            "domain": [0.0, "inf"],
            "type": "interval"
        },
        "verbose": {
            "domain": [0, "inf"],
            "type": "interval"
        },
        "warm_start": {
            "domain": [True, False],
            "type": "categorical"
        }
    }
}

# Initialisation of a instance of ConfigManager
config_manager = ConfigManager("../config")

# Save sklearn details (version and date of doc scrapping)
config_manager.save_config(
    config_filename = "all_regressors_with_its_parameters_and_domains", 
    config = all_regressors_with_its_parameters_and_domains, 
    subfolder = "models_parameters/metadata",
    create = True
)

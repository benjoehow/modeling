# modeling

This codebase enables quick training, evaluation, and (soon) the deployment of machine learning models.

A single json configuration file is used to start jobs and serves as metadata for built-in reproducibility. 

This package can be deployed as a web app or embedded package.

## Input & Output

Input and output vary based on the type of job. For a cross validation job, inputs are a json config file (metaparameters) and a pandas dataframe (training data) and outputs are dataframes containing the cross validation results as well as the resulting predicitons.

For example, the following is a config for a cross validation job:
```
{
    "task": "regression",
    "validation": {
        "splitter": {
            "id": "KFold",
            "params": {
                "n_splits": 2
            }
        },
        "evalulation": {
            "metrics": ["sklearn:mean_squared_error",
                        "sklearn:mean_absolute_error"]
        }
    },
    "model": {
        "id": "xgboost",
        "params": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "num_boost_round": [500, 1000],
            "eta": 0.3,
            "max_depth": [4, 6, 8]
        }
    },
    "data": {
        "target": "alcohol",
        "features": {
            "asis": ["malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                     "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                     "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]
        }
    }
}
```


The mode is determined by the setup of the config:
- Cross validation has a "validation" section to the config and meta parameters can be supplied as an array under ["model"]["params"]
- Single model training does not have a "validation" section and only single values can be supplied under ["model"]["params"]

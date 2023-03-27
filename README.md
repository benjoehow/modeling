# modeling

This codebase is used for quick training, evaluation, and (soon) the deployment of machine learning models.

A single json configuration file is used to start jobs and serves as metadata for built-in reproducibility. 


## Input & Output

As input, the codebase takes in a json config file (metaparameters) and a pandas dataframe (training data).

The mode is determined by the setup of the config:
- Cross Validation has a "validation" section to the config and meta parameters can be supplied as an array under ["model"]["params"]
- Single model training does not have a "validation" section and only single values can be supplied under ["model"]["params"]

Automatic configuration valdation will be implemented in the future.

Depending on the mode specified by the config - training diagnostics (evaluation metrics, feature importances) or a trained model are output. 

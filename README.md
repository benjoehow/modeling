# modeling

This codebase is used for the training and evaluation of machine learning models.

A single configuration file is used to kick off jobs and serves as metadata for traceability and efficient reproduction. 

## Functional Architecture

There are five main types of classes
1. **Adapters**: Translates the various algorithm implmentations to a common interface
2. **Expeditor**: Compiles what needs to computed (tasks) with Adapaters into Orders
3. **Orders**: Record of the tasks that need to be done and what tasks are completed 
3. **Runner** Coordinates tasks from an Order to a Processor 
3. **Processors**: Determines how (e.g. serial, parrallel) tasks are executed

## Input & Output

As input, the codebase takes in a json config file (metaparameters) and a pandas dataframe (training data).

The mode is determined by the setup of the config:
- Cross Validation has a "validation" section to the config and meta parameters can be supplied as an array under ["model"]["params"]
- Single model training does not have a "validation" section and only single values can be supplied under ["model"]["params"]

Automatic configuration valdation will be implemented in the future.

Depending on the mode specified by the config - training diagnostics (evaluation metrics, feature importances) or a trained model are output. 

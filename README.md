# modeling

This codebase is used for the training and evaluation of machine learning models in a way that is configurable and reproducible.

A single configuration file details model training and stores metadata during training. This enables traceability after training and efficient reproduction.

## Functional Architecture

There are three main types of classes
1. **Adapters**: Translates the various algorithm implmentations to a common interface
2. **Expeditor**: Compiles what needs to computed (jobs), keeps track of results, and is used to save state during processing. 
3. **Processors**: Determines how jobs from the journalist are executed.

## Input

As input, the codebase takes in a json config file (metaparameters) and a pandas dataframe (training data).
Training diagnostics (evaluation metrics, feature importances) and a trained model for prediction are output. 

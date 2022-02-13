# modeling

This package is used to train and evaluate machine learning models from varying different algorithms. 

It follows the general interface that models are a product of training data and metaparameters. 

Additionally, validation routines such as cross validation will also be supported.

## Functional Architecture

There are three main types of classes
1. **Adapters**: Translates the various algorithm implmentations to a common interface
2. **Expeditor**: Compiles what needs to computed (jobs), keeps track of results, and is used to save state during processing. 
3. **Processors**: Determines how jobs from the journalist are executed.

## Input

As input, the codebase takes in a json config file (metaparameters) and a pandas dataframe (training data).
Training diagnostics (evaluation metrics, feature importances) and a trained model for prediction are output. 

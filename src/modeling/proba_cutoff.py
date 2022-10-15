import pandas as pd

def cutoff_proba(prediction, target, strategy = "data"):
    cutoff_value = _get_data_proba_cutoff(prediction = prediction,
                                          target = target)
    
    predictions = prediction[prediction >= cutoff_value]
    
    return predictions


def _get_data_proba_cutoff(prediction, target):
    percentile_cutoff = len(target[target==True])/len(target)
    cutoff =  prediction.quantile(q = percentile_cutoff, 
                                  interpolation = "midpoint")
    return cutoff
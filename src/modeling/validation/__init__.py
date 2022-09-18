import sklearn.model_selection as selection

_split_lookup = {}
_split_lookup["KFold"] = selection.KFold
_split_lookup["RepeatedKFold"] = selection.RepeatedKFold
_split_lookup["TimeSeriesSplit"] = selection.TimeSeriesSplit

def splitter_factory(key):
    
    if key not in _split_lookup.keys():
        raise ValueError("Splitter key not found")
        
    return _split_lookup[key]

__all__ = [
    "splitter_factory"
]

import sklearn.metrics as skmetrics

_metric_lookup = {"sklearn": {}}
_metric_lookup["sklearn"]["auc"] = skmetrics.auc
_metric_lookup["sklearn"]["average_precision_score"] = skmetrics.average_precision_score
_metric_lookup["sklearn"]["balanced_accuracy_score"] = skmetrics.balanced_accuracy_score
_metric_lookup["sklearn"]["brier_score_loss"] = skmetrics.brier_score_loss
_metric_lookup["sklearn"]["classification_report"] = skmetrics.classification_report
_metric_lookup["sklearn"]["cohen_kappa_score"] = skmetrics.cohen_kappa_score
_metric_lookup["sklearn"]["confusion_matrix"] = skmetrics.confusion_matrix
_metric_lookup["sklearn"]["dcg_score"] = skmetrics.dcg_score
_metric_lookup["sklearn"]["det_curve"] = skmetrics.det_curve
_metric_lookup["sklearn"]["f1_score"] = skmetrics.f1_score
_metric_lookup["sklearn"]["fbeta_score"] = skmetrics.fbeta_score
_metric_lookup["sklearn"]["hamming_loss"] = skmetrics.hamming_loss
_metric_lookup["sklearn"]["hinge_loss"] = skmetrics.hinge_loss
_metric_lookup["sklearn"]["jaccard_score"] = skmetrics.jaccard_score
_metric_lookup["sklearn"]["log_loss"] = skmetrics.log_loss
_metric_lookup["sklearn"]["matthews_corrcoef"] = skmetrics.matthews_corrcoef
_metric_lookup["sklearn"]["multilabel_confusion_matrix"] = skmetrics.multilabel_confusion_matrix
_metric_lookup["sklearn"]["ndcg_score"] = skmetrics.ndcg_score
_metric_lookup["sklearn"]["precision_recall_curve"] = skmetrics.precision_recall_curve
_metric_lookup["sklearn"]["precision_recall_fscore_support"] = skmetrics.precision_recall_fscore_support
_metric_lookup["sklearn"]["precision_score"] = skmetrics.precision_score
_metric_lookup["sklearn"]["recall_score"] = skmetrics.recall_score
_metric_lookup["sklearn"]["explained_variance"] = skmetrics.explained_variance_score
_metric_lookup["sklearn"]["max_error"] = skmetrics.max_error
_metric_lookup["sklearn"]["neg_mean_absolute_error"] = skmetrics.mean_absolute_error
_metric_lookup["sklearn"]["neg_mean_squared_error"] = skmetrics.mean_squared_error
_metric_lookup["sklearn"]["neg_root_mean_squared_error"] = skmetrics.mean_squared_error
_metric_lookup["sklearn"]["neg_mean_squared_log_error"] = skmetrics.mean_squared_log_error
_metric_lookup["sklearn"]["neg_median_absolute_error"] = skmetrics.median_absolute_error
_metric_lookup["sklearn"]["r2"] = skmetrics.r2_score
_metric_lookup["sklearn"]["neg_mean_poisson_deviance"] = skmetrics.mean_poisson_deviance
_metric_lookup["sklearn"]["neg_mean_gamma_deviance"] = skmetrics.mean_gamma_deviance
_metric_lookup["sklearn"]["neg_mean_absolute_percentage_error"] = skmetrics.mean_absolute_percentage_error
#_metric_lookup["sklearn"]["d2_absolute_error_score"] = skmetrics.d2_absolute_error_score
#_metric_lookup["sklearn"]["d2_pinball_score"] = skmetrics.d2_pinball_score
#_metric_lookup["sklearn"]["d2_tweedie_score"] = skmetrics.d2_tweedie_score

def metric_factory(key):
    
    key_split = key.split(":")
    library = key_split[0]
    metric= key_split[1]
    
    if library not in _metric_lookup.keys():
        raise ValueError(f"Library key: {key} not found")
    if metric not in _metric_lookup[library].keys():
        raise ValueError(f"Metric key: {key} not found")
        
    return _metric_lookup[library][metric]

__all__ = [
    "splitter_factory"
]

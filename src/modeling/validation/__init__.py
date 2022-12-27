import sklearn.model_selection as selection
import sklearn.metrics as skmetrics
from .eval_func import EvalFuncWrapper

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


base_metrics = [
    ["sklearn", "auc", skmetrics.auc],
    ["sklearn", "average_precision_score", skmetrics.average_precision_score],
    ["sklearn", "balanced_accuracy_score", skmetrics.balanced_accuracy_score],
    ["sklearn", "brier_score_loss", skmetrics.brier_score_loss],
    ["sklearn", "classification_report", skmetrics.classification_report],
    ["sklearn", "cohen_kappa_score", skmetrics.cohen_kappa_score],
    ["sklearn", "confusion_matrix", skmetrics.confusion_matrix],
    ["sklearn", "dcg_score", skmetrics.dcg_score],
    ["sklearn", "det_curve", skmetrics.det_curve],
    ["sklearn", "f1_score", skmetrics.f1_score],
    ["sklearn", "fbeta_score", skmetrics.fbeta_score],
    ["sklearn", "hamming_loss", skmetrics.hamming_loss],
    ["sklearn", "hinge_loss", skmetrics.hinge_loss],
    ["sklearn", "jaccard_score", skmetrics.jaccard_score],
    ["sklearn", "log_loss", skmetrics.log_loss],
    ["sklearn", "matthews_corrcoef", skmetrics.matthews_corrcoef],
    ["sklearn", "multilabel_confusion_matrix", skmetrics.multilabel_confusion_matrix],
    ["sklearn", "ndcg_score", skmetrics.ndcg_score],
    ["sklearn", "precision_recall_curve", skmetrics.precision_recall_curve],
    ["sklearn", "precision_recall_fscore_support", skmetrics.precision_recall_fscore_support],
    ["sklearn", "precision_score", skmetrics.precision_score, ["requires_binary_output"]],
    ["sklearn", "recall_score", skmetrics.recall_score],
    ["sklearn", "explained_variance", skmetrics.explained_variance_score],
    ["sklearn", "max_error", skmetrics.max_error],
    ["sklearn", "mean_absolute_error", skmetrics.mean_absolute_error],
    ["sklearn", "mean_squared_error", skmetrics.mean_squared_error],
    ["sklearn", "root_mean_squared_error", skmetrics.mean_squared_error],
    ["sklearn", "mean_squared_log_error", skmetrics.mean_squared_log_error],
    ["sklearn", "median_absolute_error", skmetrics.median_absolute_error],
    ["sklearn", "r2", skmetrics.r2_score],
    ["sklearn", "neg_mean_poisson_deviance", skmetrics.mean_poisson_deviance],
    ["sklearn", "neg_mean_gamma_deviance", skmetrics.mean_gamma_deviance],
    ["sklearn", "neg_mean_absolute_percentage_error", skmetrics.mean_absolute_percentage_error]
]

_metric_lookup = {"sklearn": {}}
for metric in base_metrics:
    func_tmp = EvalFuncWrapper(*metric)
    _metric_lookup[func_tmp.family_id][func_tmp.id] = func_tmp

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

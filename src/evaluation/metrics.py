import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def d_calibration(y_true, y_pred, n_bins=10):
    """
    Calculate d-calibration metric for binary classification.

    Parameters:
    - y_true (np.ndarray): Ground truth binary labels
    - y_pred (np.ndarray): Predicted probabilities
    - n_bins (np.ndarray): Number of bins for the probability partition

    Returns:
    - d_calibration_value (float): d-calibration score
    - p_value (float): p_value for chi-squared test
    """
    # ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # define bin edges and bin assignments
    bin_edges = np.linspace(0, 1, n_bins+1)
    # bins are 0-indexed
    bin_indices = np.digitize(y_pred, bin_edges) - 1

    # initialize counts
    observed_counts = np.zeros(n_bins)
    expected_counts = np.zeros(n_bins)

    # compute observed and expted counts for each bin
    for bin_idx in range(n_bins):
        # find samples in the current bin
        in_bin = (bin_indices == bin_idx)
        bin_size = np.sum(in_bin)

        # skip empty bins
        if bin_size == 0:
            continue

        # observed counts: sum of true positives in the bin
        observed_counts[bin_idx] = np.sum(y_true[in_bin])

        expected_counts[bin_idx] = np.sum(y_pred[in_bin])
    
    # compute chi-squared statistic - add epsilon to avoid div by 0
    chi_squared_stat = np.sum((observed_counts - expected_counts) ** 2 / (expected_counts + 1e-10))
    p_value = chi2.sf(chi_squared_stat, df=n_bins - 1)

    return chi_squared_stat, p_value


def calculate_metrics(y_test, y_pred, threshold=0.5):
    y_pred_binary = y_pred > threshold

    metric_labels = [
        'auc',
        'ap',
        'accuracy',
        'recall',
        'specificity',
        'brier',
        'dcal',
        'dcal_pvalue',
    ]

    if np.min(y_test) == np.max(y_test):
        #return {metric_name:-100 for metric_name in metric_labels}
        return None

    dcal, dcal_pvalue = d_calibration(y_test, y_pred, n_bins=50)
    dcal_pvalue = int(dcal_pvalue >= 0.05)
    return {
        'auc': roc_auc_score(y_test, y_pred),
        'ap': average_precision_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'specificity': specificity_score(y_test, y_pred_binary),
        'brier': brier_score_loss(y_test, y_pred),
        'dcal': dcal,
        'dcal_pvalue': dcal_pvalue,
    }


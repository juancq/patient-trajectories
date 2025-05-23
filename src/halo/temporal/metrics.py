import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_recall_curve
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
        logger.warning('Only one class present in y_test. Some metrics are undefined.')
        return None

    dcal, dcal_pvalue = d_calibration(y_test, y_pred, n_bins=50)
    dcal_pvalue = int(dcal_pvalue >= 0.05)
    precision_at_recall_50, _ = precision_at_fixed_recall(
        y_true=y_test, 
        y_scores=y_pred,
        fixed_recall=0.5,
    )
    precision_at_recall_25, _ = precision_at_fixed_recall(
        y_true=y_test, 
        y_scores=y_pred,
        fixed_recall=0.25,
    )
    recall_at_specificity_90, _ = calculate_recall_at_specificity(
        true_values=y_test, 
        predictions=y_pred,
        target_specificity=0.90,
    )
    return {
        'auc': roc_auc_score(y_test, y_pred),
        'ap': average_precision_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'specificity': specificity_score(y_test, y_pred_binary),
        'brier': brier_score_loss(y_test, y_pred),
        'dcal': dcal,
        'dcal_pvalue': dcal_pvalue,
        'precision_at_recall_50': precision_at_recall_50,
        'precision_at_recall_25': precision_at_recall_25,
        'recall_at_specificity_90': recall_at_specificity_90,
    }


def precision_at_fixed_recall(y_true, y_scores, fixed_recall=0.8):
    """
    Calculate precision at a fixed recall/sensitivity value.
    
    Parameters:
    -----------
    y_true : array-like
        Binary true labels (0 or 1)
    y_scores : array-like
        Prediction scores, higher values predict label 1
    fixed_recall : float, default=0.8
        The fixed recall/sensitivity value (between 0 and 1)
        
    Returns:
    --------
    precision : float
        Precision at the specified recall value
    threshold : float
        Score threshold that achieves the specified recall
    """
    # Calculate precision-recall curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find the closest recall value to our fixed recall
    closest_idx = np.argmin(np.abs(recall_vals - fixed_recall))
    
    # Check if we found an exact match or the closest value
    if closest_idx == len(recall_vals) - 1:
        # Special case: The last threshold doesn't have a corresponding precision
        threshold = 0.0
    else:
        threshold = thresholds[closest_idx]
    
    # Get the precision at that index
    precision = precision_vals[closest_idx]
    
    return precision, threshold


def calculate_recall_at_specificity(true_values, predictions, target_specificity):
    """
    Efficiently calculates recall at a specific specificity level for large datasets.
    
    Args:
        true_values (np.array): Array of true binary values (0 or 1).
        predictions (np.array): Array of prediction probabilities (values between 0 and 1).
        target_specificity (float): The desired specificity level (between 0 and 1).
    
    Returns:
        tuple: (recall, threshold) - The recall value at the target specificity and the threshold.
               Returns (None, None) if the target specificity cannot be achieved.
    """
    # Input validation
    if not 0 <= target_specificity <= 1:
        raise ValueError("Target specificity must be between 0 and 1")
    
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    if len(true_values) != len(predictions):
        raise ValueError("Length of true_values and predictions must be the same")
    
    if len(true_values) == 0:
        return None, None
    
    # Count total positives and negatives
    total_positives = np.sum(true_values == 1)
    total_negatives = np.sum(true_values == 0)
    
    if total_negatives == 0:
        return None, None  # Cannot calculate specificity without negative samples
    
    if total_positives == 0:
        return None, None  # Cannot calculate recall without positive samples
    
    # Sort predictions and corresponding true values together
    sorted_indices = np.argsort(predictions)
    sorted_predictions = predictions[sorted_indices]
    sorted_true = true_values[sorted_indices]
    
    # Precompute if each example is positive or negative
    pos_examples = (sorted_true == 1)
    neg_examples = (sorted_true == 0)
    
    # Count cumulative positives and negatives as we move through thresholds
    # These represent FN and TN when using the index position as threshold
    cum_positives = np.cumsum(pos_examples)
    cum_negatives = np.cumsum(neg_examples)
    
    # Calculate TP, FP, TN, FN at each threshold
    # At position i, values with index >= i are classified as positive
    tp = total_positives - cum_positives
    fp = total_negatives - cum_negatives
    tn = cum_negatives
    fn = cum_positives
    
    # Calculate specificity and recall at each position
    specificity = tn / total_negatives  # TN / (TN + FP)
    recall = tp / total_positives       # TP / (TP + FN)
    
    # Find positions where specificity >= target_specificity
    valid_positions = np.where(specificity >= target_specificity)[0]
    
    if len(valid_positions) == 0:
        return None, None  # Target specificity not achievable
    
    # Find the position with the highest recall among valid positions
    best_position = valid_positions[np.argmax(recall[valid_positions])]
    best_recall = recall[best_position]
    
    # Get the corresponding threshold
    # If we're at the last position, use a threshold slightly higher than the maximum prediction
    if best_position >= len(sorted_predictions) - 1:
        best_threshold = sorted_predictions[-1] + 1e-10
    else:
        # Otherwise, use the average of the current and next prediction value
        best_threshold = (sorted_predictions[best_position] + sorted_predictions[best_position + 1]) / 2
    
    return best_recall, best_threshold

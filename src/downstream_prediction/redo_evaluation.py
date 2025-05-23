import pandas as pd
import polars as pl
import numpy as np
import os
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve


def evaluate_subsets(subset_data, y_test, y_pred, test_index, scores):
    for subset_name, subset_mask in subset_data.items():
        subset_y_pred = y_pred[subset_mask[test_index]]
        subset_y_test = y_test[subset_mask[test_index]]

        # calculate metrics for this fold
        metrics = calculate_metrics(y_test=subset_y_test, y_pred=subset_y_pred, threshold=0.5)
        # Store metrics and predictions
        for metric_name, metric_value in metrics.items():
            scores[subset_name][metric_name].append(metric_value)


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

def calculate_accuracy(file_path, threshold=0.5, fold_subset_mask=None):
    """
    Calculate accuracy from a CSV file containing ML model predictions.
    
    Parameters:
    file_path (str): Path to the CSV file
    threshold (float): Threshold for converting probabilities to binary predictions (default: 0.5)
    
    Returns:
    float: Accuracy score
    """
    try:
        # Read the CSV file
        #df = pd.read_csv(file_path)
        df = pl.read_csv(file_path)
        
        subset_results = {}
        # Check if the required columns exist
        required_columns = ['prediction', 'true']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        
        # Convert probabilities to binary predictions using the threshold
        true_values = df['true']
        predictions = df['prediction']
        precision, threshold_at_target = precision_at_fixed_recall(true_values, predictions, 
                            threshold)

        for subset_name, subset_index in fold_subset_mask.items():
            subset_true_values = true_values.filter(subset_index)
            subset_predictions = predictions.filter(subset_index)
            subset_results[subset_name], _ = precision_at_fixed_recall(subset_true_values, subset_predictions, threshold)
        
        accuracy = accuracy_score(true_values, predictions > threshold_at_target)
        
        return {'accuracy':accuracy, 'precision':precision, 'subset': subset_results}
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_all_csvs_in_folds(base_dir, threshold=0.5):
    """
    Process all CSV files in all fold directories and calculate metrics.
    
    Parameters:
    base_dir (str): Base directory containing fold_0, fold_1, etc. subdirectories
    threshold (float): Threshold for binary predictions
    
    Returns:
    dict: Dictionary of metrics for each CSV file type
    """
    base_path = Path(base_dir)
    
    # Expected fold directory names
    fold_dirs = [f"fold_{i}" for i in range(5)]
    
    # Dictionary to store accuracies for each CSV file type
    # Use defaultdict to automatically create lists for new keys

    with open(base_path / 'fold_indices_diagnosis.pkl', 'rb') as f:
        indices = pickle.load(f)
    subset = indices['subset_data']
    #subset_precision = {subset_group:[] for subset_group in subset.keys()}
    #csv_precision = {subset_group:[] for subset_group in subset.keys()}
    csv_precision = defaultdict(lambda: defaultdict(list))
    
    # Process each fold directory
    for i,fold_dir in enumerate(fold_dirs):
        fold_path = base_path / fold_dir
        
        if not fold_path.exists():
            print(f"Warning: Fold directory not found at {fold_path}")
            continue
        
        # Find all CSV files in this fold directory
        csv_files = list(fold_path.glob("*.csv"))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {fold_path}")
            continue

        # train,test fold indices
        _, test_fold_indices = indices['fold_indices'][i]
        fold_subset_mask = {}
        for subset_name, subset_mask in subset.items():
            fold_subset_mask[subset_name] = subset_mask[test_fold_indices]

        include = ['halo', 'esgpt_', 'baseline']
        # Process each CSV file in this fold
        for csv_file in csv_files:
            csv_name = csv_file.name
            if not any([x in csv_name for x in include]) or 'elasticnet' in csv_name.lower():
                continue
            result = calculate_accuracy(csv_file, threshold, fold_subset_mask)

            if result:
                precision = result['precision']

                csv_precision[csv_name]['full'].append(precision)
                print(f"{fold_dir} - {csv_name}: precision = {precision:.4f} ({precision*100:.2f}%)")

                for subset_name, subset_fold_precision in result['subset'].items():
                    csv_precision[csv_name][subset_name].append(subset_fold_precision)
            else:
                for subset_name in csv_precision[csv_name].keys():
                    csv_precision[csv_name][subset_name].append(-99)

    
    # Calculate aggregate metrics for each CSV file type
    results = []
    for csv_name, subset in csv_precision.items():
        
        clean_csv_name = csv_name.replace('_predictions.csv', '')
        name_split = clean_csv_name.split('_')
        model_name = name_split[-1]
        feature_set = '_'.join(name_split[:-1])
        
        #if not precisions:
        #    print(f"No valid accuracy values were calculated for {csv_name}")
        #    continue
        for subset_name, precisions in subset.items():
            avg_precision = np.mean(precisions)
            std_precision = np.std(precisions)
            confidence_interval_precision = 1.96 * std_precision / np.sqrt(len(precisions))

            results.append(
                {
                'subset': subset_name,
                'feature_set': feature_set,
                'model': model_name,
                'avg_precision': avg_precision,
                'std_precision': std_precision,
                'precision': f'{avg_precision:.3f} ({std_precision:.3f})',
                'precision_confidence_interval': confidence_interval_precision,
                'fold_precision': precisions
                }
            )

    
    return results

def print_results(results):
    """Print results in a formatted way."""
    print("\n" + "="*80)
    print(f"{'CSV File':<30} {'Avg Accuracy':<15} {'Std Dev':<15} {'95% CI':<15}")
    print("-"*80)
    
    for csv_name, metrics in sorted(results.items()):
        avg = metrics['avg_accuracy']
        std = metrics['std_accuracy']
        ci = metrics['confidence_interval']
        
        print(f"{csv_name:<30} {avg:.4f} ({avg*100:.2f}%) {std:.4f} ({std*100:.2f}%) ±{ci:.4f}")
    
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate cross-validation accuracy for all CSV files across folds')
    parser.add_argument('--base_dir', type=str, help='Base directory containing fold_0, fold_1, etc. subdirectories',
                        default='./folds')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold for converting probabilities to binary predictions (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results as CSV (optional)')
    
    args = parser.parse_args()

    #args.base_dir = './folds'
    
    for task in ['episode_count', 'length_of_stay', 'cost']:
        all_df = []
        for horizon in ['six_months', 'one_year', 'two_year']:
            base_dir = f'./folds/{task}/{horizon}'
            results = process_all_csvs_in_folds(base_dir, args.threshold)
            
            if results:
                # Save results to CSV if output file is specified
                output_df = pd.DataFrame(results)
                output_df['window'] = horizon
                output_df = output_df.sort_values(by=['window','subset','feature_set', 'model'])
                all_df.append(output_df)
            else:
                print("No valid results were calculated.")

        all_df = pd.concat(all_df)
        #output_name = f'threshold_metrics_{task}_{args.output}'
        output_name = f'precision_at_fixed_recall_subset_{args.threshold:0.2f}_{task}.csv'
        all_df.to_csv(output_name, index=False)
        print(f"\nResults saved to {args.output}")

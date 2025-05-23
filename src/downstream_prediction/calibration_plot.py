import argparse
import pandas as pd
import numpy as np
import os
from loguru import logger
import pickle
from pathlib import Path
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats


def process_all_csvs_in_folds(base_dir):
    """
    Process all CSV files in all fold directories and calculate metrics.
    
    Parameters:
    base_dir (str): Base directory containing fold_0, fold_1, etc. subdirectories
    
    Returns:
    dict: Dictionary of metrics for each CSV file type
    """
    base_path = Path(base_dir)
    
    # Expected fold directory names
    fold_dirs = [f"fold_{i}" for i in range(5)]
    
    with open(base_path / 'fold_indices_diagnosis.pkl', 'rb') as f:
        indices = pickle.load(f)

    predictions = []
    
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

        include = ['baseline_lgbm', 'esgpt_lgbm', 'halo_lgbm', 'halo_finetune']
        name_map = {
            'baseline_lgbm': 'Standard Model', 
            'esgpt_lgbm': 'ESGPT',
            'halo_lgbm': 'HALO',
            'halo_finetune': 'Specialized HALO',
        }
        # Process each CSV file in this fold
        for csv_file in csv_files:
            csv_name = csv_file.name
            if not any([x in csv_name for x in include]) or 'elasticnet' in csv_name.lower():
                continue

            df = pd.read_csv(csv_file)
            df = df.rename(columns={'prediction':'y_prob', 'true': 'y_true'})
            df['fold'] = i

            clean_csv_name = csv_file.name.replace('_predictions.csv', '')
            model = name_map[clean_csv_name]
            df['model'] = model
            predictions.append(df)
        
    return pd.concat(predictions)

def generate_calibration_plot(
    full_data, output_filepath, task, horizon, model_names,
    strategy, n_bins,
    ci=True
):
    #plt.figure(figsize=(10, 8))
    plt.figure(figsize=(4, 3))

    legend_handles = []
    plt.plot([0,1], [0,1], 'k--', linewidth=1.5, alpha=0.8, label='Perfectly calibrated')
    # add patch for legend
    handle = mpatches.Patch(color='k', label='Perfectly calibrated')
    legend_handles.append(handle)
    font_size = 12.5

    model_colors = sns.color_palette("Set1", len(model_names))

    for i, model_name in enumerate(model_names):
        # get all data for this model
        model_data = full_data[full_data['model']==model_name]

        if ci:
            # calculate calibration curve with CIs
            prob_true, prob_pred, ci_lower, ci_upper = calibration_curve_with_ci(
                model_data['y_true'], model_data['y_prob'], 
                n_bins=n_bins, strategy=strategy
            )
        else:
            # calculate calibration curve without CIs from bootstraps
            prob_true, prob_pred, _, _ = calibration_curve_with_ci(
                model_data['y_true'], model_data['y_prob'], 
                n_bins=n_bins, strategy=strategy, 
                n_bootstraps=None
            )

        # plot the main calibration curve
        plt.plot(prob_pred, prob_true, '-', color=model_colors[i],
            linewidth=2.0, label=model_name,
            #alpha=0.85, 
        )

        if ci:
            # add CI as shaded region
            plt.fill_between(prob_pred, ci_lower, ci_upper, alpha=0.2, color=model_colors[i])

        # add patch for legend
        handle = mpatches.Patch(color=model_colors[i], label=model_name)
        legend_handles.append(handle)

    plt.xlabel('Predicted Probability', fontsize=font_size, fontweight='bold')
    if 'count' in task:
        plt.ylabel('Fraction of Positives', fontsize=font_size, fontweight='bold')
    plt.title(
        #f'Calibration Plot - Model Comparison - {task.replace("_", " ").capitalize()}',
        f'{task.replace("_", " ").capitalize()}',
        fontsize=13
    )

    plt.grid(True, alpha=0.3)

    if 'cost' in task:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        #loc = 'best'
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1,1), fontsize=12)
    plt.savefig(f'{output_filepath}.png', bbox_inches='tight', dpi=400)


def calibration_curve_with_ci(y_true, y_prob, n_bins=10, strategy='uniform', n_bootstraps=100):
    """Calculate calibration curve with 95% confidence intervals using bootstrapping"""

    # calculate standard calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    if n_bootstraps is None:
        return prob_true, prob_pred, None, None

    bootstrap_results = np.zeros((n_bootstraps, len(prob_true)))

    # perform bootstrapping
    n_samples = len(y_true)
    for i in range(n_bootstraps):
        # sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true.iloc[indices]
        y_prob_boot = y_prob.iloc[indices]

        # calculate calibration curve
        prob_true_boot, _ = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )
        bootstrap_results[i,:] = prob_true_boot

    # calculate confidence intervals
    ci_lower = np.percentile(bootstrap_results, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_results, 97.5, axis=0)

    return prob_true, prob_pred, ci_lower, ci_upper
    
def main():
    parser = argparse.ArgumentParser(description='Calculate cross-validation accuracy for all CSV files across folds')
    parser.add_argument('--base_dir', type=str, help='Base directory containing fold_0, fold_1, etc. subdirectories',
                        default='./folds')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results as CSV (optional)')
    parser.add_argument('--temporal', action='store_true', 
                        help='Generate plot for temporal split')
    
    args = parser.parse_args()

    #args.base_dir = './folds'
    
    if args.temporal:
        for task in ['episode_count', 'length_of_stay', 'cost']:
            horizon = 'six_months'
            base_dir = f'./temporal/'
            output_name = f'./calibration_plot/temporal_calibration_plot_{task}_{horizon}'

            suffix = f'{task}_{horizon}_predictions.csv'
            prediction_files = {
                'Standard Model': f'temporal_baseline_{suffix}',
                'HALO': f'temporal_halo_{suffix}',
                'Specialized HALO': f'temporal_halo_finetune_{suffix}',
            }

            full_data = []
            for model_name, input_file in prediction_files.items():
                df = pd.read_csv(Path(base_dir) / input_file)
                df['model'] = model_name
                df = df.rename(columns={'prediction':'y_prob', 'true': 'y_true'})
                full_data.append(df)
            full_data = pd.concat(full_data)

            model_names = ['Standard Model', 'HALO', 'Specialized HALO']
            logger.info('Generating plot for temporal data')
            # generate calibration plot, one plot with a line per model
            generate_calibration_plot(
                full_data, output_name, 
                task, horizon,
                model_names,
                strategy='quantile', n_bins=50, ci=False
            )
    else:
        for task in ['episode_count', 'length_of_stay', 'cost']:
            for horizon in ['six_months', 'one_year', 'two_year']:
                base_dir = f'./folds/{task}/{horizon}'
                output_name = f'./calibration_plot/internal_validation_{task}_{horizon}'

                # aggregate predictions across all folds
                results = process_all_csvs_in_folds(base_dir)

                model_names = ['Standard Model', 'ESGPT', 'HALO', 'Specialized HALO']
                logger.info(f'Generating plot for {task} horizon {horizon}')
                # generate calibration plot, one plot with a line per model
                generate_calibration_plot(
                    results, output_name, 
                    task, horizon,
                    model_names,
                    strategy='quantile', n_bins=50,
                    ci=False
                )

if __name__ == "__main__":
    main()

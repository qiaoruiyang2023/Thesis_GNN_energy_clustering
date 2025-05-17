# src/viz_metrics.py

import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup basic logging (configure in main script usually)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Evaluation Summary Plot ---
def plot_evaluation_summary(config: dict, metrics_dict: dict):
    """
    Creates a bar chart summarizing the main averaged evaluation metrics.

    Args:
        config (dict): Configuration dictionary. Expected keys: 'SCENARIO_RESULTS_DIR', 'SCENARIO_NAME'.
        metrics_dict (dict): Dictionary containing averaged evaluation metrics
                             (keys should match outputs from evaluation.py).
    """
    if not metrics_dict:
        logger.warning("Metrics dictionary is empty. Skipping evaluation summary plot.")
        return

    logger.info("Generating evaluation summary plot...")
    plot_dir = os.path.join(config.get('SCENARIO_RESULTS_DIR', 'results/default'), 'plots')
    os.makedirs(plot_dir, exist_ok=True) # Ensure plot directory exists
    save_path = os.path.join(plot_dir, 'evaluation_summary_bars.png')

    # Define metrics potentially available in the dictionary
    metric_keys = [
        'avg_ari', 'avg_nmi', 'avg_churn_rate',
        'avg_cluster_ssr', 'avg_silhouette_score',
        'avg_feeder_violation_rate', 'avg_distance_violation_rate',
        'avg_abs_cluster_net_import', 'total_net_import' # Added based on potential evaluation output
    ]
    # Extract available metrics and their values, handle potential missing keys
    plot_metrics = {k: metrics_dict.get(k) for k in metric_keys if k in metrics_dict}
    # Filter out None or NaN values for plotting
    plot_metrics = {k: v for k, v in plot_metrics.items() if pd.notna(v)}

    if not plot_metrics:
        logger.warning("No valid averaged metrics found to plot in evaluation summary.")
        return

    try:
        plt.figure(figsize=(max(10, len(plot_metrics) * 1.5), 7)) # Adjust width based on num metrics
        keys = list(plot_metrics.keys())
        values = list(plot_metrics.values())

        bars = plt.bar(keys, values, color=sns.color_palette("viridis", len(keys)))
        plt.ylabel("Average Metric Value")
        plt.title(f"Evaluation Metric Summary - {config.get('SCENARIO_NAME', 'Unknown')}")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(bottom=min(0, min(values)*1.1 if values else 0) ) # Adjust y-lim for negative values (like silhouette)

        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}',
                     va='bottom' if yval >= 0 else 'top', # Adjust label position for neg values
                     ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        logger.info(f"Saved evaluation summary plot to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate evaluation summary plot: {e}")


# --- Time Series Metric Plots ---
def plot_metric_timeseries(config: dict):
    """
    Plots stability, synergy, and quality metrics over time from saved CSV files.
    Assumes evaluation.py saved these files in the scenario results directory.
    """
    logging.info("Generating metric time series plots...")
    results_dir = config.get('SCENARIO_RESULTS_DIR', 'results/default')
    plot_dir = os.path.join(results_dir, 'plots', 'metric_timeseries')
    os.makedirs(plot_dir, exist_ok=True)

    # Define structure for finding files and plotting
    files_metrics = {
        'stability': {'file': 'stability_metrics_timeseries.csv',
                      'cols': ['ari', 'nmi', 'churn_rate'],
                      'xlabel': 'Time Transition (t-1 -> t)',
                      'x_col': 'time_index_t'},
        'quality': {'file': 'quality_metrics_timeseries.csv',
                    'cols': ['silhouette_score'],
                    'xlabel': 'Time Index',
                    'x_col': 'time_index'},
        'synergy': {'file': 'synergy_metrics_timeseries.csv',
                    'cols': ['ssr', 'net_import'], # Add abs_net_load_sum? sum_abs_net_load?
                    'xlabel': 'Time Index (Avg. over Clusters)',
                    'x_col': 'time_index'}
    }

    for metric_type, info in files_metrics.items():
        file_path = os.path.join(results_dir, info['file'])
        if not os.path.exists(file_path):
            logger.warning(f"Metric file not found: {info['file']}. Skipping {metric_type} time series plot.")
            continue

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                 logger.warning(f"{info['file']} is empty. Skipping {metric_type} plot.")
                 continue

            x_col = info['x_col']
            if x_col not in df.columns:
                 logger.error(f"X-axis column '{x_col}' not found in {info['file']}. Skipping plot.")
                 continue

            plot_cols = [col for col in info['cols'] if col in df.columns]
            if not plot_cols:
                 logger.warning(f"No specified metric columns found in {info['file']}. Skipping plot.")
                 continue

            # Create subplots for each metric in the file
            n_plots = len(plot_cols)
            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True, squeeze=False)
            fig.suptitle(f"{metric_type.capitalize()} Metrics Over Time - {config.get('SCENARIO_NAME', '')}", fontsize=14, y=1.02)

            for i, col in enumerate(plot_cols):
                ax = axes[i, 0]
                if metric_type == 'synergy':
                    # Plot average across clusters per time step, show std dev band
                    sns.lineplot(data=df, x=x_col, y=col,
                                 estimator='mean', errorbar='sd', # Show mean and std dev across clusters
                                 ax=ax, marker='.', legend=False)
                    ax.set_ylabel(f"Avg {col.replace('_', ' ').title()}")
                else:
                    # Stability and Quality are single values per time step/transition
                    sns.lineplot(data=df, x=x_col, y=col, ax=ax, marker='.')
                    ax.set_ylabel(col.replace('_', ' ').title())

                ax.set_title(f"{col.replace('_', ' ').title()}")
                ax.grid(True, linestyle='--', alpha=0.6)

            axes[-1, 0].set_xlabel(info['xlabel']) # Set x-label only on the bottom plot
            plt.tight_layout() # Adjust layout after titles etc.
            save_path = os.path.join(plot_dir, f'{metric_type}_timeseries.png')
            plt.savefig(save_path, dpi=120)
            plt.close(fig)
            logger.info(f"Saved {metric_type} time series plot to: {save_path}")

        except Exception as e:
            logger.error(f"Failed to generate {metric_type} time series plot from {info['file']}: {e}")


# --- K Selection Plots ---
def plot_k_selection_metrics(config: dict, k_metrics_df: pd.DataFrame):
    """
    Plots metrics used for K selection (e.g., Silhouette vs. K).

    Args:
        config (dict): Config dictionary. Expected: 'SCENARIO_RESULTS_DIR', 'SCENARIO_NAME'.
        k_metrics_df (pd.DataFrame): DataFrame with columns like 'k' and 'avg_silhouette_score'.
                                      This DataFrame needs to be generated and saved by the
                                      clustering step when K selection is enabled.
    """
    # Check if K selection was enabled in the first place
    if not config.get('CLUSTER_K_SELECTION_ENABLED', False):
        logger.info("K selection was not enabled in config. Skipping K selection plot.")
        return
    if k_metrics_df is None or k_metrics_df.empty:
        logger.warning("K selection metrics DataFrame is missing or empty. Skipping K selection plot. "
                       "Ensure clustering stage saves results when K selection is enabled.")
        return

    logging.info("Generating K selection plots...")
    plot_dir = os.path.join(config.get('SCENARIO_RESULTS_DIR', 'results/default'), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, 'k_selection_silhouette.png') # Assuming Silhouette

    metric_col = 'avg_silhouette_score' # The metric used to choose K
    k_col = 'k'

    if k_col not in k_metrics_df.columns or metric_col not in k_metrics_df.columns:
        logger.error(f"Required columns ('{k_col}', '{metric_col}') not found in k_metrics_df. Columns found: {k_metrics_df.columns}")
        return

    # Drop rows where metric might be NaN (e.g., silhouette undefined)
    k_metrics_df = k_metrics_df.dropna(subset=[metric_col])
    if k_metrics_df.empty:
        logger.warning("No valid metric scores found after dropping NaNs in k_metrics_df.")
        return

    try:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=k_metrics_df, x=k_col, y=metric_col, marker='o')

        # Find and highlight the best K based on the max score
        if not k_metrics_df.empty:
             best_k_row = k_metrics_df.loc[k_metrics_df[metric_col].idxmax()]
             best_k = best_k_row[k_col]
             best_score = best_k_row[metric_col]
             plt.axvline(best_k, color='r', linestyle='--', label=f'Best K = {int(best_k)} ({best_score:.3f})')
             plt.scatter([best_k], [best_score], color='red', s=100, zorder=5) # Highlight best point

        plt.title(f"K Selection using {metric_col.replace('_', ' ').title()} - {config.get('SCENARIO_NAME', '')}")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel(metric_col.replace('_', ' ').title())
        plt.grid(True, linestyle='--', alpha=0.6)
        # Ensure x-ticks cover the tested range
        plt.xticks(np.arange(k_metrics_df[k_col].min(), k_metrics_df[k_col].max() + 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        logger.info(f"Saved K selection plot to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate K selection plot: {e}")


# --- Standalone Test Block ---
if __name__ == '__main__':
    logger.info("--- Testing viz_metrics.py standalone ---")

    # Dummy Config
    config_test = {
        'SCENARIO_NAME': 'Metrics_Viz_Test',
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Metrics_Viz_Test'),
        'CLUSTER_K_SELECTION_ENABLED': True # Enable for K plot test
    }
    # Create dummy directories and files for testing
    results_dir = config_test['SCENARIO_RESULTS_DIR']
    os.makedirs(os.path.join(results_dir, 'plots', 'metric_timeseries'), exist_ok=True)

    # Dummy Averaged Metrics
    dummy_avg_metrics = {
        'avg_ari': 0.75, 'avg_nmi': 0.8, 'avg_churn_rate': 0.15,
        'avg_cluster_ssr': 0.6, 'avg_silhouette_score': 0.45,
        'avg_feeder_violation_rate': 0.1, 'avg_distance_violation_rate': 0.05,
        'avg_abs_cluster_net_import': 1500.0, 'total_net_import': 100.0
    }
    # Save dummy json for summary plot function to read (optional, can also pass dict directly)
    with open(os.path.join(results_dir, 'evaluation_metrics.json'), 'w') as f:
         json.dump(dummy_avg_metrics, f)

    # Dummy Time Series Metric Files
    ts_len = 20
    pd.DataFrame({
        'time_index_t': range(1, ts_len + 1),
        'ari': np.linspace(0.6, 0.8, ts_len) + np.random.rand(ts_len) * 0.05,
        'nmi': np.linspace(0.7, 0.85, ts_len) + np.random.rand(ts_len) * 0.05,
        'churn_rate': np.linspace(0.2, 0.1, ts_len) + np.random.rand(ts_len) * 0.02
    }).to_csv(os.path.join(results_dir, 'stability_metrics_timeseries.csv'), index=False)

    pd.DataFrame({
        'time_index': range(ts_len),
        'silhouette_score': np.linspace(0.3, 0.5, ts_len) + np.random.rand(ts_len) * 0.1 - 0.05
    }).to_csv(os.path.join(results_dir, 'quality_metrics_timeseries.csv'), index=False)

    synergy_data = []
    for t in range(ts_len):
        for c in range(4): # Assume 4 clusters
             synergy_data.append({'time_index':t, 'cluster_id':c, 'ssr': 0.5 + (c*0.1) + np.random.rand()*0.1, 'net_import': (c-1.5)*100 + np.random.randn()*50})
    pd.DataFrame(synergy_data).to_csv(os.path.join(results_dir, 'synergy_metrics_timeseries.csv'), index=False)

    # Dummy K Selection Results File/DataFrame
    dummy_k_results = pd.DataFrame({
        'k': range(3, 9),
        'avg_silhouette_score': [0.3, 0.45, 0.5, 0.48, 0.46, 0.4]
    })
    # Save dummy file (clustering script should do this eventually)
    dummy_k_results.to_csv(os.path.join(results_dir, 'k_selection_results.csv'), index=False)


    # --- Call plotting functions ---
    logger.info("--- Running Standalone Plot Function Tests ---")
    plot_evaluation_summary(config_test, dummy_avg_metrics)
    plot_metric_timeseries(config_test)
    # Load the k results df before passing
    try:
         k_res_df = pd.read_csv(os.path.join(results_dir, 'k_selection_results.csv'))
         plot_k_selection_metrics(config_test, k_res_df)
    except FileNotFoundError:
        logger.error("Standalone test: k_selection_results.csv not found.")
    except Exception as e:
         logger.error(f"Error plotting K selection: {e}")

    logger.info(f"--- Standalone test complete for viz_metrics.py. Check plots in {os.path.join(results_dir, 'plots')} ---")
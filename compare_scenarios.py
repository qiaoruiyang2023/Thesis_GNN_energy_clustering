# compare_scenarios.py

import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PYTHON_EXE = sys.executable # Use the same python executable that runs this script
MAIN_SCRIPT = 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/main.py'
CONFIG_DIR = 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/configs'
RESULTS_BASE_DIR = 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/results/' # Assumes results are saved here by main.py

# List of scenario config filenames to compare
SCENARIO_CONFIGS = [
    'scenario_A_config.yaml',
    'scenario_B_config.yaml',
    'scenario_C_config.yaml',
    'scenario_D_config.yaml',
]

# Metrics to extract and compare (must match keys in evaluation_metrics.json)
METRICS_TO_COMPARE = [
    'avg_ari',
    'avg_nmi',
    'avg_churn_rate',
    'avg_cluster_ssr',
    'avg_silhouette_score',
    'avg_feeder_violation_rate',
    'avg_distance_violation_rate',
    'avg_abs_cluster_net_import', # Added based on evaluation.py output
]

# --- Helper Function ---
def get_scenario_name_from_config(config_filename):
    """Extracts scenario name (used for results folder) from config filename."""
    # Basic extraction, assumes structure like 'scenario_X_config.yaml' -> 'Scenario_X_...'
    # A more robust way would be to load the yaml, but this avoids yaml dependency here.
    base = os.path.basename(config_filename)
    if base.startswith('scenario_A'): return 'Scenario_A_No_Constraint_KNN'
    if base.startswith('scenario_B'): return 'Scenario_B_Feeder_Constraint'
    if base.startswith('scenario_C'): return 'Scenario_C_Distance_Constraint'
    if base.startswith('scenario_D'): return 'Scenario_D_Combined_Constraint'
    logging.warning(f"Could not determine scenario name from {config_filename}, using filename base.")
    return os.path.splitext(base)[0] # Fallback

# --- Main Comparison Logic ---
def run_comparison():
    all_metrics_data = []
    scenario_names_processed = []

    # 1. Run main.py for each scenario
    logging.info("--- Running Pipeline for Each Scenario ---")
    for config_file in SCENARIO_CONFIGS:
        config_path = os.path.join(CONFIG_DIR, config_file)
        scenario_name = get_scenario_name_from_config(config_file)
        scenario_names_processed.append(scenario_name)
        logging.info(f"Running scenario: {scenario_name} (Config: {config_path})")

        if not os.path.exists(config_path):
            logging.error(f"Config file not found: {config_path}. Skipping scenario.")
            continue

        command = [PYTHON_EXE, MAIN_SCRIPT, '--config', config_path]
        try:
            # Run main.py as a subprocess
            # Check=True will raise CalledProcessError if main.py exits with non-zero code
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Finished {scenario_name}. Output:\n{process.stdout[-500:]}") # Log last part of output
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {scenario_name}. Return code: {e.returncode}")
            logging.error(f"Stderr:\n{e.stderr}")
            logging.error(f"Stdout:\n{e.stdout}")
            # Continue to next scenario or stop? For now, continue.
        except Exception as e:
            logging.error(f"An unexpected error occurred running {scenario_name}: {e}")

    # 2. Collect results
    logging.info("\n--- Collecting Results ---")
    for scenario_name in scenario_names_processed:
        metrics_path = os.path.join(RESULTS_BASE_DIR, scenario_name, 'evaluation_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                metrics['scenario'] = scenario_name # Add scenario name for grouping
                all_metrics_data.append(metrics)
                logging.info(f"Loaded metrics for {scenario_name}")
            except Exception as e:
                logging.error(f"Error reading metrics file {metrics_path}: {e}")
        else:
            logging.warning(f"Metrics file not found for scenario {scenario_name} at {metrics_path}")

    if not all_metrics_data:
        logging.error("No metrics data collected. Exiting comparison.")
        return

    # 3. Create DataFrame and Plots
    results_df = pd.DataFrame(all_metrics_data)
    results_df = results_df.set_index('scenario')

    # Select only the desired metrics for comparison table/plots
    comparison_df = results_df[METRICS_TO_COMPARE].copy()

    print("\n--- Comparison Results ---")
    print(comparison_df.to_string())

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comparison_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    comparison_csv_path = os.path.join(output_dir, 'scenario_comparison_metrics.csv')
    comparison_df.to_csv(comparison_csv_path)
    logging.info(f"Saved comparison metrics table to: {comparison_csv_path}")

    # Generate Plots (Example: Bar chart for each metric)
    try:
        num_metrics = len(METRICS_TO_COMPARE)
        # Adjust layout based on number of metrics
        ncols = 3
        nrows = (num_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten() # Flatten to 1D array for easy iteration

        for i, metric in enumerate(METRICS_TO_COMPARE):
            if metric in comparison_df.columns:
                sns.barplot(x=comparison_df.index, y=comparison_df[metric], ax=axes[i], palette="viridis")
                axes[i].set_title(metric)
                axes[i].set_xlabel("Scenario")
                axes[i].set_ylabel("Value")
                axes[i].tick_params(axis='x', rotation=30)
                # Add value labels
                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt='%.3f', fontsize=8)
            else:
                 axes[i].set_title(f"{metric}\n(Not Found)")
                 axes[i].axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'scenario_comparison_plots.png')
        plt.savefig(plot_path)
        plt.close(fig)
        logging.info(f"Saved comparison plots to: {plot_path}")

    except Exception as e:
        logging.error(f"Error generating comparison plots: {e}")

if __name__ == "__main__":
    run_comparison()
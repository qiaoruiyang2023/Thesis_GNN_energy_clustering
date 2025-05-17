# sensitivity_analysis.py

import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import yaml # Required for loading/modifying config
import shutil # For managing temporary config files/dirs
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PYTHON_EXE = sys.executable
MAIN_SCRIPT = 'main.py'
RESULTS_BASE_DIR = 'results'
TEMP_CONFIG_DIR = '_temp_configs_sensitivity' # Directory for temporary configs

# Parameters for Sensitivity Analysis
BASE_CONFIG_PATH = 'configs/scenario_C_config.yaml' # Base scenario to modify
PARAMETER_TO_VARY = 'DISTANCE_THRESHOLD_KM' # Key in the config file
PARAMETER_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0] # Values to test

# Base name for results folders and output files
ANALYSIS_NAME = f'Sensitivity_{PARAMETER_TO_VARY}'

# Metrics to extract and plot
METRICS_TO_PLOT = [
    'avg_ari',
    'avg_nmi',
    'avg_churn_rate',
    'avg_cluster_ssr',
    'avg_silhouette_score',
    'avg_distance_violation_rate', # This should change with the threshold
]

# --- Main Sensitivity Logic ---
def run_sensitivity():
    all_metrics_data = []
    run_details = [] # Store parameter value and corresponding results dir

    # Create temp config dir
    if os.path.exists(TEMP_CONFIG_DIR):
        shutil.rmtree(TEMP_CONFIG_DIR) # Clean up previous runs
    os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

    # 1. Load base config
    try:
        with open(BASE_CONFIG_PATH, 'r') as f:
            base_config_data = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load base config {BASE_CONFIG_PATH}: {e}")
        return

    logging.info(f"--- Running Sensitivity Analysis on '{PARAMETER_TO_VARY}' ---")
    logging.info(f"Base config: {BASE_CONFIG_PATH}")
    logging.info(f"Values to test: {PARAMETER_VALUES}")

    # 2. Loop through parameter values
    for value in PARAMETER_VALUES:
        logging.info(f"\n--- Testing {PARAMETER_TO_VARY} = {value} ---")
        current_config = base_config_data.copy() # Start fresh from base

        # Modify the parameter
        if PARAMETER_TO_VARY not in current_config:
             logging.error(f"Parameter '{PARAMETER_TO_VARY}' not found in base config. Skipping value {value}.")
             continue
        current_config[PARAMETER_TO_VARY] = value

        # Create a unique scenario name and results dir for this run
        # Ensure scenario name is filesystem-friendly
        value_str = str(value).replace('.', '_')
        scenario_name = f"{ANALYSIS_NAME}_{value_str}"
        current_config['SCENARIO_NAME'] = scenario_name
        # NOTE: config_loader.py adds PROJECT_ROOT, so only relative path needed here theoretically,
        # but main.py expects a file path argument. We save a temp file.
        temp_config_filename = f"temp_config_{scenario_name}.yaml"
        temp_config_path = os.path.join(TEMP_CONFIG_DIR, temp_config_filename)

        # Save the modified config to the temporary file
        try:
            with open(temp_config_path, 'w') as f:
                yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logging.error(f"Failed to save temporary config {temp_config_path}: {e}")
            continue

        # Run main.py with the temporary config
        command = [PYTHON_EXE, MAIN_SCRIPT, '--config', temp_config_path]
        try:
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Finished run for {PARAMETER_TO_VARY}={value}. Output tail:\n{process.stdout[-500:]}")
            # Store details for result collection
            results_dir_expected = os.path.join(RESULTS_BASE_DIR, scenario_name)
            run_details.append({'param_value': value, 'results_dir': results_dir_expected})
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running for {PARAMETER_TO_VARY}={value}. Return code: {e.returncode}")
            logging.error(f"Stderr:\n{e.stderr}")
            logging.error(f"Stdout:\n{e.stdout}")
        except Exception as e:
            logging.error(f"An unexpected error occurred running for {PARAMETER_TO_VARY}={value}: {e}")


    # 3. Collect results
    logging.info("\n--- Collecting Sensitivity Results ---")
    for detail in run_details:
        metrics_path = os.path.join(detail['results_dir'], 'evaluation_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                metrics[PARAMETER_TO_VARY] = detail['param_value'] # Add the parameter value
                all_metrics_data.append(metrics)
                logging.info(f"Loaded metrics for {PARAMETER_TO_VARY}={detail['param_value']}")
            except Exception as e:
                logging.error(f"Error reading metrics file {metrics_path}: {e}")
        else:
            logging.warning(f"Metrics file not found for run {PARAMETER_TO_VARY}={detail['param_value']} at {metrics_path}")

    # Cleanup temp config directory
    try:
        shutil.rmtree(TEMP_CONFIG_DIR)
        logging.info(f"Removed temporary config directory: {TEMP_CONFIG_DIR}")
    except Exception as e:
        logging.warning(f"Could not remove temporary config directory {TEMP_CONFIG_DIR}: {e}")

    if not all_metrics_data:
        logging.error("No metrics data collected for sensitivity analysis. Exiting.")
        return

    # 4. Create DataFrame and Plots
    results_df = pd.DataFrame(all_metrics_data)
    # Sort by the parameter value for plotting
    results_df = results_df.sort_values(by=PARAMETER_TO_VARY)

    print("\n--- Sensitivity Analysis Results ---")
    print(results_df[[PARAMETER_TO_VARY] + METRICS_TO_PLOT].to_string())

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{ANALYSIS_NAME}_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    sensitivity_csv_path = os.path.join(output_dir, f'{ANALYSIS_NAME}_metrics.csv')
    results_df.to_csv(sensitivity_csv_path, index=False)
    logging.info(f"Saved sensitivity metrics table to: {sensitivity_csv_path}")

    # Generate Plots (Line plot for each metric vs. parameter)
    try:
        num_metrics = len(METRICS_TO_PLOT)
        ncols = 3
        nrows = (num_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()

        for i, metric in enumerate(METRICS_TO_PLOT):
             if metric in results_df.columns:
                 sns.lineplot(x=PARAMETER_TO_VARY, y=metric, data=results_df, marker='o', ax=axes[i])
                 axes[i].set_title(f'{metric} vs {PARAMETER_TO_VARY}')
                 axes[i].set_xlabel(PARAMETER_TO_VARY)
                 axes[i].set_ylabel(metric)
                 axes[i].grid(True, linestyle='--', alpha=0.6)
             else:
                 axes[i].set_title(f"{metric}\n(Not Found)")
                 axes[i].axis('off')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        fig.suptitle(f'Sensitivity Analysis: Metrics vs {PARAMETER_TO_VARY}', fontsize=16)
        plot_path = os.path.join(output_dir, f'{ANALYSIS_NAME}_plots.png')
        plt.savefig(plot_path)
        plt.close(fig)
        logging.info(f"Saved sensitivity plots to: {plot_path}")

    except Exception as e:
        logging.error(f"Error generating sensitivity plots: {e}")


if __name__ == "__main__":
    run_sensitivity()
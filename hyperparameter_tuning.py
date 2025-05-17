# hyperparameter_tuning.py

import optuna
import os
import sys
import logging
import yaml
import torch
import random
import numpy as np
import time
import shutil
from datetime import datetime

# --- Add src to path ---
# This allows importing modules from src when running this script from the project root
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import pipeline components ---
try:
    from config_loader import load_config # Need base config structure
    from data_processing import process_data
    from graph_construction import build_graph
    from datasets import SpatioTemporalGraphDataset
    from models import SpatioTemporalGNNEmbedder
    from training import train_model # We'll call train_model directly
    from inference import generate_embeddings
    from clustering import perform_temporal_clustering
    from evaluation import run_evaluation # Need this to get the metric
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Ensure this script is run from the project root directory and src is accessible.")
    sys.exit(1)

# --- Logging ---
# Optuna also uses logging, configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_CONFIG_PATH = 'configs/scenario_B_config.yaml' # Choose a base scenario for tuning
N_TRIALS = 20 # Number of tuning trials to run
METRIC_TO_OPTIMIZE = 'avg_silhouette_score' # Metric from evaluation_metrics.json
OPTIMIZE_DIRECTION = 'maximize' # 'maximize' or 'minimize'
STUDY_NAME = f'HyperOpt_{os.path.splitext(os.path.basename(BASE_CONFIG_PATH))[0]}'
STORAGE_DB_URL = f'sqlite:///{STUDY_NAME}.db' # Store results in a database file

# --- Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna to optimize."""
    run_start_time = time.time()
    logger.info(f"\n--- Starting Trial {trial.number} ---")

    # --- Load Base Config ---
    # Use load_config to get derived paths correctly, but we'll override parts
    # Pass a dummy filename to load_config, we only need the base structure
    try:
        # Load the *actual* base config to get defaults
        with open('configs/base_config.yaml', 'r') as f:
             base_cfg_data = yaml.safe_load(f)
        # Load the chosen scenario config to get scenario specifics
        with open(BASE_CONFIG_PATH, 'r') as f:
             scenario_cfg_data = yaml.safe_load(f)
        # Merge them manually (simplified version of config_loader's update_dict)
        config = base_cfg_data.copy()
        config.update(scenario_cfg_data)

    except Exception as e:
        logger.error(f"Error loading base/scenario config: {e}")
        # Returning a high value for minimization or low for maximization signals failure
        return float('inf') if OPTIMIZE_DIRECTION == 'minimize' else float('-inf')

    # --- Suggest Hyperparameters ---
    config['LEARNING_RATE'] = trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True)
    config['WEIGHT_DECAY'] = trial.suggest_float('WEIGHT_DECAY', 1e-6, 1e-3, log=True)
    config['CONTRASTIVE_MARGIN'] = trial.suggest_float('CONTRASTIVE_MARGIN', 0.5, 2.0)
    config['EMBEDDING_DIM'] = trial.suggest_categorical('EMBEDDING_DIM', [32, 64, 128])
    config['GNN_HIDDEN_DIM'] = trial.suggest_categorical('GNN_HIDDEN_DIM', [32, 64, 128])
    config['LSTM_HIDDEN_DIM'] = trial.suggest_categorical('LSTM_HIDDEN_DIM', [32, 64, 128])
    config['GNN_LAYERS'] = trial.suggest_int('GNN_LAYERS', 1, 3)
    # config['GNN_HEADS'] = trial.suggest_categorical('GNN_HEADS', [2, 4, 8]) # Keep fixed or tune carefully
    # config['LSTM_LAYERS'] = trial.suggest_int('LSTM_LAYERS', 1, 2) # Keep fixed or tune carefully
    config['DROPOUT_RATE'] = trial.suggest_float('DROPOUT_RATE', 0.1, 0.5)
    # config['EMBEDDING_SMOOTHING_WINDOW_W'] = trial.suggest_int('EMBEDDING_SMOOTHING_WINDOW_W', 1, 8) # Optional to tune

    # Use fewer epochs during tuning for speed, override config
    config['EPOCHS'] = 15 # Or make this tunable: trial.suggest_int('EPOCHS', 10, 30)

    # --- Setup Unique Output Dir for Trial ---
    # Necessary to avoid clashes when running pipeline components that save files
    trial_scenario_name = f"{STUDY_NAME}_trial_{trial.number}"
    config['SCENARIO_NAME'] = trial_scenario_name
    config['RESULTS_DIR'] = os.path.join(RESULTS_BASE_DIR, STUDY_NAME) # Group trial results
    config['SCENARIO_RESULTS_DIR'] = os.path.join(config['RESULTS_DIR'], trial_scenario_name)
    # Need PROJECT_ROOT for derived paths inside pipeline components
    config['PROJECT_ROOT'] = project_root
    for key in ['RAW_DATA_DIR', 'PROCESSED_DATA_DIR']:
         if key in config:
              config[key] = os.path.join(project_root, config[key]) # Ensure paths are absolute

    # Create output dirs for this trial
    os.makedirs(config['SCENARIO_RESULTS_DIR'], exist_ok=True)
    os.makedirs(os.path.join(config['SCENARIO_RESULTS_DIR'], 'plots'), exist_ok=True)
    os.makedirs(os.path.join(config['SCENARIO_RESULTS_DIR'], 'model'), exist_ok=True)

    # Log the config for this trial
    logger.info(f"Trial {trial.number} Config: {config}")

    # --- Set Seed for Reproducibility within Trial (Optional) ---
    seed = config.get('SEED', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # Can slow down training
        # torch.backends.cudnn.benchmark = False

    # --- Execute Pipeline Steps ---
    try:
        # Data processing and graph construction usually only need to be done once
        # if the base scenario's processed data exists. We assume it does here.
        # If not, uncomment these lines (but makes tuning much slower):
        # logger.info(f"[Trial {trial.number}] Running Data Processing...")
        # _, _ = process_data(config)
        # logger.info(f"[Trial {trial.number}] Running Graph Construction...")
        # _ = build_graph(config) # Assumes graph depends only on base scenario, not tuned params

        logger.info(f"[Trial {trial.number}] Initializing Dataset...")
        # Ensure dataset uses the SCENARIO_NAME from the *base* config for loading graph/data
        # We pass the *tuned* config for model/training params later
        base_scenario_config = config.copy() # Create a copy
        base_scenario_config['SCENARIO_NAME'] = scenario_cfg_data.get('SCENARIO_NAME', os.path.splitext(os.path.basename(BASE_CONFIG_PATH))[0])
        dataset = SpatioTemporalGraphDataset(base_scenario_config) # Use base name to load graph

        logger.info(f"[Trial {trial.number}] Initializing Model...")
        static_dim = dataset.x_static.shape[1]
        dynamic_dim = dataset.dynamic_features.shape[2]
        model = SpatioTemporalGNNEmbedder(config, static_feature_dim=static_dim, dynamic_feature_dim=dynamic_dim)

        logger.info(f"[Trial {trial.number}] Training Model...")
        training_losses = train_model(config, model, dataset) # train_model saves the model

        logger.info(f"[Trial {trial.number}] Generating Embeddings...")
        # Re-init model architecture, weights loaded inside generate_embeddings
        inference_model = SpatioTemporalGNNEmbedder(config, static_feature_dim=static_dim, dynamic_feature_dim=dynamic_dim)
        final_embeddings = generate_embeddings(config, inference_model, dataset)

        logger.info(f"[Trial {trial.number}] Performing Clustering...")
        cluster_assignments = perform_temporal_clustering(config, final_embeddings, dataset) # Saves assignments

        logger.info(f"[Trial {trial.number}] Running Evaluation...")
        # Pass the actual dataset object used (based on base config for loading)
        evaluation_metrics = run_evaluation(config, cluster_assignments, final_embeddings, dataset) # Saves metrics

        # Check if the metric exists and is valid
        metric_value = evaluation_metrics.get(METRIC_TO_OPTIMIZE)
        if metric_value is None or np.isnan(metric_value):
            logger.warning(f"Trial {trial.number}: Metric '{METRIC_TO_OPTIMIZE}' not found or NaN in results. Returning failure value.")
            # Return a poor value to discourage Optuna from this direction
            return float('inf') if OPTIMIZE_DIRECTION == 'minimize' else float('-inf')

        logger.info(f"Trial {trial.number} finished. {METRIC_TO_OPTIMIZE}: {metric_value:.6f}. Duration: {time.time() - run_start_time:.2f}s")
        return metric_value

    except Exception as e:
        logger.exception(f"Trial {trial.number} failed: {e}")
        # Return a poor value if the trial crashes
        return float('inf') if OPTIMIZE_DIRECTION == 'minimize' else float('-inf')

# --- Main Tuning Execution ---
if __name__ == "__main__":
    logger.info(f"--- Starting Hyperparameter Tuning ---")
    logger.info(f"Study Name: {STUDY_NAME}")
    logger.info(f"Storage URL: {STORAGE_DB_URL}")
    logger.info(f"Metric: {METRIC_TO_OPTIMIZE} ({OPTIMIZE_DIRECTION})")
    logger.info(f"Number of Trials: {N_TRIALS}")

    # Create or load the study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB_URL,
        direction=OPTIMIZE_DIRECTION,
        load_if_exists=True # Continue previous study if DB exists
    )

    # Run the optimization
    study.optimize(objective, n_trials=N_TRIALS, timeout=None) # No timeout

    # --- Report Results ---
    logger.info("\n--- Tuning Complete ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best value ({METRIC_TO_OPTIMIZE}): {best_trial.value}")
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    # Optional: Save study results DataFrame
    try:
        results_df = study.trials_dataframe()
        results_df_path = os.path.join(RESULTS_BASE_DIR, STUDY_NAME, f'{STUDY_NAME}_trials.csv')
        os.makedirs(os.path.dirname(results_df_path), exist_ok=True)
        results_df.to_csv(results_df_path, index=False)
        logger.info(f"Saved study trials dataframe to: {results_df_path}")
    except Exception as e:
        logger.error(f"Could not save study trials dataframe: {e}")

    # Optional: Generate Optuna plots (requires matplotlib installed)
    try:
        if optuna.visualization.is_available():
            plots_dir = os.path.join(RESULTS_BASE_DIR, STUDY_NAME, 'optuna_plots')
            os.makedirs(plots_dir, exist_ok=True)

            fig_hist = optuna.visualization.plot_optimization_history(study)
            fig_hist.write_image(os.path.join(plots_dir, 'optimization_history.png'))

            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.write_image(os.path.join(plots_dir, 'slice.png'))

            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_importance.write_image(os.path.join(plots_dir, 'param_importances.png'))

            logger.info(f"Saved Optuna visualization plots to: {plots_dir}")
        else:
            logger.info("Optuna visualization is not available. Install plotly for plots.")
    except Exception as e:
        logger.error(f"Could not generate Optuna plots: {e}")
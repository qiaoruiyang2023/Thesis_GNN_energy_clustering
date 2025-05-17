# main.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
import time
import torch
import numpy as np
import random
import sys
import yaml # For potentially logging full config
import matplotlib.pyplot as plt # For plotting training loss
from datetime import datetime # <-- ADD THIS LINE
import copy

# Ensure the src directory is in the Python path
# This allows importing modules from src when running main.py from the project root
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import pipeline components from src modules
try:
    from config_loader import load_config
    from data_processing import process_data
    from graph_construction import build_graph
    from datasets import SpatioTemporalGraphDataset
    from models import SpatioTemporalGNNEmbedder
    # loss_fn is used within training.py
    from training import train_model # train_model now returns losses
    from inference import generate_embeddings
    from clustering import perform_temporal_clustering
    from evaluation import run_evaluation
    from visualization import run_visualization # Main orchestrator for plots
except ImportError as e:
    print(f"Error importing modules from src: {e}")
    print(f"Ensure you are running main.py from the project root directory: {project_root}")
    print(f"Ensure src path ({src_path}) is accessible.")
    print(f"PYTHONPATH: {sys.path}")
    sys.exit(1)


# --- Logging Setup ---
# Configure root logger for application-wide logging
# Add FileHandler to save logs
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger() # Get root logger
logger.setLevel(logging.INFO) # Set minimum level

# Clear existing handlers (useful if re-running in interactive session)
# for handler in logger.handlers[:]:
#    logger.removeHandler(handler)

# Console Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(stream_handler)

# File Handler (log to main project directory for now)
# file_handler = logging.FileHandler('pipeline.log', mode='a') # Append mode
# file_handler.setFormatter(log_formatter)
# if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
#    logger.addHandler(file_handler)

# Get logger for this specific module
module_logger = logging.getLogger(__name__)

# --- Random Seed Function ---
def set_seed(seed_value):
    """Sets random seed for reproducibility across libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU
        # Optional: Configure cuDNN for reproducibility, may impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    module_logger.info(f"Random seed set to: {seed_value}")

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Spatio-Temporal GNN Clustering Pipeline.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the scenario configuration file (e.g., 'configs/scenario_B_config.yaml')."
    )
    # Add --skip-train argument? Or --mode? Keep simple for now.
    # parser.add_argument('--mode', type=str, default='full', choices=['full', 'process', 'train', 'eval'], help="Pipeline execution mode.")
    return parser.parse_args()

# --- Helper: Plot Training Loss ---
def plot_training_loss(config: dict, train_losses: list, test_losses: list):
    """
    Plot training and test loss curves for each epoch.

    Args:
        config (dict): Configuration dictionary
        train_losses (list): List of training losses
        test_losses (list): List of test losses
    """
    if not train_losses or not test_losses:
        module_logger.warning("No loss data provided, skipping plot.")
        return

    plot_dir = os.path.join(config.get('SCENARIO_RESULTS_DIR', 'results/default'), 'plots')
    os.makedirs(plot_dir, exist_ok=True)  # Ensure plots directory exists
    save_path = os.path.join(plot_dir, 'training_loss_curve.png')

    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Plot training and test losses
        plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Training Loss', color='blue')
        plt.plot(epochs, test_losses, marker='s', linestyle='--', label='Test Loss', color='red')
        
        plt.title(f"Training and Test Loss Curves - {config.get('SCENARIO_NAME', 'Unknown')}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=100)
        plt.close()
        module_logger.info(f"Saved loss curve plot to: {save_path}")
    except Exception as e:
        module_logger.error(f"Failed to plot loss curves: {e}")


# --- Main Pipeline ---
def main():
    """Main function to orchestrate the pipeline execution."""
    pipeline_start_time = time.time()
    module_logger.info("=================================================")
    module_logger.info("=== Starting Spatio-Temporal Clustering Pipeline ===")
    module_logger.info("=================================================")

    # --- Configuration ---
    args = parse_args()
    try:
        # Ensure config path uses correct separators for OS
        config_path_norm = os.path.normpath(args.config)
        config = load_config(os.path.basename(config_path_norm), config_dir=os.path.dirname(config_path_norm))
        scenario_name = config.get('SCENARIO_NAME', f'UnknownScenario_{datetime.now():%Y%m%d%H%M%S}')
        module_logger.info(f"Loaded configuration for: {scenario_name}")

        # Log config details (optional, can be very verbose)
        # module_logger.debug(f"Full Configuration:\n{yaml.dump(config, default_flow_style=False)}")

        # Add/Update file handler to log to the specific scenario results directory
        results_dir = config.get('SCENARIO_RESULTS_DIR', os.path.join('results', scenario_name))
        os.makedirs(results_dir, exist_ok=True) # Ensure dir exists
        log_file_path = os.path.join(results_dir, 'pipeline_run.log')
        file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
        file_handler.setFormatter(log_formatter)
        # Remove existing file handlers before adding new one to avoid duplicate logs
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        logger.addHandler(file_handler)
        module_logger.info(f"Logging detailed output to: {log_file_path}")

        test_config = copy.deepcopy(config)
        train_config = copy.deepcopy(config)
        model_dir = os.path.join(config['SCENARIO_RESULTS_DIR'], 'model')

    except Exception as e:
        module_logger.exception(f"Error loading configuration file {args.config}: {e}")
        sys.exit(1)

    # --- Set Seed ---
    # Use seed from config, default to 42 if not present
    set_seed(config.get('SEED', 42))

    # --- Execute Pipeline Stages ---
    final_embeddings = None # Initialize variables to None
    cluster_assignments = None
    evaluation_metrics = None
    dataset_obj = None
    training_epoch_losses = []

    try:
        # --- 1. Data Processing ---
        stage_start = time.time()
        module_logger.info("--- Stage 1: Data Processing ---")
        module_logger.info("Starting data processing and train-test split...")
        processed_data = process_data(config)
        train_data = processed_data['train']
        test_data = processed_data['test']
        train_static_df, train_dynamic_df = train_data
        test_static_df, test_dynamic_df = test_data
        print("Train buildings:", train_static_df.index[:5])
        print("Test buildings:", test_static_df.index[:5])
        print("Train dynamic df shape:", train_dynamic_df.shape)
        print("Test dynamic df shape:", test_dynamic_df.shape)
        module_logger.info(f"Train set size: {len(train_static_df)} buildings")
        module_logger.info(f"Test set size: {len(test_static_df)} buildings")
        
        # 创建训练和测试配置
        train_config = copy.deepcopy(config)
        test_config = copy.deepcopy(config)
        
        # 设置训练和测试数据目录
        train_config['PROCESSED_DATA_DIR'] = os.path.join(config['PROCESSED_DATA_DIR'], 'train')
        test_config['PROCESSED_DATA_DIR'] = os.path.join(config['PROCESSED_DATA_DIR'], 'test')
        
        # 设置训练和测试结果目录
        train_config['SCENARIO_RESULTS_DIR'] = os.path.join(config['SCENARIO_RESULTS_DIR'], 'train')
        test_config['SCENARIO_RESULTS_DIR'] = os.path.join(config['SCENARIO_RESULTS_DIR'], 'test')
        
        # 设置数据集模式
        train_config['MODE'] = 'train'
        test_config['MODE'] = 'test'
        
        # 创建结果目录
        os.makedirs(train_config['SCENARIO_RESULTS_DIR'], exist_ok=True)
        os.makedirs(test_config['SCENARIO_RESULTS_DIR'], exist_ok=True)

        module_logger.info(f"--- Stage 1: Data Processing Finished (Duration: {time.time() - stage_start:.2f}s) ---")

        # --- 2. Graph Construction ---
        stage_start = time.time()
        module_logger.info("--- Stage 2: Graph Construction ---")
        train_graph = build_graph(train_config)
        test_graph = build_graph(test_config)
        module_logger.info(f"--- Stage 2: Graph Construction Finished (Duration: {time.time() - stage_start:.2f}s) ---")

        # --- 3. Dataset Initialization ---
        stage_start = time.time()
        module_logger.info("--- Stage 3: Dataset Initialization ---")
        train_dataset = SpatioTemporalGraphDataset(train_config)
        test_dataset = SpatioTemporalGraphDataset(test_config)
        module_logger.info(f"--- Stage 3: Dataset Initialization Finished (Duration: {time.time() - stage_start:.2f}s) ---")

        # --- 4. Model Initialization ---
        stage_start = time.time()
        module_logger.info("--- Stage 4: Model Initialization ---")
        static_dim = train_dataset.x_static.shape[1]
        dynamic_dim = train_dataset.dynamic_features.shape[2]
        model = SpatioTemporalGNNEmbedder(config, static_feature_dim=static_dim, dynamic_feature_dim=dynamic_dim)
        module_logger.info(f"Initialized model '{config.get('MODEL_NAME', 'N/A')}'")
        # module_logger.debug(model) # Log full model architecture if needed
        module_logger.info(f"--- Stage 4: Model Initialization Finished (Duration: {time.time() - stage_start:.2f}s) ---")

        # --- 5. Model Training ---
        stage_start = time.time()
        module_logger.info("--- Stage 5: Model Training ---")
        training_losses, test_losses = train_model(train_config, model, train_dataset)
        module_logger.info(f"--- Stage 5: Model Training Finished (Duration: {time.time() - stage_start:.2f}s) ---")

        # --- 6. Plot Training Loss ---
        plot_training_loss(config, training_losses, test_losses)

        # --- 7. Inference ---
        stage_start = time.time()
        module_logger.info("--- Stage 6: Inference ---")
        
        # 训练集推理
        train_embeddings = generate_embeddings(train_config, model, train_dataset)
        train_clusters = perform_temporal_clustering(train_config, train_embeddings, train_dataset)
        train_metrics = run_evaluation(train_config, train_clusters, train_embeddings, train_dataset)
        
        # 测试集推理
        test_embeddings = generate_embeddings(test_config, model, test_dataset)
        test_clusters = perform_temporal_clustering(test_config, test_embeddings, test_dataset)
        test_metrics = run_evaluation(test_config, test_clusters, test_embeddings, test_dataset)
        
        module_logger.info(f"--- Stage 6: Inference Finished (Duration: {time.time() - stage_start:.2f}s) ---")

        # --- 8. Visualization ---
        stage_start = time.time()
        module_logger.info("--- Stage 7: Visualization ---")
        
        # 训练集可视化
        run_visualization(train_config, train_clusters, train_embeddings, train_metrics, train_dataset)
        
        # 测试集可视化
        run_visualization(test_config, test_clusters, test_embeddings, test_metrics, test_dataset)
        
        module_logger.info(f"--- Stage 7: Visualization Finished (Duration: {time.time() - stage_start:.2f}s) ---")

    except FileNotFoundError as e:
        module_logger.exception(f"Pipeline halted: A required file was not found. "
                            f"Ensure previous steps generated necessary outputs or paths in config are correct. Error: {e}")
        sys.exit(1)
    except ValueError as e:
         module_logger.exception(f"Pipeline halted: A value error occurred, possibly due to configuration or data issues. Error: {e}")
         sys.exit(1)
    except ImportError as e:
         module_logger.exception(f"Pipeline halted: An import error occurred. Ensure all dependencies are installed. Error: {e}")
         sys.exit(1)
    except Exception as e: # Catch any other unexpected error
        module_logger.exception(f"An unexpected error occurred during pipeline execution: {e}")
        sys.exit(1)

    # --- Pipeline Completion ---
    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    module_logger.info("=================================================")
    module_logger.info(f"=== Pipeline Execution Completed Successfully ===")
    module_logger.info(f"=== Scenario: {config.get('SCENARIO_NAME', 'Unknown Scenario')}")
    module_logger.info(f"=== Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    module_logger.info(f"=== Results saved in: {config.get('SCENARIO_RESULTS_DIR')}")
    module_logger.info("=================================================")


if __name__ == "__main__":
    main()
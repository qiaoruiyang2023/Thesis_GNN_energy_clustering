# src/inference.py

import torch
# from torch.utils.data import DataLoader # <-- REMOVE Standard DataLoader
from torch_geometric.loader import DataLoader # <-- ADD PyG DataLoader
import os
import logging
import time
import numpy as np
from tqdm import tqdm # For progress bar

# Import from our source modules
# Ensure these paths are correct relative to where this script is or adjust sys.path
try:
    from models import SpatioTemporalGNNEmbedder # Assuming models.py is in the same directory or src/
    from datasets import SpatioTemporalGraphDataset # Assuming datasets.py is in the same directory or src/
except ImportError as e:
     # Basic logging might not be configured yet if run standalone very early
     print(f"[ERROR] Error importing project modules in inference.py: {e}")
     print("[ERROR] Ensure inference.py is run in an environment where 'models', 'datasets' are accessible.")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_embeddings(config: dict, model: SpatioTemporalGNNEmbedder, dataset: SpatioTemporalGraphDataset) -> torch.Tensor:
    """
    Generates node embeddings for the entire time series using a trained model.

    Args:
        config (dict): Configuration dictionary.
        model (SpatioTemporalGNNEmbedder): Initialized model instance (weights should be loaded).
        dataset (SpatioTemporalGraphDataset): Initialized dataset instance.

    Returns:
        torch.Tensor: A tensor containing node embeddings for all nodes over all
                      original time steps. Shape: [num_nodes, num_timesteps, embedding_dim]
    """
    logging.info(f"--- Starting Embedding Generation for Scenario: {config.get('SCENARIO_NAME', 'Unknown')} ---")
    start_time = time.time()

    # --- Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        logging.info(f"CUDA is available. Running inference on GPU {gpu_index}: {gpu_name}.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Running inference on CPU.")

    # --- Load Trained Model Weights ---
    # 确保使用训练模型的路径
    model_save_dir = os.path.join(config['SCENARIO_RESULTS_DIR'].replace('\\test', '\\train'), 'model')
    model_load_path = os.path.join(model_save_dir, 'final_model.pt')

    if not os.path.exists(model_load_path):
        logging.error(f"Trained model file not found at: {model_load_path}. Run training first.")
        raise FileNotFoundError(f"Model file not found: {model_load_path}")

    try:
        # Load state dict onto the correct device directly
        # Set weights_only=True for security unless you trust the source implicitly
        model.load_state_dict(torch.load(model_load_path, map_location=device, weights_only=True))
        logging.info(f"Loaded trained model weights from: {model_load_path}")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        raise

    model.to(device) # Ensure model is on the correct device after loading weights
    model.eval() # Set model to evaluation mode (important!)

    # --- Setup PyG DataLoader ---
    # Use batch_size=1 for full graph processing. Shuffle=False for sequential order.
    # The import at the top ensures this uses torch_geometric.loader.DataLoader
    num_workers = config.get('DATALOADER_WORKERS', 0)
    dataloader = DataLoader( # This is now PyG DataLoader
        dataset,
        batch_size=1, # Process one sequence start index at a time
        shuffle=False, # Important for reconstructing time series in order
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    logging.info(f"PyG DataLoader created for inference (shuffle=False, num_workers={num_workers}, pin_memory={dataloader.pin_memory}).")

    # --- Inference Loop & Reconstruction ---
    num_nodes = dataset.num_nodes
    num_timesteps = dataset.num_timesteps
    seq_len = dataset.seq_len
    embedding_dim = config.get('EMBEDDING_DIM') # Get from config
    if embedding_dim is None:
        raise ValueError("EMBEDDING_DIM not found in config.")


    # Initialize tensors to store summed embeddings and counts for averaging overlaps
    # Place these on CPU initially to avoid potential GPU memory overflow if num_timesteps is large
    all_embeddings_sum = torch.zeros((num_nodes, num_timesteps, embedding_dim), dtype=torch.float32, device='cpu')
    all_embeddings_counts = torch.zeros((num_nodes, num_timesteps, 1), dtype=torch.float32, device='cpu')

    logging.info(f"Reconstructing full embedding tensor ({num_nodes} nodes, {num_timesteps} timesteps, {embedding_dim} dim)...")

    with torch.no_grad(): # Disable gradient calculations for inference
        pbar = tqdm(dataloader, desc="Generating Embeddings", leave=True, unit="sequence")
        for i, batch in enumerate(pbar): # This loop should now work
            # batch contains data for sequence starting at time index i
            start_time_idx = i # Because shuffle=False and batch_size=1
            end_time_idx = start_time_idx + seq_len

            try:
                batch = batch.to(device) # Move data to device
            except Exception as e_move:
                logging.error(f"Error moving batch {i} to device: {e_move}. Skipping.")
                continue

            try:
                # Perform forward pass
                # Output shape: [num_nodes, seq_len, embedding_dim]
                output_embeddings_seq = model(batch)

                # Add results to the sum and increment counts for the corresponding time window
                # Move result to CPU before adding to avoid GPU memory issues if reconstruction tensor is large
                all_embeddings_sum[:, start_time_idx:end_time_idx, :] += output_embeddings_seq.detach().cpu()
                all_embeddings_counts[:, start_time_idx:end_time_idx, :] += 1.0

            except Exception as e_infer:
                 logging.error(f"Error during inference forward pass for batch {i}: {e_infer}. Skipping.")
                 continue


    # Calculate the average embeddings for overlapping windows
    # Avoid division by zero for time steps that might not have been covered
    all_embeddings_counts[all_embeddings_counts == 0] = 1.0 # Prevent division by zero
    final_embeddings = all_embeddings_sum / all_embeddings_counts

    end_time = time.time()
    inference_duration = end_time - start_time
    logging.info(f"--- Embedding Generation Finished. Duration: {inference_duration:.2f} seconds ---")
    logging.info(f"Final embeddings tensor shape: {final_embeddings.shape}")

    # Optional: Save Embeddings here if needed, e.g., for debugging or caching
    # save_path = os.path.join(config['SCENARIO_RESULTS_DIR'], 'final_embeddings.pt')
    # torch.save(final_embeddings, save_path)
    # logging.info(f"Saved final embeddings to: {save_path}")

    return final_embeddings


# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    # This block requires successful execution of previous steps and saved files
    logging.info("--- Testing inference script standalone ---")

     # --- CUDA Check ---
    if torch.cuda.is_available():
        print(f"Is CUDA available? {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available.")


    # 1. Load Config (replace with actual loader or use dummy)
    config_test = {
        'SCENARIO_NAME': 'Scenario_Standalone_Test', # Use a consistent name
        'PROCESSED_DATA_DIR': os.path.join('..', 'data', 'processed'), # Adjust relative path
        'RESULTS_DIR': os.path.join('..', 'results'),
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Scenario_Standalone_Test'),# Adjust
        'MODEL_NAME': 'GATLSTMEmbedder',
        'INPUT_SEQ_LEN': 12, # Must match dataset used for training
        'EMBEDDING_DIM': 16, # Must match trained model
        'DATALOADER_WORKERS': 0,
        # Add other params needed by model/dataset init if not loaded from graph/config
        'LSTM_HIDDEN_DIM': 32, 'LSTM_LAYERS': 1, 'GNN_HIDDEN_DIM': 32,
        'GNN_LAYERS': 1, 'GNN_HEADS': 2, 'DROPOUT_RATE': 0.1,
        'SEED': 42
    }
    # Ensure results dir exists for model loading
    os.makedirs(config_test['SCENARIO_RESULTS_DIR'], exist_ok=True)
    os.makedirs(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'model'), exist_ok=True)

    try:
        # 2. Initialize Dataset (needs processed files)
        # Attempt to load data for the test scenario or fallback
        dataset_scenario = config_test['SCENARIO_NAME']
        graph_path = os.path.join(config_test['PROCESSED_DATA_DIR'], f"graph_{dataset_scenario}.pt")
        dyn_path = os.path.join(config_test['PROCESSED_DATA_DIR'], 'processed_dynamic_features.parquet')

        if not os.path.exists(graph_path) or not os.path.exists(dyn_path):
             logging.warning(f"Data for {dataset_scenario} not found. Trying 'Scenario_B_Feeder_Constraint' data for standalone test.")
             config_test['SCENARIO_NAME'] = 'Scenario_B_Feeder_Constraint' # Fallback for test
             graph_path = os.path.join(config_test['PROCESSED_DATA_DIR'], f"graph_{config_test['SCENARIO_NAME']}.pt")
             if not os.path.exists(graph_path) or not os.path.exists(dyn_path):
                  raise FileNotFoundError(f"Required processed data not found for fallback scenario either: {graph_path} or {dyn_path}")

        logging.info(f"--- Standalone: Initializing Dataset using config for {config_test['SCENARIO_NAME']} ---")
        dataset = SpatioTemporalGraphDataset(config_test)

        # 3. Initialize Model architecture (weights will be loaded inside generate_embeddings)
        static_dim = dataset.x_static.shape[1]
        dynamic_dim = dataset.dynamic_features.shape[2]
        # Create model instance but don't load weights here
        model_instance = SpatioTemporalGNNEmbedder(config_test, static_feature_dim=static_dim, dynamic_feature_dim=dynamic_dim)
        logging.info(f"--- Standalone: Model architecture initialized ---")

        # 4. Generate Embeddings (needs trained model file)
        model_path = os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'model', 'final_model.pt')
        # Create a dummy trained model if it doesn't exist FOR TESTING ONLY
        if not os.path.exists(model_path):
            logging.warning(f"Trained model not found at {model_path}. Creating a dummy saved model for testing.")
            # Save the freshly initialized model state - this won't give meaningful embeddings
            torch.save(model_instance.state_dict(), model_path)


        logging.info(f"--- Standalone: Starting Embedding Generation ---")
        final_embeddings = generate_embeddings(config_test, model_instance, dataset) # Pass the initialized model

        print("\n--- Standalone Inference Test ---")
        print(f"Generated embeddings shape: {final_embeddings.shape}")
        # Check shape consistency
        expected_shape = (dataset.num_nodes, dataset.num_timesteps, config_test['EMBEDDING_DIM'])
        if final_embeddings.shape == expected_shape:
             print("Inference test successful! Output shape is correct.")
        else:
             print(f"Error: Output shape mismatch! Expected {expected_shape}, Got {final_embeddings.shape}")


    except FileNotFoundError as e:
        print(f"\nError: Prerequisite file not found. Ensure previous steps (data processing, graph construction, training) ran successfully. Details: {e}")
    except Exception as e:
        logging.exception(f"An error occurred during inference standalone test: {e}")
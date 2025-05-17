# src/clustering.py
import torch
import numpy as np
import pandas as pd
import os
import logging
import time
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score # <-- IMPORT ADDED
from tqdm import tqdm

# Import from other modules (ensure paths are correct)
try:
    from src.datasets import SpatioTemporalGraphDataset # Used only for type hinting and standalone test
except ImportError:
    # Fallback for standalone execution if src isn't in path easily
    SpatioTemporalGraphDataset = None # Assign None or a dummy class


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def smooth_embeddings(embeddings_tensor: torch.Tensor, window_size: int) -> np.ndarray:
    """
    Applies a moving average filter over the time dimension of node embeddings.

    Args:
        embeddings_tensor (torch.Tensor): Embeddings [nodes, timesteps, dim].
        window_size (int): Size of the moving average window.

    Returns:
        np.ndarray: Smoothed embeddings [nodes, timesteps, dim].
    """
    if not isinstance(embeddings_tensor, torch.Tensor):
        raise TypeError("Input embeddings_tensor must be a PyTorch Tensor.")
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    if window_size == 1:
        logging.info("Window size is 1, skipping embedding smoothing.")
        # Ensure output is numpy and on CPU
        return embeddings_tensor.detach().cpu().numpy()

    logging.info(f"Applying moving average smoothing with window size {window_size}...")
    start_time = time.time()

    # Move tensor to CPU and convert to numpy for pandas rolling operation
    embeddings_np = embeddings_tensor.detach().cpu().numpy()
    num_nodes, num_timesteps, embedding_dim = embeddings_np.shape
    smoothed_embeddings_np = np.zeros_like(embeddings_np)

    pbar_smooth = tqdm(total=num_nodes, desc="Smoothing Embeddings", leave=False, unit="node")
    for i in range(num_nodes):
        # Create a DataFrame for the current node's time series
        # Using DataFrame rolling is convenient for handling edges (min_periods=1)
        node_ts_df = pd.DataFrame(embeddings_np[i, :, :]) # Shape: [num_timesteps, embedding_dim]
        smoothed_ts = node_ts_df.rolling(window=window_size, min_periods=1).mean()
        smoothed_embeddings_np[i, :, :] = smoothed_ts.values
        pbar_smooth.update(1)
    pbar_smooth.close()

    end_time = time.time()
    logging.info(f"Embedding smoothing finished. Duration: {end_time - start_time:.2f} seconds")
    return smoothed_embeddings_np


def apply_clustering(embeddings_at_t: np.ndarray, config: dict, k_clusters: int) -> np.ndarray:
    """
    Applies a clustering algorithm to embeddings at a specific time step for a given K.

    Args:
        embeddings_at_t (np.ndarray): Node embeddings [nodes, dim].
        config (dict): Configuration dictionary.
        k_clusters (int): The number of clusters (K) to form.

    Returns:
        np.ndarray: Cluster labels for each node [nodes].
    """
    method = config.get('CLUSTER_METHOD', 'SpectralClustering')
    seed = config.get('SEED', 42)
    scale_embeddings = config.get('SCALE_EMBEDDINGS_BEFORE_CLUSTERING', False)

    if embeddings_at_t.shape[0] < k_clusters:
        logging.warning(f"Clustering t={config.get('_current_t', '?')}: Number of nodes ({embeddings_at_t.shape[0]}) is less than K ({k_clusters}). Assigning all to cluster 0.")
        return np.zeros(embeddings_at_t.shape[0], dtype=int)
    if embeddings_at_t.shape[0] <= 1:
        logging.warning(f"Clustering t={config.get('_current_t', '?')}: Only {embeddings_at_t.shape[0]} node(s). Assigning to cluster 0.")
        return np.zeros(embeddings_at_t.shape[0], dtype=int)


    embeddings_processed = embeddings_at_t
    if scale_embeddings:
        scaler = StandardScaler()
        embeddings_processed = scaler.fit_transform(embeddings_at_t)

    labels = np.zeros(embeddings_at_t.shape[0], dtype=int) # Default labels
    try:
        if method.lower() == 'kmeans':
            kmeans = KMeans(n_clusters=k_clusters, random_state=seed, n_init='auto')
            labels = kmeans.fit_predict(embeddings_processed)
        elif method.lower() == 'spectralclustering':
            # Handle n_neighbors > n_samples scenario
            n_neighbors = min(config.get('SPECTRAL_N_NEIGHBORS', 10), embeddings_processed.shape[0] - 1)
            if n_neighbors < 1: n_neighbors = 1 # Ensure at least 1 neighbor if possible

            spectral = SpectralClustering(n_clusters=k_clusters,
                                          assign_labels='kmeans',
                                          random_state=seed,
                                          affinity='nearest_neighbors',
                                          n_neighbors=n_neighbors,
                                          n_init=10) # n_init for kmeans assigner
            labels = spectral.fit_predict(embeddings_processed)
        else:
            logging.error(f"Unsupported CLUSTER_METHOD: {method}. Defaulting to KMeans.")
            kmeans = KMeans(n_clusters=k_clusters, random_state=seed, n_init='auto')
            labels = kmeans.fit_predict(embeddings_processed)

    except Exception as e:
        logging.error(f"Clustering failed at t={config.get('_current_t', '?')} with K={k_clusters}, Method={method}: {e}. Assigning nodes to cluster 0.")
        # Keep default labels (all zeros)

    return labels


def _calculate_average_silhouette(embeddings_np: np.ndarray, assignments_dict: dict, num_timesteps: int) -> float:
    """Helper function to calculate average Silhouette score over time for given assignments."""
    silhouette_scores = []
    logging.info(f"Calculating average Silhouette score for K={len(np.unique(list(assignments_dict.get(0, {}).values())))}...") # Estimate K from first step
    pbar_sil = tqdm(range(num_timesteps), desc="Calculating Silhouette", leave=False, unit="step")

    for t in pbar_sil:
        assignments_t = assignments_dict.get(t)
        if not assignments_t: continue # Skip if no assignments for this step

        # Ensure consistent node order (use keys from assignments which should match node_ids order)
        # Note: This assumes assignments_dict keys are string versions of node_ids used to index embeddings_np
        try:
            labels_t = np.array([assignments_t.get(str(node_id), -1) for node_id in range(embeddings_np.shape[0])]) # Assumes node_ids are 0..N-1 here if not passed
            # TODO: Improve node ID handling - ideally pass node_ids list
        except Exception as e:
             logging.error(f"Error creating labels array for silhouette calc at t={t}: {e}")
             continue

        embeddings_t = embeddings_np[:, t, :]

        valid_mask = (labels_t != -1) # Filter nodes that might have been missed
        if np.sum(valid_mask) < 2: continue

        embeddings_t_valid = embeddings_t[valid_mask]
        labels_t_valid = labels_t[valid_mask]

        n_labels = len(np.unique(labels_t_valid))
        if n_labels < 2 or n_labels >= len(labels_t_valid):
            continue # Score undefined

        try:
            score = silhouette_score(embeddings_t_valid, labels_t_valid)
            silhouette_scores.append(score)
        except Exception as e_sil:
            logging.warning(f"Could not calculate Silhouette score for time step {t}: {e_sil}")
            continue

    avg_score = np.nanmean(silhouette_scores) if silhouette_scores else -1.0 # Return -1 if no scores calculable
    logging.info(f"Average Silhouette score: {avg_score:.4f}")
    return avg_score


def save_cluster_assignments(cluster_assignments: dict, node_ids: list, timestamps: list, config: dict, k_value: int):
    """
    Saves the cluster assignments dictionary to a CSV file, potentially including K in filename.
    """
    scenario_dir = config['SCENARIO_RESULTS_DIR']
    # Optional: Add K to filename if selecting K, otherwise use default name
    k_suffix = f"_k{k_value}" if config.get('CLUSTER_K_SELECTION_ENABLED', False) else ""
    filename = f'cluster_assignments{k_suffix}.csv'
    save_path = os.path.join(scenario_dir, filename)
    logging.info(f"Saving cluster assignments for K={k_value} to: {save_path}")

    records = []
    try:
        sorted_time_indices = sorted(cluster_assignments.keys())
        node_id_map = {i: str(node_id) for i, node_id in enumerate(node_ids)} # Map node index to original ID string

        for time_idx in sorted_time_indices:
            timestamp = timestamps[time_idx] # Get corresponding timestamp string
            assignments_t = cluster_assignments.get(time_idx, {}) # Inner dict {node_id_str: cluster_id}

            # Iterate through node_ids list to ensure all nodes are included in order
            for node_idx, original_node_id in enumerate(node_ids):
                node_id_str = str(original_node_id) # Key in assignments_t should be string
                cluster_label = assignments_t.get(node_id_str, -1) # Use .get for safety, default to -1 if missing
                records.append({
                    'timestamp': timestamp,
                    'time_index': time_idx,
                    'building_id': original_node_id, # Save original ID
                    'cluster_id': cluster_label
                })

        results_df = pd.DataFrame(records)
        # Ensure correct dtypes before saving if needed (already done in format_cluster_assignments)
        results_df.sort_values(by=['time_index', 'building_id'], inplace=True)
        results_df.to_csv(save_path, index=False)
        logging.info(f"Successfully saved {len(results_df)} assignment records to {filename}.")

    except Exception as e:
        logging.error(f"Error saving cluster assignments for K={k_value}: {e}")


def perform_temporal_clustering(config: dict, final_embeddings: torch.Tensor, dataset: SpatioTemporalGraphDataset) -> dict:
    """执行时间序列聚类"""
    mode = dataset.mode
    logging.info(f"--- Starting Temporal Clustering for {mode} set ---")
    
    # 修改结果保存路径以包含模式标识
    scenario_dir = config['SCENARIO_RESULTS_DIR']
    k_value = config.get('NUM_CLUSTERS_K', 6)
    filename = f'cluster_assignments_{mode}.csv'
    save_path = os.path.join(scenario_dir, filename)
    
    start_time_total = time.time()

    # --- Parameters ---
    window_size = config.get('EMBEDDING_SMOOTHING_WINDOW_W', 1)
    k_selection_enabled = config.get('CLUSTER_K_SELECTION_ENABLED', False)
    node_ids = dataset.node_ids # Original node IDs in order
    num_nodes, num_timesteps, _ = final_embeddings.shape

    # --- Smooth Embeddings ---
    # Perform smoothing once, regardless of K selection
    smoothed_embeddings_np = smooth_embeddings(final_embeddings, window_size)

    # --- Determine K and Perform Clustering ---
    best_k = config.get('NUM_CLUSTERS_K', 6) # Default K if selection disabled or fails
    final_cluster_assignments = {}

    if k_selection_enabled:
        logging.info("--- K Selection Enabled ---")
        k_range_cfg = config.get('CLUSTER_K_SELECTION_RANGE', [3, 10])
        if not isinstance(k_range_cfg, list) or len(k_range_cfg) != 2:
            logging.error("Invalid CLUSTER_K_SELECTION_RANGE in config. Must be list [min_k, max_k]. Using default K.")
            k_range = [best_k, best_k] # Fallback to default K
        else:
            k_range = list(range(k_range_cfg[0], k_range_cfg[1] + 1))

        best_score = -1.1 # Silhouette scores are between -1 and 1
        best_assignments_for_k = None

        for k in k_range:
            logging.info(f"--- Evaluating K = {k} ---")
            start_time_k = time.time()
            cluster_assignments_k = {}
            pbar_cluster = tqdm(total=num_timesteps, desc=f"Clustering K={k}", leave=False, unit="step")

            for t in range(num_timesteps):
                config['_current_t'] = t # Pass time index for logging inside apply_clustering
                embeddings_at_t = smoothed_embeddings_np[:, t, :]
                labels_t = apply_clustering(embeddings_at_t, config, k_clusters=k)
                # Store assignments using original node IDs (as strings) as keys
                assignments_t = {str(node_ids[i]): int(labels_t[i]) for i in range(num_nodes)}
                cluster_assignments_k[t] = assignments_t
                pbar_cluster.update(1)
            pbar_cluster.close()

            # Calculate average silhouette score for this K
            # Pass node_ids if helper needs it, current version assumes 0..N-1 indexing
            avg_silhouette = _calculate_average_silhouette(smoothed_embeddings_np, cluster_assignments_k, num_timesteps)
            logging.info(f"Finished K={k}. Avg Silhouette: {avg_silhouette:.4f}. Duration: {time.time() - start_time_k:.2f}s")

            if avg_silhouette > best_score:
                 best_score = avg_silhouette
                 best_k = k
                 best_assignments_for_k = cluster_assignments_k # Store the assignments for the best K

        logging.info(f"--- K Selection Finished. Best K={best_k} with score {best_score:.4f} ---")
        if best_assignments_for_k is None:
             logging.warning("No best assignments found during K selection (possibly all scores were invalid). Clustering with default K.")
             # Rerun with default K if needed, or handle error
             k_fixed = config.get('NUM_CLUSTERS_K', 6)
             logging.info(f"Re-running clustering with default K = {k_fixed}")
             best_k = k_fixed # Ensure best_k is set
             # Re-run clustering loop (could be refactored into a function)
             pbar_cluster = tqdm(total=num_timesteps, desc=f"Clustering K={best_k}", leave=False, unit="step")
             for t in range(num_timesteps):
                 config['_current_t'] = t
                 embeddings_at_t = smoothed_embeddings_np[:, t, :]
                 labels_t = apply_clustering(embeddings_at_t, config, k_clusters=best_k)
                 assignments_t = {str(node_ids[i]): int(labels_t[i]) for i in range(num_nodes)}
                 final_cluster_assignments[t] = assignments_t
                 pbar_cluster.update(1)
             pbar_cluster.close()
        else:
             final_cluster_assignments = best_assignments_for_k

    else: # K selection disabled
        logging.info(f"--- K Selection Disabled. Using fixed K = {best_k} ---")
        start_time_k = time.time()
        pbar_cluster = tqdm(total=num_timesteps, desc=f"Clustering K={best_k}", leave=False, unit="step")
        for t in range(num_timesteps):
            config['_current_t'] = t
            embeddings_at_t = smoothed_embeddings_np[:, t, :]
            labels_t = apply_clustering(embeddings_at_t, config, k_clusters=best_k)
            assignments_t = {str(node_ids[i]): int(labels_t[i]) for i in range(num_nodes)}
            final_cluster_assignments[t] = assignments_t
            pbar_cluster.update(1)
        pbar_cluster.close()
        logging.info(f"Finished clustering for fixed K={best_k}. Duration: {time.time() - start_time_k:.2f}s")

    # --- Save Results for the chosen K ---
    if config.get('SAVE_CLUSTER_ASSIGNMENTS', True) and final_cluster_assignments:
        try:
            # Reconstruct timestamps for saving
            start_time_dt = pd.to_datetime("00:00:00", format="%H:%M:%S") # Assumes data starts at midnight
            freq = config.get('TIME_SERIES_FREQUENCY', '15min')
            timestamps = pd.date_range(start_time_dt, periods=num_timesteps, freq=freq).strftime('%H:%M:%S').tolist()
            save_cluster_assignments(final_cluster_assignments, node_ids, timestamps, config, k_value=best_k)
        except Exception as e:
            logging.error(f"Could not reconstruct timestamps or save assignments. Error: {e}")

    clustering_duration = time.time() - start_time_total
    logging.info(f"--- Temporal Clustering Finished. Chosen K={best_k}. Total Duration: {clustering_duration:.2f} seconds ---")

    # Clean up temp key in config if added
    if '_current_t' in config: del config['_current_t']

    return final_cluster_assignments


# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    logging.info("--- Testing clustering script standalone ---")

    # Dummy config for testing standalone
    config_test = {
        'SCENARIO_NAME': 'Scenario_Clustering_Test',
        'PROCESSED_DATA_DIR': os.path.join('..', 'data', 'processed'), # Adjust if needed
        'RESULTS_DIR': os.path.join('..', 'results'),
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Scenario_Clustering_Test'),
        'EMBEDDING_SMOOTHING_WINDOW_W': 4,
        'CLUSTER_METHOD': 'SpectralClustering', # Test SpectralClustering
        'SPECTRAL_N_NEIGHBORS': 10, # Param for Spectral
        'NUM_CLUSTERS_K': 5,       # Default K if selection is off
        'CLUSTER_K_SELECTION_ENABLED': True, # Enable K selection
        'CLUSTER_K_SELECTION_RANGE': [4, 7], # Test K=4, 5, 6, 7
        'SEED': 42,
        'SAVE_CLUSTER_ASSIGNMENTS': True,
        'TIME_SERIES_FREQUENCY': '15min',
        'EMBEDDING_DIM': 16, # Example dimension
        'SCALE_EMBEDDINGS_BEFORE_CLUSTERING': False, # Test without scaling
    }
    os.makedirs(config_test['SCENARIO_RESULTS_DIR'], exist_ok=True)

    try:
        # Create Dummy Embeddings
        num_nodes_test = 50
        num_timesteps_test = 96 # One day at 15 min intervals
        dummy_embeddings = torch.randn(num_nodes_test, num_timesteps_test, config_test['EMBEDDING_DIM'])

        # Create Dummy Dataset object (only needs node_ids for this test)
        if SpatioTemporalGraphDataset is None: # Define dummy if import failed
             class DummyDataset:
                 def __init__(self, n_nodes):
                      self.node_ids = [f'B{i:03d}' for i in range(n_nodes)]
        else: # Use the actual class name if imported
             class DummyDataset(SpatioTemporalGraphDataset): # Inherit to satisfy type hint if needed
                 def __init__(self, n_nodes):
                      self.node_ids = [f'B{i:03d}' for i in range(n_nodes)]
                      # Add dummy attributes if needed by other parts of the class if not fully mocked
                      self.num_nodes = n_nodes

        dummy_dataset = DummyDataset(num_nodes_test)

        # Perform Clustering (with K selection)
        cluster_assignments_final = perform_temporal_clustering(config_test, dummy_embeddings, dummy_dataset)

        print("\n--- Standalone Clustering Test Finished ---")
        if cluster_assignments_final:
            first_t_assign = cluster_assignments_final.get(0, {})
            num_clusters_found = len(np.unique(list(first_t_assign.values()))) if first_t_assign else 'N/A'
            print(f"Generated cluster assignments for {len(cluster_assignments_final)} time steps.")
            print(f"Number of clusters found (at t=0 for best K): {num_clusters_found}")
            print("Assignments for T=0, Best K (first 5 nodes):")
            for i, node_id in enumerate(dummy_dataset.node_ids[:5]):
                print(f"  Node {node_id}: Cluster {first_t_assign.get(str(node_id), 'N/A')}")
            print(f"Check results/{config_test['SCENARIO_NAME']}/ for cluster_assignments_kX.csv")
        else:
            print("Clustering did not produce final assignments.")

    except Exception as e:
        logging.exception(f"An error occurred during clustering standalone test: {e}")
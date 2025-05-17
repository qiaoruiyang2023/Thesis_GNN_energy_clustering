# src/evaluation.py

# src/evaluation.py

import torch
import numpy as np
import pandas as pd
import os
import logging
import time
import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings

# Import necessary functions from other project modules
try:
    from src.datasets import SpatioTemporalGraphDataset
    from src.clustering import smooth_embeddings # <-- IMPORT ADDED
except ImportError as e:
    logging.error(f"Error importing project modules in evaluation.py: {e}")
    # Handle error or re-raise if critical
    raise

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress specific warnings if they become too noisy
# warnings.filterwarnings('ignore', message='Graph is not fully connected') # Example for spectral clustering warning
# warnings.filterwarnings('ignore', message='KMeans is known to have a memory leak') # Example for KMeans warning

# --- Helper Functions ---

def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance calculation."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def format_cluster_assignments(cluster_assignments_dict: dict, node_ids: list, num_timesteps: int) -> pd.DataFrame:
    """Converts the cluster assignment dictionary {time_idx: {node_id: cluster_id}} to a long-format DataFrame."""
    records = []
    node_id_map = {str(node_id): node_id for node_id in node_ids}

    for t in range(num_timesteps):
        assignments_t = cluster_assignments_dict.get(t, {})
        if not assignments_t: # Handle potentially empty time steps if clustering failed
             logging.warning(f"No cluster assignments found for time index {t}.")
             continue
        for node_id_str, cluster_id in assignments_t.items():
            original_node_id = node_id_map.get(node_id_str)
            if original_node_id is not None:
                records.append({
                    'time_index': t,
                    'building_id': original_node_id, # Use original ID type
                    'cluster_id': cluster_id
                })
            else:
                logging.warning(f"Node ID '{node_id_str}' found in cluster assignments but not in dataset's node_ids at time {t}.")

    if not records:
        logging.warning("No records generated from cluster assignments dictionary.")
        return pd.DataFrame(columns=['time_index', 'building_id', 'cluster_id'])

    df = pd.DataFrame(records)
    # Ensure correct types
    df['time_index'] = df['time_index'].astype(int)
    # Convert building_id to string for consistency in merges later
    df['building_id'] = df['building_id'].astype(str)
    df['cluster_id'] = df['cluster_id'].astype(int)
    return df


# --- Metric Calculation Functions ---

def calculate_synergy_metrics(cluster_assign_df: pd.DataFrame, dynamic_features_df: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Calculates synergy metrics (SSR, NetImport) for each cluster over time.

    Returns:
        tuple[dict, pd.DataFrame]:
            - Dictionary with average metrics (avg_cluster_ssr, avg_abs_cluster_net_import, total_net_import).
            - DataFrame with metrics per cluster per time step.
    """
    logging.info("Calculating synergy metrics...")
    avg_metrics = {'avg_cluster_ssr': np.nan, 'avg_abs_cluster_net_import': np.nan, 'total_net_import': np.nan}
    empty_df = pd.DataFrame(columns=['time_index', 'cluster_id', 'net_import', 'ssr']) # Define empty df structure

    if 'net_load' not in dynamic_features_df.columns:
        logging.error("'net_load' column not found in dynamic features. Cannot calculate synergy.")
        return avg_metrics, empty_df
    if cluster_assign_df.empty:
         logging.warning("Cluster assignment DataFrame is empty. Skipping synergy calculation.")
         return avg_metrics, empty_df

    # Prepare dynamic features: ensure building_id is string and add time_index if missing
    dynamic_features_df['building_id'] = dynamic_features_df['building_id'].astype(str)
    if 'time_index' not in dynamic_features_df.columns:
         # Assuming dynamic features are sorted by time for each building
         dynamic_features_df['time_index'] = dynamic_features_df.groupby('building_id').cumcount()

    # Merge assignments with net load data
    try:
        # Select only necessary columns for merge to avoid conflicts
        dyn_subset = dynamic_features_df[['building_id', 'time_index', 'net_load']]
        merged_df = pd.merge(cluster_assign_df, dyn_subset, on=['building_id', 'time_index'], how='left')
    except Exception as e:
         logging.error(f"Error merging cluster assignments with dynamic features: {e}. Check keys/columns.")
         logging.error(f"Cluster Assign Columns: {cluster_assign_df.columns}")
         logging.error(f"Dynamic Features Subset Columns: {dyn_subset.columns}")
         return avg_metrics, empty_df

    if merged_df['net_load'].isnull().any():
        num_missing = merged_df['net_load'].isnull().sum()
        logging.warning(f"{num_missing} missing net_load values after merging. Filling with 0 for synergy calc.")
        merged_df['net_load'].fillna(0.0, inplace=True)

    # Group by time and cluster
    grouped = merged_df.groupby(['time_index', 'cluster_id'])

    # Calculate metrics per group
    synergy_results_df = pd.DataFrame()
    synergy_results_df['net_import'] = grouped['net_load'].sum()
    synergy_results_df['abs_net_load_sum'] = np.abs(synergy_results_df['net_import'])
    synergy_results_df['sum_abs_net_load'] = grouped['net_load'].apply(lambda x: np.sum(np.abs(x)))

    # Calculate Self-Sufficiency Ratio (SSR) = 1 - |sum(net_load)| / sum(|net_load|)
    # Handle division by zero: if sum_abs_net_load is 0, cluster is perfectly self-sufficient (SSR=1)
    synergy_results_df['ssr'] = 1.0 # Default to 1
    # Calculate only where denominator is non-zero
    valid_denom_mask = synergy_results_df['sum_abs_net_load'] != 0
    synergy_results_df.loc[valid_denom_mask, 'ssr'] = 1.0 - (synergy_results_df.loc[valid_denom_mask, 'abs_net_load_sum'] / synergy_results_df.loc[valid_denom_mask, 'sum_abs_net_load'])
    # Ensure SSR is between 0 and 1
    synergy_results_df['ssr'] = np.clip(synergy_results_df['ssr'], 0, 1)
    synergy_results_df = synergy_results_df.reset_index() # Make time_index and cluster_id columns

    # --- Aggregate Results ---
    if not synergy_results_df.empty:
         avg_metrics['avg_cluster_ssr'] = synergy_results_df['ssr'].mean()
         avg_metrics['avg_abs_cluster_net_import'] = synergy_results_df['abs_net_load_sum'].mean()
         # Calculate total net import (sum over time for each cluster, then sum over clusters)
         # This should equal the sum of net_load in the original dynamic data for included buildings
         avg_metrics['total_net_import'] = synergy_results_df['net_import'].sum()

    logging.info(f"Synergy Metrics (Avg): SSR={avg_metrics['avg_cluster_ssr']:.4f}, Abs Cluster Net Import={avg_metrics['avg_abs_cluster_net_import']:.2f}")
    
    # 修改这一行，使用 config 中的 MODE
    mode = config.get('MODE', 'unknown')
    print(f"Saving {mode} metrics, time steps: {synergy_results_df['time_index'].nunique()}")
    
    return avg_metrics, synergy_results_df


def calculate_stability_metrics(cluster_assign_df: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Calculates stability metrics (ARI, NMI, Churn Rate) over time.

    Returns:
        tuple[dict, pd.DataFrame]:
            - Dictionary with average metrics (avg_ari, avg_nmi, avg_churn_rate).
            - DataFrame with metrics per time step transition (t-1 to t).
    """
    logging.info("Calculating stability metrics...")
    avg_metrics = {'avg_ari': np.nan, 'avg_nmi': np.nan, 'avg_churn_rate': np.nan}
    stability_results_list = []
    empty_df = pd.DataFrame(columns=['time_index_t', 'ari', 'nmi', 'churn_rate'])

    if cluster_assign_df.empty:
        logging.warning("Cluster assignment DataFrame is empty. Skipping stability calculation.")
        return avg_metrics, empty_df

    num_timesteps = cluster_assign_df['time_index'].max() + 1
    num_nodes = cluster_assign_df['building_id'].nunique()

    if num_timesteps < 2:
        logging.warning("Need at least 2 time steps for stability metrics. Skipping.")
        return avg_metrics, empty_df

    # Pivot to easily access labels per time step: index=time_index, columns=building_id
    try:
        pivot_labels = cluster_assign_df.pivot(index='time_index', columns='building_id', values='cluster_id')
    except Exception as e:
        logging.error(f"Error pivoting cluster assignments for stability calculation: {e}")
        # Check for duplicates, which usually cause pivot errors
        duplicates = cluster_assign_df[cluster_assign_df.duplicated(subset=['time_index', 'building_id'], keep=False)]
        logging.error(f"Duplicate assignments found:\n{duplicates}")
        return avg_metrics, empty_df

    aris = []
    nmis = []
    churns = []

    logging.info(f"Calculating stability for {num_timesteps-1} transitions...")
    for t in tqdm(range(1, num_timesteps), desc="Calculating Stability", leave=False):
        try:
            # Use .get() with default to handle missing time indices gracefully
            labels_t = pivot_labels.loc[t]
            labels_t_minus_1 = pivot_labels.loc[t-1]

            # Align labels by building_id (index) and drop buildings not present at both times
            common_labels = pd.concat([labels_t_minus_1, labels_t], axis=1, keys=['t_minus_1', 't']).dropna()

            if len(common_labels) < 2: # Need at least 2 common points
                ari_t, nmi_t, churn_t = np.nan, np.nan, np.nan
                continue

            labels_t_valid = common_labels['t'].values
            labels_t_minus_1_valid = common_labels['t_minus_1'].values

            # Calculate churn rate
            churn_t = np.sum(labels_t_valid != labels_t_minus_1_valid) / len(labels_t_valid)

            # Calculate ARI/NMI
            if len(np.unique(labels_t_valid)) < 2 or len(np.unique(labels_t_minus_1_valid)) < 2:
                # Handle cases where all points are in one cluster (or only one common point)
                # NMI/ARI are 1 if the single cluster label is identical, 0 otherwise
                ari_t = 1.0 if np.array_equal(labels_t_valid, labels_t_minus_1_valid) else 0.0
                nmi_t = 1.0 if np.array_equal(labels_t_valid, labels_t_minus_1_valid) else 0.0
            else:
                with warnings.catch_warnings(): # Suppress potential sklearn warnings here too if needed
                     # warnings.simplefilter("ignore")
                     ari_t = adjusted_rand_score(labels_t_minus_1_valid, labels_t_valid)
                     nmi_t = normalized_mutual_info_score(labels_t_minus_1_valid, labels_t_valid)

            aris.append(ari_t)
            nmis.append(nmi_t)
            churns.append(churn_t)
            stability_results_list.append({'time_index_t': t, 'ari': ari_t, 'nmi': nmi_t, 'churn_rate': churn_t})

        except KeyError:
            logging.warning(f"Missing time index {t} or {t-1} in pivoted labels. Skipping stability calc for this step.")
            stability_results_list.append({'time_index_t': t, 'ari': np.nan, 'nmi': np.nan, 'churn_rate': np.nan})
            continue
        except Exception as e_stab:
             logging.error(f"Error calculating stability for t={t}: {e_stab}")
             stability_results_list.append({'time_index_t': t, 'ari': np.nan, 'nmi': np.nan, 'churn_rate': np.nan})
             continue

    # Calculate averages, ignoring NaNs
    avg_metrics['avg_ari'] = np.nanmean(aris) if aris else np.nan
    avg_metrics['avg_nmi'] = np.nanmean(nmis) if nmis else np.nan
    avg_metrics['avg_churn_rate'] = np.nanmean(churns) if churns else np.nan

    stability_results_df = pd.DataFrame(stability_results_list)

    logging.info(f"Stability Metrics (Avg): ARI={avg_metrics['avg_ari']:.4f}, NMI={avg_metrics['avg_nmi']:.4f}, Churn Rate={avg_metrics['avg_churn_rate']:.4f}")
    return avg_metrics, stability_results_df


# Keep feasibility check as overall average - per-timestep is complex
def calculate_feasibility_metrics(cluster_assign_df: pd.DataFrame, static_features_df: pd.DataFrame, config: dict) -> dict:
    """Calculates feasibility violation rates (overall average)."""
    avg_metrics = {'avg_feeder_violation_rate': np.nan, 'avg_distance_violation_rate': np.nan}
    if not config.get('CHECK_FEASIBILITY_POST_CLUSTER', False):
        logging.info("Skipping feasibility checks as per scenario configuration.")
        return avg_metrics

    logging.info("Calculating feasibility metrics (overall average)...")
    if 'line_id' not in static_features_df.columns:
        logging.warning("Missing 'line_id' in static features, cannot check feeder violations.")
        # Return dict with NaNs, avg_distance might still be calculated
    if not all(col in static_features_df.columns for col in ['lat', 'lon']):
        logging.warning("Missing 'lat'/'lon' in static features, cannot check distance violations.")
        return avg_metrics # Can't calculate distance either

    threshold_km = config.get('FEASIBILITY_DISTANCE_THRESHOLD_KM', 0.5)
    num_timesteps = cluster_assign_df['time_index'].max() + 1 if not cluster_assign_df.empty else 0

    # Merge static info once
    # Ensure indices/keys are compatible strings
    static_info = static_features_df[['line_id', 'lat', 'lon']].astype({'line_id': str})
    static_info.index = static_info.index.astype(str)
    cluster_assign_df['building_id'] = cluster_assign_df['building_id'].astype(str)
    # Perform merge, keep track of original cluster_assign_df index if needed
    merged_df = pd.merge(cluster_assign_df, static_info, left_on='building_id', right_index=True, how='left')

    if merged_df['lat'].isnull().any() or merged_df['lon'].isnull().any():
         logging.warning("Missing lat/lon data after merging static features. Distance checks might be incomplete.")
    if merged_df['line_id'].isnull().any():
         logging.warning("Missing line_id data after merging static features. Feeder checks might be incomplete.")


    total_pairs_checked = 0
    feeder_violations = 0
    distance_violations = 0

    # Check pairs within each cluster at each time step
    for t in tqdm(range(num_timesteps), desc="Checking Feasibility", leave=False):
        df_t = merged_df[merged_df['time_index'] == t]
        if df_t.empty: continue

        for cluster_id in df_t['cluster_id'].unique():
            cluster_nodes = df_t[df_t['cluster_id'] == cluster_id]
            n_cluster = len(cluster_nodes)
            if n_cluster < 2: continue

            # Get building IDs and corresponding static info for pairs
            building_ids_in_cluster = cluster_nodes['building_id'].tolist()

            for i in range(n_cluster):
                for j in range(i + 1, n_cluster):
                    id1, id2 = building_ids_in_cluster[i], building_ids_in_cluster[j]
                    try:
                         node1_info = static_info.loc[id1]
                         node2_info = static_info.loc[id2]
                    except KeyError:
                         logging.warning(f"Building ID {id1} or {id2} not found in static_info index during feasibility check.")
                         continue # Skip pair if static info missing

                    total_pairs_checked += 1

                    # Feeder check (if line_id available)
                    if 'line_id' in node1_info and 'line_id' in node2_info:
                         if node1_info['line_id'] != node2_info['line_id']:
                             feeder_violations += 1

                    # Distance check (if lat/lon available)
                    if pd.notna(node1_info['lon']) and pd.notna(node1_info['lat']) and pd.notna(node2_info['lon']) and pd.notna(node2_info['lat']):
                         dist = haversine_np(node1_info['lon'], node1_info['lat'], node2_info['lon'], node2_info['lat'])
                         if dist > threshold_km:
                             distance_violations += 1

    # Calculate average rates
    if total_pairs_checked > 0:
        avg_metrics['avg_feeder_violation_rate'] = (feeder_violations / total_pairs_checked) if 'line_id' in static_features_df.columns else np.nan
        avg_metrics['avg_distance_violation_rate'] = (distance_violations / total_pairs_checked) if all(c in static_features_df.columns for c in ['lat','lon']) else np.nan
    else:
        # If no pairs checked, violation rates are arguably 0, or undefined (NaN)
        logging.warning("No pairs checked for feasibility (clusters might be too small or only one cluster). Setting rates to 0.")
        avg_metrics['avg_feeder_violation_rate'] = 0.0
        avg_metrics['avg_distance_violation_rate'] = 0.0


    logging.info(f"Feasibility Metrics (Avg): Feeder Violation Rate={avg_metrics['avg_feeder_violation_rate']:.4f}, Distance Violation Rate={avg_metrics['avg_distance_violation_rate']:.4f}")
    return avg_metrics


def calculate_quality_metrics(smoothed_embeddings_np: np.ndarray, cluster_assign_df: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Calculates clustering quality metrics (Silhouette Score) over time.

    Args:
        smoothed_embeddings_np (np.ndarray): Smoothed embeddings [num_nodes, num_timesteps, embedding_dim].
                                              Node order must match node order in cluster_assign_df pivot.
        cluster_assign_df (pd.DataFrame): Long-format cluster assignments.
        config (dict): Configuration dictionary.

    Returns:
        tuple[dict, pd.DataFrame]:
            - Dictionary with average metric (avg_silhouette_score).
            - DataFrame with metric per time step.
    """
    logging.info("Calculating quality metrics (Silhouette Score)...")
    avg_metrics = {'avg_silhouette_score': np.nan}
    quality_results_list = []
    empty_df = pd.DataFrame(columns=['time_index', 'silhouette_score'])

    if cluster_assign_df.empty:
        logging.warning("Cluster assignment DataFrame is empty. Skipping quality calculation.")
        return avg_metrics, empty_df

    num_nodes_embed, num_timesteps, _ = smoothed_embeddings_np.shape
    num_nodes_assign = cluster_assign_df['building_id'].nunique()
    max_time_assign = cluster_assign_df['time_index'].max()

    if num_nodes_embed != num_nodes_assign:
         logging.error(f"Embeddings node count ({num_nodes_embed}) != Assignments node count ({num_nodes_assign}). Cannot calculate Silhouette.")
         return avg_metrics, empty_df
    if num_timesteps != (max_time_assign + 1):
         logging.warning(f"Embeddings timestep count ({num_timesteps}) != Assignments timestep count ({max_time_assign + 1}). Using assignment count.")
         num_timesteps = max_time_assign + 1 # Adjust based on assignments

    # Pivot assignments for easy lookup: index=time_index, columns=building_id
    try:
        pivot_labels = cluster_assign_df.pivot(index='time_index', columns='building_id', values='cluster_id')
        # Ensure node order in embeddings matches pivot columns
        node_order = pivot_labels.columns.tolist() # This is the required order
        # We assume the embeddings provided have nodes in the same order as pivot_labels.columns
        # (This relies on dataset.node_ids being consistent and used correctly upstream)

    except Exception as e:
        logging.error(f"Error pivoting cluster assignments for quality calculation: {e}")
        return avg_metrics, empty_df

    silhouette_scores = []
    for t in tqdm(range(num_timesteps), desc="Calculating Silhouette", leave=False):
        score_t = np.nan # Default to NaN
        try:
            embeddings_t = smoothed_embeddings_np[:, t, :] # Assumes node order matches pivot cols
            if t not in pivot_labels.index:
                 logging.warning(f"Time index {t} not found in pivoted labels. Skipping Silhouette.")
                 continue

            labels_t = pivot_labels.loc[t].values

            # Filter out nodes with NaN labels (e.g., if a node wasn't assigned)
            valid_mask = ~np.isnan(labels_t)
            if np.sum(valid_mask) < 2: continue # Need at least 2 assigned points

            embeddings_t_valid = embeddings_t[valid_mask]
            labels_t_valid = labels_t[valid_mask].astype(int) # Ensure integer labels

            # Silhouette score requires at least 2 unique cluster labels and more points than clusters
            n_labels = len(np.unique(labels_t_valid))
            if n_labels < 2 or n_labels >= len(labels_t_valid):
                 # Score is ill-defined, skip or assign default (e.g., 0 or NaN)
                 logging.debug(f"Skipping Silhouette for t={t}: Found {n_labels} unique labels for {len(labels_t_valid)} points.")
                 continue

            score_t = silhouette_score(embeddings_t_valid, labels_t_valid)
            silhouette_scores.append(score_t)

        except Exception as e_qual:
            logging.warning(f"Could not calculate Silhouette score for time step {t}: {e_qual}")
            # Keep score_t as NaN

        quality_results_list.append({'time_index': t, 'silhouette_score': score_t})

    # Calculate average, ignoring NaNs
    avg_metrics['avg_silhouette_score'] = np.nanmean(silhouette_scores) if silhouette_scores else np.nan
    quality_results_df = pd.DataFrame(quality_results_list)

    logging.info(f"Quality Metrics (Avg): Silhouette Score={avg_metrics['avg_silhouette_score']:.4f}")
    return avg_metrics, quality_results_df


def calculate_transition_matrix(cluster_assign_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculates the average transition matrix between clusters over consecutive time steps.

    Returns:
        pd.DataFrame: A DataFrame where index is cluster_id at t-1, columns are
                      cluster_id at t, and values are average transition counts/probabilities.
                      Returns empty DataFrame if calculation fails.
    """
    logging.info("Calculating cluster transition matrix...")
    empty_df = pd.DataFrame()
    if cluster_assign_df.empty:
        logging.warning("Cluster assignment DataFrame is empty. Skipping transition calculation.")
        return empty_df

    num_timesteps = cluster_assign_df['time_index'].max() + 1
    if num_timesteps < 2:
        logging.warning("Need at least 2 time steps for transition matrix. Skipping.")
        return empty_df

    # Pivot to easily access labels per time step
    try:
        pivot_labels = cluster_assign_df.pivot(index='time_index', columns='building_id', values='cluster_id')
    except Exception as e:
        logging.error(f"Error pivoting cluster assignments for transition calculation: {e}")
        return empty_df

    all_transitions = []
    num_clusters = config.get('NUM_CLUSTERS_K', 6)
    cluster_ids = list(range(num_clusters)) # Assumes cluster IDs are 0 to K-1

    logging.info(f"Calculating transitions for {num_timesteps-1} steps...")
    for t in range(1, num_timesteps):
        try:
            labels_t = pivot_labels.loc[t]
            labels_t_minus_1 = pivot_labels.loc[t-1]

            # Align and drop NaNs
            common_labels = pd.concat([labels_t_minus_1, labels_t], axis=1, keys=['t_minus_1', 't']).dropna().astype(int)
            if common_labels.empty: continue

            # Calculate transition counts for this step using crosstab
            transition_counts = pd.crosstab(common_labels['t_minus_1'], common_labels['t'])

            # Reindex to ensure all possible clusters are present, fill missing with 0
            transition_counts = transition_counts.reindex(index=cluster_ids, columns=cluster_ids, fill_value=0)
            all_transitions.append(transition_counts)

        except KeyError:
            logging.warning(f"Missing time index {t} or {t-1} in pivoted labels for transitions.")
            continue
        except Exception as e_trans:
            logging.error(f"Error calculating transitions for t={t}: {e_trans}")
            continue

    if not all_transitions:
        logging.warning("No valid transitions found to calculate average matrix.")
        return empty_df

    # Calculate the average transition matrix (element-wise mean)
    avg_transition_matrix = pd.DataFrame(0, index=cluster_ids, columns=cluster_ids)
    if all_transitions:
        avg_transition_matrix = pd.concat(all_transitions).groupby(level=0).mean()
        # Reindex again in case some clusters never appeared as source/target
        avg_transition_matrix = avg_transition_matrix.reindex(index=cluster_ids, columns=cluster_ids, fill_value=0.0)

    # Optional: Normalize rows to get probabilities (sum of each row = 1)
    # transition_prob_matrix = avg_transition_matrix.apply(lambda x: x / x.sum(), axis=1).fillna(0.0)

    logging.info("Finished calculating average transition matrix.")
    # Return the average count matrix for now
    return avg_transition_matrix


# --- Main Evaluation Runner ---

def run_evaluation(config: dict, cluster_assignments: dict, final_embeddings: torch.Tensor,
                   dataset: SpatioTemporalGraphDataset):
    """运行评估流程"""
    logging.info(f"--- Starting Evaluation Pipeline for {dataset.mode} set ---")
    all_avg_metrics = {}
    results_dir = config['SCENARIO_RESULTS_DIR']

    # 添加模式标识到结果文件名
    mode = dataset.mode
    metrics_save_path = os.path.join(results_dir, f'evaluation_metrics_{mode}.json')
    summary_save_path = os.path.join(results_dir, f'evaluation_summary_{mode}.txt')

    # --- Prepare Inputs ---
    logging.info("Preparing data for evaluation...")
    # Convert cluster assignments dict to DataFrame
    num_timesteps = final_embeddings.shape[1]
    node_ids = dataset.node_ids # Get original node IDs in correct order
    cluster_assign_df = format_cluster_assignments(cluster_assignments, node_ids, num_timesteps)

    # Load processed dynamic features (need net_load)
    dynamic_features_path = os.path.join(config['PROCESSED_DATA_DIR'], 'processed_dynamic_features.parquet')
    try:
        dynamic_features_df = pd.read_parquet(dynamic_features_path)
    except Exception as e:
        logging.error(f"Failed to load dynamic features for evaluation: {e}")
        # Return empty results if dynamic features are crucial and missing
        return all_avg_metrics

    # Load processed static features (need line_id, lat, lon for feasibility)
    static_features_path = os.path.join(config['PROCESSED_DATA_DIR'], 'processed_static_features.csv')
    try:
        static_features_df = pd.read_csv(static_features_path, index_col='building_id')
        static_features_df.index = static_features_df.index.astype(str) # Ensure index is string
    except Exception as e:
        logging.error(f"Failed to load static features for evaluation: {e}")
        # Feasibility checks will fail if static features are missing
        # Return empty dict or proceed and let checks handle missing data? Proceed for now.
        static_features_df = pd.DataFrame() # Use empty df

    # --- Calculate Metrics (Time Series and Averages) ---
    # Synergy
    avg_synergy_metrics, synergy_results_df = calculate_synergy_metrics(cluster_assign_df.copy(), dynamic_features_df.copy(), config)
    all_avg_metrics.update(avg_synergy_metrics)
    if config.get('VIZ_PLOT_METRIC_TS', True) and not synergy_results_df.empty:
         synergy_results_df.to_csv(os.path.join(results_dir, f'synergy_metrics_timeseries_{mode}.csv'), index=False)
         logging.info(f"Saved synergy time series metrics to CSV.")

    # Stability
    avg_stability_metrics, stability_ts_df = calculate_stability_metrics(cluster_assign_df.copy(), config)
    all_avg_metrics.update(avg_stability_metrics)
    if config.get('VIZ_PLOT_METRIC_TS', True) and not stability_ts_df.empty:
         stability_ts_df.to_csv(os.path.join(results_dir, f'stability_metrics_timeseries_{mode}.csv'), index=False)
         logging.info(f"Saved stability time series metrics to CSV.")

    # Feasibility (returns only averages)
    feasibility_metrics = calculate_feasibility_metrics(cluster_assign_df.copy(), static_features_df.copy(), config)
    all_avg_metrics.update(feasibility_metrics)

    # Quality (Silhouette)
    try:
        window_size = config.get('EMBEDDING_SMOOTHING_WINDOW_W', 1)
        # Smoothing embeddings again - could be optimized if clustering passes smoothed embeddings
        logging.info(f"Applying moving average smoothing with window size {window_size} for quality metrics...")
        start_smooth_time = time.time()
        smoothed_embeddings_np = smooth_embeddings(final_embeddings, window_size)
        logging.info(f"Embedding smoothing finished. Duration: {time.time() - start_smooth_time:.2f} seconds")

        avg_quality_metrics, quality_ts_df = calculate_quality_metrics(smoothed_embeddings_np, cluster_assign_df.copy(), config)
        all_avg_metrics.update(avg_quality_metrics)
        if config.get('VIZ_PLOT_METRIC_TS', True) and not quality_ts_df.empty:
             quality_ts_df.to_csv(os.path.join(results_dir, f'quality_metrics_timeseries_{mode}.csv'), index=False)
             logging.info(f"Saved quality time series metrics to CSV.")

    except Exception as e:
        logging.error(f"Failed to calculate quality metrics: {e}")
        all_avg_metrics.update({'avg_silhouette_score': np.nan})

    # Transitions
    if config.get('VIZ_PLOT_TRANSITIONS', True):
        transition_matrix_df = calculate_transition_matrix(cluster_assign_df.copy(), config)
        if not transition_matrix_df.empty:
            transition_matrix_df.to_csv(os.path.join(results_dir, f'transition_matrix_{mode}.csv'))
            logging.info(f"Saved average transition matrix to CSV.")

    # --- Save Averaged Metrics Summary ---
    try:
        # Convert numpy types to native Python types for JSON
        serializable_metrics = {}
        for k, v in all_avg_metrics.items():
             if isinstance(v, (np.generic, np.ndarray)):
                 serializable_metrics[k] = v.item() # Convert numpy number types
             elif pd.isna(v):
                  serializable_metrics[k] = None # Convert Pandas/Numpy NaN to None
             else:
                 serializable_metrics[k] = v # Keep other types (int, float, bool)

        with open(metrics_save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logging.info(f"Saved AVERAGE evaluation metrics to: {metrics_save_path}")

        # Generate summary table text from the averaged metrics
        summary_text = f"Evaluation Summary for: {config.get('SCENARIO_NAME', 'Unknown')}\n"
        summary_text += "=" * 40 + "\n"
        for key, value in serializable_metrics.items():
            summary_text += f"{key:<30}: {value:.4f}\n" if isinstance(value, (float, np.floating)) else f"{key:<30}: {value}\n"
        summary_text += "=" * 40 + "\n"

        with open(summary_save_path, 'w') as f:
            f.write(summary_text)
        logging.info(f"Saved evaluation summary table to: {summary_save_path}")

    except Exception as e:
        logging.error(f"Error saving evaluation results: {e}")

    logging.info(f"--- Evaluation Pipeline Finished ---")
    # Return only the dictionary of averaged metrics for potential use in main.py
    return all_avg_metrics


# Example Usage Block - No changes needed here
if __name__ == '__main__':
     # ... (The standalone test block remains the same) ...
     pass

# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    # This block requires outputs from clustering, inference, and data processing
    logging.info("Testing evaluation script standalone...")

    # 1. Load Config (replace with actual loader)
    config = {
        'SCENARIO_NAME': 'Scenario_B_Feeder_Constraint', # Change to test others
        'PROCESSED_DATA_DIR': os.path.join('..', 'data', 'processed'), # Adjust relative path
        'RESULTS_DIR': os.path.join('..', 'results'),
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Scenario_B_Feeder_Constraint'),# Adjust
        'CHECK_FEASIBILITY_POST_CLUSTER': False, # Example for Scenario B
        'FEASIBILITY_DISTANCE_THRESHOLD_KM': 0.5,
        'EMBEDDING_SMOOTHING_WINDOW_W': 4,
        'EMBEDDING_DIM': 16 # Example
    }
    os.makedirs(config['SCENARIO_RESULTS_DIR'], exist_ok=True)


    try:
        # 2. Load dummy/real cluster assignments (replace with loading from file)
        assign_file = os.path.join(config['SCENARIO_RESULTS_DIR'], 'cluster_assignments.csv')
        if not os.path.exists(assign_file):
             print(f"Cluster assignments file not found: {assign_file}. Cannot run evaluation.")
        else:
             cluster_assign_df_loaded = pd.read_csv(assign_file)
             # Convert back to dictionary format expected? No, run_evaluation uses the DataFrame format now.
             # Recreate dict format for testing if needed, or adapt run_evaluation inputs
             cluster_assignments_dict = {}
             for name, group in cluster_assign_df_loaded.groupby('time_index'):
                  cluster_assignments_dict[name] = group.set_index('building_id')['cluster_id'].to_dict()


             # 3. Load dummy/real embeddings (replace with loading from file or inference output)
             num_nodes = cluster_assign_df_loaded['building_id'].nunique()
             num_timesteps = cluster_assign_df_loaded['time_index'].max() + 1
             dummy_embeddings = torch.randn(num_nodes, num_timesteps, config['EMBEDDING_DIM'])

             # 4. Load dummy/real dataset object (only needs node_ids)
             class DummyDataset:
                 def __init__(self, assign_df):
                     self.node_ids = assign_df['building_id'].unique().tolist()
             dummy_dataset = DummyDataset(cluster_assign_df_loaded)

             # 5. Run Evaluation (run_evaluation now handles loading internal files)
             run_evaluation(config, cluster_assignments_dict, dummy_embeddings, dummy_dataset) # Pass dict here

             print("\n--- Standalone Evaluation Test Finished ---")
             print(f"Check {config['SCENARIO_RESULTS_DIR']} for evaluation_metrics.json and evaluation_summary.txt")

    except FileNotFoundError:
         print("\nError: Prerequisite file not found (e.g., cluster_assignments.csv, processed data). Ensure previous steps ran.")
    except Exception as e:
        logging.exception(f"An error occurred during evaluation test: {e}")
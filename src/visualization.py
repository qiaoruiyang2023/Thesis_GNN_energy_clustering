# src/visualization.py

import torch
import numpy as np
import pandas as pd
import os
import logging
import time

# Import necessary functions from other project modules
try:
    # Needed for data preparation within this script
    from src.evaluation import format_cluster_assignments
    from src.clustering import smooth_embeddings
    from src.datasets import SpatioTemporalGraphDataset # For type hinting

    # Import plotting functions from sub-modules
    from src.viz_embedding import plot_embedding_projection
    from src.viz_clusters import (plot_cluster_evolution_gif,
                                  plot_average_cluster_timeseries,
                                  plot_cluster_composition,
                                  plot_transition_heatmap)
                                  # Import optional plots if implemented:
                                  # plot_geographic_cluster_shape,
                                  # plot_sankey_transitions,
                                  # plot_graph_structure
    from src.viz_metrics import (plot_evaluation_summary,
                                 plot_metric_timeseries,
                                 plot_k_selection_metrics)

except ImportError as e:
    logging.error(f"Error importing project modules in visualization.py: {e}")
    logging.error("Ensure visualization.py is run in an environment where other src modules are accessible.")
    # Define dummy functions if imports fail, allowing main script to potentially run without crashing here
    def plot_embedding_projection(*args, **kwargs): logger.error("plot_embedding_projection not loaded")
    def plot_cluster_evolution_gif(*args, **kwargs): logger.error("plot_cluster_evolution_gif not loaded")
    def plot_average_cluster_timeseries(*args, **kwargs): logger.error("plot_average_cluster_timeseries not loaded")
    def plot_cluster_composition(*args, **kwargs): logger.error("plot_cluster_composition not loaded")
    def plot_transition_heatmap(*args, **kwargs): logger.error("plot_transition_heatmap not loaded")
    def plot_evaluation_summary(*args, **kwargs): logger.error("plot_evaluation_summary not loaded")
    def plot_metric_timeseries(*args, **kwargs): logger.error("plot_metric_timeseries not loaded")
    def plot_k_selection_metrics(*args, **kwargs): logger.error("plot_k_selection_metrics not loaded")
    def format_cluster_assignments(*args, **kwargs): logger.error("format_cluster_assignments not loaded"); return pd.DataFrame() # Return empty df
    def smooth_embeddings(*args, **kwargs): logger.error("smooth_embeddings not loaded"); return np.array([]) # Return empty array


# Setup basic logging (configure in main script usually)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Main Visualization Runner ---

def run_visualization(config: dict, cluster_assignments: dict, final_embeddings: torch.Tensor,
                      evaluation_metrics: dict, dataset: SpatioTemporalGraphDataset):
    """
    Runs all visualization steps based on the configuration.

    Loads necessary data files saved by previous steps (static/dynamic features,
    time-series metrics, transition matrix, k-selection results) and uses
    passed-in data (assignments, embeddings, avg metrics, dataset metadata).

    Args:
        config (dict): Configuration dictionary.
        cluster_assignments (dict): Clustering results {t: {node_id_str: cluster_id}}.
        final_embeddings (torch.Tensor): Embeddings output from inference [N, T, D].
        evaluation_metrics (dict): Dictionary of *averaged* evaluation metrics.
        dataset (SpatioTemporalGraphDataset): Dataset object containing metadata (node_ids).
    """
    mode = dataset.mode
    scenario_name = config.get('SCENARIO_NAME', 'Unknown')
    results_dir = config.get('SCENARIO_RESULTS_DIR', f'results/{scenario_name}')
    processed_data_dir = config.get('PROCESSED_DATA_DIR', 'data/processed')
    logging.info(f"--- Starting Visualization Pipeline for {mode} set ---")

    if not config.get('SAVE_PLOTS', True):
        logging.info("SAVE_PLOTS is False in config. Skipping visualization.")
        return

    # --- Prepare Core Data Structures ---
    logging.info("Preparing data for visualization...")
    start_prep_time = time.time()
    try:
        num_nodes, num_timesteps, embedding_dim = final_embeddings.shape
        node_ids = dataset.node_ids # Original node IDs in order

        # Convert assignments dict to DataFrame (needed by multiple plotters)
        cluster_assign_df = format_cluster_assignments(cluster_assignments, node_ids, num_timesteps)

        # Load static features (needed for GIF, composition, embedding coloring)
        static_features_path = os.path.join(processed_data_dir, 'processed_static_features.csv')
        if os.path.exists(static_features_path):
            # Load, ensuring index is string 'building_id'
            static_features_df = pd.read_csv(static_features_path, index_col='building_id')
            static_features_df.index = static_features_df.index.astype(str)
            logging.info(f"Loaded static features: {static_features_df.shape}")
        else:
            logging.warning(f"Static features file not found: {static_features_path}. Some plots may be limited.")
            static_features_df = pd.DataFrame(index=pd.Index([str(n) for n in node_ids], name='building_id')) # Create empty df with index

        # Load dynamic features (needed for average cluster TS plot)
        dynamic_features_path = os.path.join(processed_data_dir, 'processed_dynamic_features.parquet')
        if config.get('VIZ_PLOT_AVG_TS', False) and os.path.exists(dynamic_features_path):
             dynamic_features_df = pd.read_parquet(dynamic_features_path)
             logging.info(f"Loaded dynamic features: {dynamic_features_df.shape}")
        else:
             dynamic_features_df = pd.DataFrame() # Empty df if not needed or not found

        # Smooth embeddings (needed for embedding plots)
        if config.get('VIZ_PLOT_TSNE', False) or config.get('VIZ_PLOT_UMAP', False):
             window_size = config.get('EMBEDDING_SMOOTHING_WINDOW_W', 1)
             logging.info("Smoothing embeddings for visualization...")
             smoothed_embeddings_np = smooth_embeddings(final_embeddings, window_size)
        else:
             smoothed_embeddings_np = None # Not needed

        # Load transition matrix (needed for transition heatmap)
        transition_matrix_path = os.path.join(results_dir, 'transition_matrix.csv')
        if config.get('VIZ_PLOT_TRANSITIONS', False) and os.path.exists(transition_matrix_path):
             transition_matrix_df = pd.read_csv(transition_matrix_path, index_col=0) # Assumes index is cluster_id_t-1
             logging.info("Loaded transition matrix.")
        else:
             transition_matrix_df = None

        # Load K selection results (needed for K selection plot)
        # IMPORTANT: Assumes clustering.py saves this when enabled!
        k_selection_path = os.path.join(results_dir, 'k_selection_results.csv') # Define expected filename
        if config.get('CLUSTER_K_SELECTION_ENABLED', False) and config.get('VIZ_PLOT_K_SELECTION', True) and os.path.exists(k_selection_path):
             k_selection_results_df = pd.read_csv(k_selection_path)
             logging.info("Loaded K selection results.")
        else:
             k_selection_results_df = None

        logging.info(f"Data preparation for visualization finished. Duration: {time.time() - start_prep_time:.2f}s")

    except Exception as e:
        logging.error(f"Error during data preparation for visualization: {e}")
        return # Stop if essential data prep fails

    # --- Generate Plots ---
    # 1. Evaluation Summary Bar Chart (using averaged metrics passed in)
    if config.get('VIZ_PLOT_SUMMARY', True):
        plot_evaluation_summary(config, evaluation_metrics)

    # 2. Time Series Metrics Plots (loads data from CSVs)
    if config.get('VIZ_PLOT_METRIC_TS', True):
        plot_metric_timeseries(config)

    # 3. K Selection Plot (if enabled and data exists)
    if config.get('CLUSTER_K_SELECTION_ENABLED', False) and config.get('VIZ_PLOT_K_SELECTION', True):
         if k_selection_results_df is not None:
             plot_k_selection_metrics(config, k_selection_results_df)
         else:
              logging.warning("K selection enabled but results file not found/loaded. Skipping K plot.")


    # 4. Cluster Evolution GIF
    if config.get('VIZ_GENERATE_GIF', True):
        if not static_features_df.empty and 'lat' in static_features_df.columns:
            plot_cluster_evolution_gif(config, cluster_assign_df.copy(), static_features_df.copy())
        else:
            logging.warning("Skipping GIF plot: Static features or lat/lon missing.")

    # 5. Average Cluster Time Series
    if config.get('VIZ_PLOT_AVG_TS', True):
        if not dynamic_features_df.empty:
            # Ensure 'time_index' column exists in both DataFrames
            for df, name in [(cluster_assign_df, 'cluster_assign_df'), (dynamic_features_df, 'dynamic_features_df')]:
                if 'time_index' not in df.columns:
                    # Try to find a similar column
                    for alt in ['t', 'step', 'timestamp']:
                        if alt in df.columns:
                            df.rename(columns={alt: 'time_index'}, inplace=True)
                            logging.warning(f"Renamed column '{alt}' to 'time_index' in {name}.")
                            break
                    else:
                        raise KeyError(f"'time_index' column not found in {name}. Columns: {df.columns.tolist()}")
            # Ensure building_id is string and time_index is int for both DataFrames
            cluster_assign_df['building_id'] = cluster_assign_df['building_id'].astype(str)
            dynamic_features_df['building_id'] = dynamic_features_df['building_id'].astype(str)
            # Robustly convert time_index to int if it's a time type
            if not pd.api.types.is_integer_dtype(dynamic_features_df['time_index']):
                # If time_index is datetime.time or similar, map each unique time to a step index
                unique_times = sorted(dynamic_features_df['time_index'].unique())
                time_to_idx = {t: i for i, t in enumerate(unique_times)}
                dynamic_features_df['time_index'] = dynamic_features_df['time_index'].map(time_to_idx)
            cluster_assign_df['time_index'] = cluster_assign_df['time_index'].astype(int)
            dynamic_features_df['time_index'] = dynamic_features_df['time_index'].astype(int)
            # Print unique values for debugging
            print("cluster_assign_df time_index:", cluster_assign_df['time_index'].unique())
            print("dynamic_features_df time_index:", dynamic_features_df['time_index'].unique())
            print("cluster_assign_df building_id:", cluster_assign_df['building_id'].unique()[:5])
            print("dynamic_features_df building_id:", dynamic_features_df['building_id'].unique()[:5])
            # Align cluster assignments and dynamic features by building_id and time_index
            merged_df = pd.merge(dynamic_features_df, cluster_assign_df[['building_id', 'time_index', 'cluster_id']],
                                 on=['building_id', 'time_index'], how='inner')
            if merged_df.empty:
                logging.warning("Merged DataFrame for average cluster time series is empty. Check ID and time alignment.")
            else:
                # Pass dynamic_features_df as the third argument to avoid AttributeError
                plot_average_cluster_timeseries(config, merged_df, dynamic_features_df)
        else:
             logging.warning("Skipping average cluster TS plot: Dynamic features not loaded.")

    # 6. Cluster Composition
    if config.get('VIZ_PLOT_COMPOSITION', True):
        if not static_features_df.empty:
             plot_cluster_composition(config, cluster_assign_df.copy(), static_features_df.copy())
        else:
             logging.warning("Skipping composition plot: Static features not loaded.")

    # 7. Transition Heatmap
    if config.get('VIZ_PLOT_TRANSITIONS', True):
        if transition_matrix_df is not None:
             plot_transition_heatmap(config, transition_matrix_df)
        else:
             logging.warning("Skipping transition heatmap: Transition matrix file not found/loaded.")

    # 8. Embedding Projections (t-SNE/UMAP)
    if config.get('VIZ_PLOT_TSNE', True) or config.get('VIZ_PLOT_UMAP', False):
        if smoothed_embeddings_np is not None:
            time_indices_to_plot = []
            ts_setting = config.get('VIZ_EMBEDDING_TIMESTEPS', ['middle']) # Default to middle
            if 'first' in ts_setting: time_indices_to_plot.append(0)
            if 'middle' in ts_setting: time_indices_to_plot.append(num_timesteps // 2)
            if 'last' in ts_setting: time_indices_to_plot.append(num_timesteps - 1)
            # Add specific indices if provided as integers
            time_indices_to_plot.extend([t for t in ts_setting if isinstance(t, int) and 0 <= t < num_timesteps])
            time_indices_to_plot = sorted(list(set(time_indices_to_plot))) # Unique & sorted

            logging.info(f"Generating embedding projections for time indices: {time_indices_to_plot}")
            for t_idx in time_indices_to_plot:
                # Extract labels for this timestep
                labels_t = cluster_assign_df.loc[cluster_assign_df['time_index'] == t_idx].set_index('building_id')['cluster_id']
                # Reindex labels to match the node_ids order and fill missing with -1
                labels_t_array = labels_t.reindex([str(n) for n in node_ids], fill_value=-1).values

                # Plot t-SNE
                if config.get('VIZ_PLOT_TSNE', True):
                     plot_embedding_projection(config, smoothed_embeddings_np[:, t_idx, :], labels_t_array,
                                               node_ids, static_features_df.copy(), t_idx, method='tsne')
                # Plot UMAP
                if config.get('VIZ_PLOT_UMAP', False):
                     plot_embedding_projection(config, smoothed_embeddings_np[:, t_idx, :], labels_t_array,
                                               node_ids, static_features_df.copy(), t_idx, method='umap')
        else:
             logging.warning("Skipping embedding plots: Smoothed embeddings not available.")


    # --- Call other optional plotting functions ---
    # if config.get('VIZ_PLOT_GEO_SHAPES', False): plot_geographic_cluster_shape(...)
    # if config.get('VIZ_PLOT_SANKEY', False): plot_sankey_transitions(...)
    # if config.get('VIZ_PLOT_GRAPH', False): plot_graph_structure(...)


    logging.info(f"--- Visualization Pipeline Finished ---")

    # --- Export results to Neo4j ---
    try:
        from src.neo4j_export import Neo4jExporter
        exporter = Neo4jExporter()
        exporter.export_all(static_features_df, cluster_assign_df, dynamic_features_df)
        exporter.close()
        logging.info("Results exported to Neo4j successfully.")
    except Exception as e:
        logging.error(f"Failed to export results to Neo4j: {e}")


# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    # This block requires outputs from previous steps (or extensive mocking)
    logging.info("--- Testing visualization script standalone ---")

    # Minimal config for testing
    config_test = {
        'SCENARIO_NAME': 'Viz_Standalone_Test',
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Viz_Standalone_Test'),
        'PROCESSED_DATA_DIR': os.path.join('..', 'data', 'processed'),
        'NUM_CLUSTERS_K': 4,
        'SAVE_PLOTS': True,
        'VIZ_PLOT_SUMMARY': True,
        'VIZ_PLOT_METRIC_TS': True, # Assumes dummy CSVs exist from viz_metrics test
        'VIZ_GENERATE_GIF': False, # Disable GIF for faster test
        'VIZ_PLOT_AVG_TS': True,
        'VIZ_PLOT_COMPOSITION': True,
        'VIZ_PLOT_TRANSITIONS': True, # Assumes dummy transition matrix exists
        'VIZ_PLOT_TSNE': True,
        'VIZ_EMBEDDING_TIMESTEPS': ['middle'],
        'SCALE_EMBEDDINGS_BEFORE_TSNE': True,
        'CLUSTER_K_SELECTION_ENABLED': False, # Disable K selection plot for this test
        'TIME_SERIES_FREQUENCY': '15min',
        'STATIC_CATEGORICAL_COLS': ['building_function', 'has_solar'] # Example for composition plot
    }
    os.makedirs(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'plots'), exist_ok=True)

    # Create minimal dummy data needed as input arguments
    num_nodes_test = 30
    num_timesteps_test = 10
    dummy_embeddings_tensor = torch.randn(num_nodes_test, num_timesteps_test, 16)
    dummy_assignments = {}
    for t in range(num_timesteps_test):
         dummy_assignments[t] = {f'B{i:03d}': np.random.randint(0, config_test['NUM_CLUSTERS_K']) for i in range(num_nodes_test)}

    dummy_eval_metrics = {'avg_ari': 0.8, 'avg_silhouette_score': 0.5} # Minimal example

    class DummyDataset: # Minimal dataset mock
         node_ids = [f'B{i:03d}' for i in range(num_nodes_test)]
    dummy_dataset = DummyDataset()

    # Create dummy files that run_visualization expects to load
    # (Static features are needed for several plots)
    dummy_static_data = {'building_id': dummy_dataset.node_ids,
                         'lat': np.random.rand(num_nodes_test)*0.1+52.0,
                         'lon': np.random.rand(num_nodes_test)*0.1+4.0,
                         'building_function': ['residential']*(num_nodes_test//2) + ['non_residential']*(num_nodes_test - num_nodes_test//2),
                         'has_solar': [True, False] * (num_nodes_test//2)}
    dummy_static_df_vis = pd.DataFrame(dummy_static_data).set_index('building_id')
    os.makedirs(config_test['PROCESSED_DATA_DIR'], exist_ok=True)
    static_path_test = os.path.join(config_test['PROCESSED_DATA_DIR'], 'processed_static_features.csv')
    dummy_static_df_vis.to_csv(static_path_test)
    print(f"Created dummy static features at {static_path_test}")

    # Dummy dynamic features (needed for avg ts plot)
    dynamic_records = []
    for t in range(num_timesteps_test):
        for i in range(num_nodes_test):
            dynamic_records.append({'time_index': t, 'building_id': f'B{i:03d}', 'net_load': np.random.randn()*10+20})
    dummy_dynamic_df_vis = pd.DataFrame(dynamic_records)
    dynamic_path_test = os.path.join(config_test['PROCESSED_DATA_DIR'], 'processed_dynamic_features.parquet')
    dummy_dynamic_df_vis.to_parquet(dynamic_path_test)
    print(f"Created dummy dynamic features at {dynamic_path_test}")

    # Dummy transition matrix (needed for heatmap)
    dummy_transition_matrix_vis = pd.DataFrame(np.random.rand(config_test['NUM_CLUSTERS_K'], config_test['NUM_CLUSTERS_K']) * 5,
                                               index=range(config_test['NUM_CLUSTERS_K']), columns=range(config_test['NUM_CLUSTERS_K']))
    transition_path_test = os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'transition_matrix.csv')
    dummy_transition_matrix_vis.to_csv(transition_path_test)
    print(f"Created dummy transition matrix at {transition_path_test}")

    # Dummy timeseries metrics files (needed for metric ts plot) - use files from viz_metrics test if run prior
    # Or create minimal versions here:
    pd.DataFrame({'time_index_t':[1,2],'ari':[0.7,0.75]}).to_csv(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'stability_metrics_timeseries.csv'), index=False)
    pd.DataFrame({'time_index':[0,1],'silhouette_score':[0.5,0.55]}).to_csv(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'quality_metrics_timeseries.csv'), index=False)
    pd.DataFrame({'time_index':[0,0,1,1],'cluster_id':[0,1,0,1],'ssr':[0.6,0.7,0.65,0.75]}).to_csv(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'synergy_metrics_timeseries.csv'), index=False)


    print("\n--- Running run_visualization with dummy data ---")
    try:
        run_visualization(config_test, dummy_assignments, dummy_embeddings_tensor, dummy_eval_metrics, dummy_dataset)
        print(f"\n--- Standalone visualization test finished. Check plots in {os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'plots')} ---")
    except Exception as e:
        logging.exception(f"Error during standalone visualization test: {e}")
# src/viz_clusters.py

import numpy as np
import pandas as pd
import os
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm

# Optional imports (need installation: pip install imageio networkx shapely plotly)
try:
    import imageio.v2 as imageio # Use v2 for consistent API
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logging.debug("imageio library not found. GIF generation will be unavailable.")
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.debug("networkx library not found. Graph structure plotting will be unavailable.")
try:
    from shapely.geometry import MultiPoint
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logging.debug("shapely library not found. Geographic cluster shape plotting will be unavailable.")
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.debug("plotly library not found. Sankey diagram plotting will be unavailable.")

# Setup basic logging (configure in main script usually)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- GIF Plot (Iterative Writing) ---
def plot_cluster_evolution_gif(config: dict, cluster_assign_df: pd.DataFrame, static_features_df: pd.DataFrame):
    """
    Generates a GIF showing cluster assignments on a map over time using iterative writing.

    Args:
        config (dict): Configuration dictionary. Expected keys: 'SCENARIO_RESULTS_DIR',
                       'NUM_CLUSTERS_K', 'VIZ_GIF_PLOT_EVERY_N_STEPS', 'VIZ_GIF_FRAME_DURATION'.
        cluster_assign_df (pd.DataFrame): Long-format DF with ['time_index', 'building_id', 'cluster_id'].
                                          'building_id' should be string type.
        static_features_df (pd.DataFrame): DF with static features, indexed by 'building_id' (string),
                                           must include 'lat', 'lon'.
    """
    if not IMAGEIO_AVAILABLE:
        logger.warning("Skipping GIF generation: imageio library not found. Install with 'pip install imageio'")
        return
    if cluster_assign_df.empty:
        logger.warning("Skipping GIF generation: Cluster assignment DataFrame is empty.")
        return
    if not all(col in static_features_df.columns for col in ['lat', 'lon']):
         logger.warning("Skipping GIF generation: Static features DataFrame missing 'lat' or 'lon'.")
         return

    logger.info("Generating cluster evolution GIF (iterative write)...")
    start_time = time.time()

    # --- Setup Paths ---
    results_dir = config['SCENARIO_RESULTS_DIR']
    plot_dir = os.path.join(results_dir, 'plots')
    gif_path = os.path.join(plot_dir, 'cluster_evolution.gif')
    temp_frame_dir = os.path.join(plot_dir, 'temp_frames_gif')
    os.makedirs(temp_frame_dir, exist_ok=True)

    # --- Prepare Data ---
    static_features_df.index = static_features_df.index.astype(str)
    cluster_assign_df['building_id'] = cluster_assign_df['building_id'].astype(str)
    # Merge coordinates safely
    plot_data = pd.merge(cluster_assign_df, static_features_df[['lat', 'lon']].reset_index(), # Use reset_index if static has building_id index
                         on='building_id', how='left')

    # Handle missing coordinates
    if plot_data[['lat', 'lon']].isnull().values.any():
        missing_coords_count = plot_data['lat'].isnull().sum()
        plot_data.dropna(subset=['lat', 'lon'], inplace=True)
        logger.warning(f"GIF: Missing coordinates for {missing_coords_count} building assignments. These points omitted.")

    if plot_data.empty:
        logger.error("GIF: No data with valid coordinates available for plotting.")
        try: shutil.rmtree(temp_frame_dir) # Clean up if empty
        except Exception: pass
        return

    # --- Plotting Parameters ---
    time_indices = sorted(plot_data['time_index'].unique())
    num_clusters = config.get('NUM_CLUSTERS_K', 6) # Use the K from clustering run
    try:
        colors = cm.get_cmap('viridis', num_clusters)
    except ValueError: # Handle K=1 case
        colors = cm.get_cmap('viridis')

    plot_every_n_steps = config.get('VIZ_GIF_PLOT_EVERY_N_STEPS', 5)
    plot_time_indices = time_indices[::plot_every_n_steps]
    if not plot_time_indices: plot_time_indices = time_indices[:1] # Ensure at least one frame

    # Consistent plot bounds
    min_lon, max_lon = plot_data['lon'].min(), plot_data['lon'].max()
    min_lat, max_lat = plot_data['lat'].min(), plot_data['lat'].max()
    # Add buffer, handle cases where range is zero
    lon_buffer = (max_lon - min_lon) * 0.05 if max_lon > min_lon else 0.1
    lat_buffer = (max_lat - min_lat) * 0.05 if max_lat > min_lat else 0.1


    # --- Generate Frames and Write GIF Iteratively ---
    frame_paths = []
    logging.info(f"GIF: Generating and writing {len(plot_time_indices)} frames...")
    try:
        with imageio.get_writer(gif_path, mode='I',
                                duration=config.get('VIZ_GIF_FRAME_DURATION', 0.3)*1000, # duration in ms
                                loop=0) as writer:
            pbar_gif = tqdm(plot_time_indices, desc="Generating/Writing GIF", leave=False, unit="frame")
            for t_idx in pbar_gif:
                frame_path = os.path.join(temp_frame_dir, f"frame_{t_idx:05d}.png")
                data_t = plot_data[plot_data['time_index'] == t_idx]

                if data_t.empty: continue

                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(data_t['lon'], data_t['lat'],
                                     c=data_t['cluster_id'], cmap=colors,
                                     vmin=0, vmax=max(0, num_clusters - 1),
                                     s=20, alpha=0.8, zorder=2)

                # Format timestamp string
                try:
                    freq_minutes = int(config.get('TIME_SERIES_FREQUENCY', '15min').replace('min',''))
                    total_minutes = t_idx * freq_minutes
                    hours, minutes = divmod(total_minutes, 60)
                    timestamp_str = f"{hours:02d}:{minutes:02d}:00"
                except:
                    timestamp_str = f"Index {t_idx}"

                ax.set_title(f"Building Clusters at Time: {timestamp_str}")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_xlim(min_lon - lon_buffer, max_lon + lon_buffer)
                ax.set_ylim(min_lat - lat_buffer, max_lat + lat_buffer)
                ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

                plt.tight_layout()
                plt.savefig(frame_path, dpi=100) # Adjust DPI if needed
                plt.close(fig)

                # Append frame to GIF
                writer.append_data(imageio.imread(frame_path))
                frame_paths.append(frame_path) # Keep track for cleanup

        logging.info(f"GIF saved successfully to {gif_path}")

    except MemoryError:
         logging.error(f"MemoryError creating GIF, even with iterative writing. "
                       f"Consider increasing VIZ_GIF_PLOT_EVERY_N_STEPS further or reducing plot resolution/size.")
    except Exception as e:
         logging.error(f"Failed during GIF generation/writing: {e}")
    finally:
        # Clean up temporary frames
        logging.info("GIF: Cleaning up temporary frame files...")
        import shutil # Use shutil for robust removal
        try:
            if os.path.exists(temp_frame_dir):
                 shutil.rmtree(temp_frame_dir)
        except Exception as e_clean:
            logging.warning(f"GIF: Could not remove temporary frame directory {temp_frame_dir}: {e_clean}")

    end_time = time.time()
    logging.info(f"GIF generation finished. Duration: {end_time - start_time:.2f} seconds")


# --- Cluster Time Series Profile ---
def plot_average_cluster_timeseries(config: dict, cluster_assign_df: pd.DataFrame, dynamic_features_df: pd.DataFrame):
    """Plots the average net load time series for each cluster."""
    logging.info("Generating average cluster time series plot...")
    plot_dir = os.path.join(config['SCENARIO_RESULTS_DIR'], 'plots')
    save_path = os.path.join(plot_dir, 'average_cluster_timeseries.png')

    if 'net_load' not in dynamic_features_df.columns:
        logging.error("Cannot plot average time series: 'net_load' missing from dynamic features.")
        return
    if cluster_assign_df.empty:
        logging.warning("Cannot plot average time series: Cluster assignments empty.")
        return

    # Ensure consistent types for merge
    dynamic_features_df['building_id'] = dynamic_features_df['building_id'].astype(str)
    cluster_assign_df['building_id'] = cluster_assign_df['building_id'].astype(str)
    # Add time index if missing from dynamic df
    if 'time_index' not in dynamic_features_df.columns:
         dynamic_features_df['time_index'] = dynamic_features_df.groupby('building_id').cumcount()

    # Merge cluster assignments with dynamic data
    try:
         merged_df = pd.merge(dynamic_features_df[['building_id', 'time_index', 'net_load']],
                              cluster_assign_df[['building_id', 'time_index', 'cluster_id']],
                              on=['building_id', 'time_index'], how='inner')
    except Exception as e:
        logging.error(f"Error merging data for average TS plot: {e}")
        return

    if merged_df.empty:
        logging.warning("No matching data found after merging assignments and dynamic features for TS plot.")
        return

    # Calculate mean and std deviation per cluster per time step
    agg_ts = merged_df.groupby(['time_index', 'cluster_id'])['net_load'].agg(['mean', 'std']).reset_index()

    # Plot
    try:
        num_clusters = agg_ts['cluster_id'].nunique()
        # Handle case where K=1 or fewer clusters are actually present
        if num_clusters < 1:
             logging.warning("No clusters found in aggregated data. Skipping TS plot.")
             return
        palette = sns.color_palette("viridis", max(1, num_clusters)) # Ensure palette size >= 1

        plt.figure(figsize=(15, 7))
        sns.lineplot(data=agg_ts, x='time_index', y='mean', hue='cluster_id',
                     palette=palette, legend='full', marker='.')

        # Optional: Add shaded standard deviation
        # sns.lineplot(data=agg_ts, x='time_index', y='mean', hue='cluster_id', palette=palette, legend=False, alpha=0.3, errorbar='sd') # Simpler way with seaborn >= 0.12?

        plt.title(f"Average Net Load Time Series per Cluster - {config.get('SCENARIO_NAME', '')}")
        plt.xlabel("Time Index (e.g., 15-min intervals)")
        plt.ylabel("Average Net Load (kW)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info(f"Saved average cluster time series plot to: {save_path}")

    except Exception as e:
        logging.error(f"Failed to generate average cluster time series plot: {e}")


# --- Cluster Composition ---
def plot_cluster_composition(config: dict, cluster_assign_df: pd.DataFrame, static_features_df: pd.DataFrame):
    """Plots the distribution of static features within each cluster for the middle time step."""
    logging.info("Generating cluster composition plots...")
    plot_dir = os.path.join(config['SCENARIO_RESULTS_DIR'], 'plots', 'cluster_composition')
    os.makedirs(plot_dir, exist_ok=True)

    if cluster_assign_df.empty or static_features_df.empty:
        logging.warning("Cannot plot composition: Cluster assignments or static features empty.")
        return

    # Use assignments from the middle time step
    target_time_index = cluster_assign_df['time_index'].max() // 2
    assignments_t = cluster_assign_df[cluster_assign_df['time_index'] == target_time_index]
    logging.info(f"Analyzing composition for time index {target_time_index}")

    if assignments_t.empty:
        logging.warning(f"No assignments found for time index {target_time_index}. Skipping composition plots.")
        return

    # Merge with static features
    static_features_df.index = static_features_df.index.astype(str)
    assignments_t['building_id'] = assignments_t['building_id'].astype(str)
    # Use reset_index() on static_features_df if its index is building_id
    merged_df = pd.merge(assignments_t, static_features_df.reset_index(), on='building_id', how='left')

    # Identify potential categorical/boolean features to plot
    potential_cat_cols = config.get('STATIC_CATEGORICAL_COLS', [])
    # Also consider boolean flags explicitly
    bool_cols = ['has_solar', 'has_battery']
    plot_cols = [col for col in potential_cat_cols + bool_cols if col in merged_df.columns]
    # Exclude high-cardinality categoricals for clarity
    plot_cols = [col for col in plot_cols if merged_df[col].nunique() < 15]

    logging.info(f"Plotting composition for features: {plot_cols}")

    for feature in plot_cols:
        if merged_df[feature].isnull().all():
             logging.warning(f"Skipping composition plot for '{feature}': All values are null.")
             continue
        try:
            plt.figure(figsize=(10, 6))
            # Calculate percentage within each cluster
            # Normalize calculates proportion, mul(100) converts to percent
            composition = merged_df.groupby('cluster_id')[feature].value_counts(normalize=True).mul(100).unstack(fill_value=0)

            # Choose colormap based on number of categories in the feature
            num_cats = len(composition.columns)
            cmap = cm.get_cmap('viridis', max(num_cats, 2)) # Ensure cmap has enough colors

            composition.plot(kind='bar', stacked=True, colormap=cmap, alpha=0.85)

            plt.title(f"Composition by '{feature}' within Clusters (t={target_time_index})")
            plt.xlabel("Cluster ID")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=0)
            plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            save_path = os.path.join(plot_dir, f'composition_{feature}.png')
            plt.savefig(save_path, dpi=120)
            plt.close()
            logging.info(f"Saved composition plot for '{feature}' to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to generate composition plot for feature '{feature}': {e}")


# --- Transition Matrix ---
def plot_transition_heatmap(config: dict, transition_matrix_df: pd.DataFrame):
    """Plots the average cluster transition matrix as a heatmap."""
    if transition_matrix_df is None or transition_matrix_df.empty:
        logging.warning("Skipping transition heatmap: Transition matrix DataFrame is empty or None.")
        return

    logging.info("Generating cluster transition heatmap...")
    plot_dir = os.path.join(config['SCENARIO_RESULTS_DIR'], 'plots')
    save_path = os.path.join(plot_dir, 'transition_heatmap.png')

    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix_df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, square=True)
        plt.title(f"Average Cluster Transition Counts (t-1 to t) - {config.get('SCENARIO_NAME', '')}")
        plt.xlabel("Cluster ID at time 't'")
        plt.ylabel("Cluster ID at time 't-1'")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        logging.info(f"Saved transition heatmap to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to generate transition heatmap: {e}")


# --- Placeholder/Optional Functions ---

def plot_geographic_cluster_shape(config: dict, cluster_assign_df: pd.DataFrame, static_features_df: pd.DataFrame, time_index: int):
    """Placeholder: Plots geographic cluster shapes (e.g., convex hulls) for a time step."""
    if not SHAPELY_AVAILABLE:
        logging.debug("Skipping geographic shapes plot: shapely library not found.")
        return
    logging.info(f"Generating geographic cluster shapes plot for t={time_index} (Placeholder)...")
    # Implementation would involve:
    # 1. Merging data for the specific time_index.
    # 2. Grouping by cluster_id.
    # 3. For each cluster group, create shapely MultiPoint from lat/lon.
    # 4. Calculate convex_hull for each MultiPoint.
    # 5. Plot hulls on a map using matplotlib.patches or geopandas.
    pass

def plot_sankey_transitions(config: dict, cluster_assign_df: pd.DataFrame, t_start: int, t_end: int):
    """Placeholder: Generates a Sankey diagram showing cluster transitions."""
    if not PLOTLY_AVAILABLE:
        logging.debug("Skipping Sankey plot: plotly library not found.")
        return
    logging.info(f"Generating Sankey transition plot from t={t_start} to t={t_end} (Placeholder)...")
    # Implementation would involve:
    # 1. Filtering cluster_assign_df for t_start and t_end.
    # 2. Merging the two time steps on building_id.
    # 3. Calculating counts of transitions (cluster_id_start -> cluster_id_end).
    # 4. Formatting data for plotly.graph_objects.Sankey.
    pass

def plot_graph_structure(config: dict, graph_data, cluster_labels: dict, time_index: int):
     """Placeholder: Plots the graph structure colored by clusters."""
     if not NETWORKX_AVAILABLE:
         logging.debug("Skipping graph structure plot: networkx library not found.")
         return
     logging.info(f"Generating graph structure plot for t={time_index} (Placeholder)...")
     # Implementation would involve:
     # 1. Getting edge_index and pos from graph_data (passed from main viz script).
     # 2. Creating a networkx graph from edge_index.
     # 3. Getting cluster labels for the given time_index.
     # 4. Getting node positions (pos attribute).
     # 5. Drawing the graph using nx.draw or nx.draw_networkx, passing pos and node_color based on labels.
     pass


# --- Standalone Test Block ---
if __name__ == '__main__':
    logger.info("--- Testing viz_clusters.py standalone ---")

    # Create dummy dataframes similar to what would be passed
    num_nodes_test = 50
    num_timesteps_test = 10 # Keep low for testing
    num_clusters_test = 4

    # Dummy Cluster Assignments
    assign_records = []
    current_assignments = np.random.randint(0, num_clusters_test, num_nodes_test)
    for t in range(num_timesteps_test):
        # Simulate some minor changes/churn
        if t > 0:
            change_indices = np.random.choice(num_nodes_test, size=5, replace=False)
            current_assignments[change_indices] = np.random.randint(0, num_clusters_test, size=5)
        for i in range(num_nodes_test):
            assign_records.append({'time_index': t, 'building_id': f'B{i:03d}', 'cluster_id': current_assignments[i]})
    dummy_assign_df = pd.DataFrame(assign_records)

    # Dummy Static Features
    dummy_static_data = {
        'building_id': [f'B{i:03d}' for i in range(num_nodes_test)],
        'lat': np.random.rand(num_nodes_test) * 0.1 + 52.0,
        'lon': np.random.rand(num_nodes_test) * 0.1 + 4.0,
        'building_function': ['residential'] * (num_nodes_test // 2) + ['non_residential'] * (num_nodes_test - num_nodes_test // 2),
        'has_solar': np.random.choice([True, False], num_nodes_test)
    }
    dummy_static_df = pd.DataFrame(dummy_static_data).set_index('building_id')

    # Dummy Dynamic Features
    dynamic_records = []
    for t in range(num_timesteps_test):
         for i in range(num_nodes_test):
              dynamic_records.append({'time_index': t, 'building_id': f'B{i:03d}', 'net_load': np.random.randn() * 10 + 20})
    dummy_dynamic_df = pd.DataFrame(dynamic_records)

    # Dummy Transition Matrix
    dummy_transition_matrix = pd.DataFrame(np.random.rand(num_clusters_test, num_clusters_test) * 10,
                                           index=range(num_clusters_test), columns=range(num_clusters_test))

    # Dummy Config
    config_test = {
        'SCENARIO_NAME': 'Cluster_Viz_Test',
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Cluster_Viz_Test'),
        'NUM_CLUSTERS_K': num_clusters_test,
        'VIZ_GIF_PLOT_EVERY_N_STEPS': 2, # Plot fewer frames for test
        'VIZ_GIF_FRAME_DURATION': 0.5,
        'TIME_SERIES_FREQUENCY': '15min',
        'STATIC_CATEGORICAL_COLS': ['building_function', 'has_solar'] # Specify for composition plot
    }
    os.makedirs(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'plots', 'cluster_composition'), exist_ok=True)

    # --- Call plotting functions ---
    logger.info("--- Running Standalone Plot Function Tests ---")
    if IMAGEIO_AVAILABLE:
        plot_cluster_evolution_gif(config_test, dummy_assign_df.copy(), dummy_static_df.copy())
    else: logger.warning("Skipping GIF test - imageio not installed.")

    plot_average_cluster_timeseries(config_test, dummy_assign_df.copy(), dummy_dynamic_df.copy())
    plot_cluster_composition(config_test, dummy_assign_df.copy(), dummy_static_df.copy())
    plot_transition_heatmap(config_test, dummy_transition_matrix.copy())

    logger.info("--- Standalone test complete for viz_clusters.py ---")
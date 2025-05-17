# src/viz_embedding.py

import numpy as np
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Optional imports (need installation: pip install umap-learn plotly)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.debug("umap-learn library not found. UMAP plotting will be unavailable.")
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.debug("plotly library not found. Interactive plotting will be unavailable.")

# Setup basic logging (configure in main script usually)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_embedding_projection(config: dict,
                              embeddings: np.ndarray,
                              labels: np.ndarray,
                              node_ids: list,
                              static_features_df: pd.DataFrame,
                              time_index: int,
                              method: str = 'tsne'):
    """
    Generates a 2D projection (t-SNE or UMAP) plot of node embeddings for
    a specific time step, saving both static and optionally interactive plots.

    Args:
        config (dict): Configuration dictionary, expected to contain keys like:
                       'SCENARIO_RESULTS_DIR', 'SEED', 'NUM_CLUSTERS_K',
                       'SCALE_EMBEDDINGS_BEFORE_TSNE', 'VIZ_INTERACTIVE_PLOTS'.
        embeddings (np.ndarray): Node embeddings for the time step [num_nodes, embedding_dim].
                                  Node order must match node_ids.
        labels (np.ndarray): Cluster labels for the nodes [num_nodes]. Assumed to be integer labels.
        node_ids (list): List of original node IDs corresponding to the embedding rows.
        static_features_df (pd.DataFrame): DataFrame with static features, indexed by node_id (str).
                                           Used for hover info in interactive plots.
        time_index (int): The time step index being plotted.
        method (str): Projection method ('tsne' or 'umap'). Defaults to 'tsne'.
    """
    method_str = method.lower()
    if method_str not in ['tsne', 'umap']:
        logger.error(f"Unsupported projection method: {method}. Use 'tsne' or 'umap'.")
        return
    if method_str == 'umap' and not UMAP_AVAILABLE:
        logger.error("UMAP method requested, but umap-learn library not found. Skipping plot.")
        return

    # --- Setup Paths ---
    plot_base_dir = os.path.join(config.get('SCENARIO_RESULTS_DIR', 'results/default'), 'plots')
    projection_plot_dir = os.path.join(plot_base_dir, 'embedding_projections')
    os.makedirs(projection_plot_dir, exist_ok=True)

    # Format timestamp for filename (HHMMSS)
    try:
        freq_minutes = int(config.get('TIME_SERIES_FREQUENCY', '15min').replace('min',''))
        total_minutes = time_index * freq_minutes
        hours = total_minutes // 60
        minutes = total_minutes % 60
        timestamp = f"{hours:02d}{minutes:02d}00"
        timestamp_title = f"{hours:02d}:{minutes:02d}:00"
    except:
        timestamp = f"idx{time_index:05d}" # Fallback if frequency isn't easily parsed
        timestamp_title = f"Index {time_index}"

    base_filename = f'embedding_{method_str}_t{timestamp}'

    logger.info(f"Generating {method_str.upper()} plot for time step {time_index} ({timestamp_title})...")

    # --- Input Validation ---
    n_nodes = embeddings.shape[0]
    if n_nodes < 2:
        logger.warning(f"Skipping {method_str.upper()} plot for t={time_index}: Only {n_nodes} node(s).")
        return
    if len(labels) != n_nodes or len(node_ids) != n_nodes:
         logger.error(f"Shape mismatch for t={time_index}: Embeddings ({n_nodes}), Labels ({len(labels)}), Node IDs ({len(node_ids)}). Skipping plot.")
         return

    # --- Preprocessing ---
    # Filter out nodes with invalid labels if any (-1 or NaN)
    valid_mask = (labels != -1) & (~np.isnan(labels))
    n_valid_nodes_initial = np.sum(valid_mask)
    if n_valid_nodes_initial < 2:
        logger.warning(f"Skipping {method_str.upper()} plot for t={time_index}: Fewer than 2 valid labeled nodes ({n_valid_nodes_initial}).")
        return

    embeddings_valid = embeddings[valid_mask]
    labels_valid = labels[valid_mask].astype(int) # Ensure integer labels for coloring/stats
    node_ids_valid = np.array(node_ids)[valid_mask]
    n_valid_nodes = embeddings_valid.shape[0] # Should match n_valid_nodes_initial

    # Check if enough unique labels for meaningful plot
    unique_labels_valid = np.unique(labels_valid)
    n_unique_labels = len(unique_labels_valid)
    if n_unique_labels < 1:
        logger.warning(f"Skipping {method_str.upper()} plot for t={time_index}: No valid unique labels found.")
        return
    if n_unique_labels < 2:
        logger.info(f"Plotting {method_str.upper()} for t={time_index} with only 1 cluster label ({unique_labels_valid[0]}).")


    # --- Scaling (Optional) ---
    if config.get('SCALE_EMBEDDINGS_BEFORE_TSNE', True):
        logger.debug(f"Scaling {n_valid_nodes} embeddings before {method_str.upper()}.")
        try:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_valid)
        except Exception as e:
            logger.error(f"Error scaling embeddings for {method_str.upper()} plot: {e}. Using unscaled.")
            embeddings_scaled = embeddings_valid
    else:
        embeddings_scaled = embeddings_valid

    # --- Projection ---
    try:
        if method_str == 'umap':
            # UMAP specific parameters
            n_neighbors = min(15, n_valid_nodes - 1) if n_valid_nodes > 1 else 1
            min_dist = 0.1
            reducer = umap.UMAP(n_components=2,
                                n_neighbors=max(1, n_neighbors), # Ensure n_neighbors >= 1
                                min_dist=min_dist,
                                metric='euclidean',
                                random_state=config.get('SEED', 42))
        else: # TSNE
            # TSNE specific parameters
            perplexity = min(30.0, n_valid_nodes - 1.0) if n_valid_nodes > 1 else 1.0
            reducer = TSNE(n_components=2,
                           perplexity=max(1.0, perplexity), # Ensure perplexity >= 1
                           random_state=config.get('SEED', 42),
                           n_iter=config.get('VIZ_TSNE_N_ITER', 300), # Allow config control
                           init='pca', learning_rate='auto')

        logger.info(f"Running {reducer.__class__.__name__} on {n_valid_nodes} nodes (t={time_index})...")
        embeddings_2d = reducer.fit_transform(embeddings_scaled)
        logger.info(f"Projection finished for t={time_index}.")

    except ValueError as e: # Catch specific errors like perplexity > n_samples-1
        logger.error(f"ValueError during {method_str.upper()} for t={time_index}: {e}. Skipping plot.")
        return
    except Exception as e:
         logger.error(f"Error during {method_str.upper()} calculation for t={time_index}: {e}. Skipping plot.")
         return

    # --- Create Plotting DataFrame ---
    # Ensures index alignment if static_features_df index isn't perfectly ordered
    plot_df = pd.DataFrame({
        'dim1': embeddings_2d[:, 0],
        'dim2': embeddings_2d[:, 1],
        'cluster_id': labels_valid, # Keep as int for matplotlib cmap, convert later for plotly category
        'node_id': node_ids_valid # Use valid node IDs corresponding to embeddings
    })
    # Merge static features safely
    static_features_df.index = static_features_df.index.astype(str) # Ensure index is string
    plot_df = pd.merge(plot_df, static_features_df, left_on='node_id', right_index=True, how='left')


    # --- Plotting ---
    num_clusters = config.get('NUM_CLUSTERS_K', 6) # K used for clustering
    # Use a qualitative map if few clusters, sequential if many? Viridis is safe.
    try:
        colors = cm.get_cmap('viridis', num_clusters)
    except ValueError: # Handle case where num_clusters < 1 (shouldn't happen here)
        colors = cm.get_cmap('viridis')


    # ** 1. Matplotlib Static Plot **
    try:
        fig, ax = plt.subplots(figsize=(12, 10)) # Slightly larger figure
        scatter = ax.scatter(plot_df['dim1'], plot_df['dim2'],
                             c=plot_df['cluster_id'], # Color by numerical labels
                             cmap=colors,
                             vmin=0, vmax=max(0, num_clusters - 1), # Handle K=1 case for vmin/vmax
                             s=20, alpha=0.8, zorder=2) # Points on top of grid

        ax.set_title(f"{method_str.upper()} of Embeddings at Time Index: {time_index} ({timestamp_title})")
        ax.set_xlabel(f"{method_str.upper()} Dimension 1")
        ax.set_ylabel(f"{method_str.upper()} Dimension 2")
        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)

        # Add colorbar legend (handle case with single label)
        if n_unique_labels > 1 :
             bounds = np.arange(num_clusters + 1) - 0.5
             norm = plt.Normalize(vmin=-0.5, vmax=num_clusters - 0.5)
             sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
             sm.set_array([]) # Necessary for colorbar
             cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(num_clusters), boundaries=bounds, label='Cluster ID')
             # cbar = plt.colorbar(scatter, label='Cluster ID', ticks=np.arange(num_clusters))
        else:
             # Add text label if only one cluster
             ax.text(0.95, 0.95, f'Cluster {unique_labels_valid[0]}', transform=ax.transAxes,
                     ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', fc=colors(0), alpha=0.5))


        plt.tight_layout()
        save_path_mpl = os.path.join(projection_plot_dir, f'{base_filename}_matplotlib.png')
        plt.savefig(save_path_mpl, dpi=150) # Control DPI
        plt.close(fig)
        logger.info(f"Saved {method_str.upper()} plot (matplotlib) to: {save_path_mpl}")
    except Exception as e:
         logger.error(f"Failed to generate matplotlib {method_str.upper()} plot for t={time_index}: {e}")

    # ** 2. Plotly Interactive Plot (Optional) **
    if config.get('VIZ_INTERACTIVE_PLOTS', False):
        if PLOTLY_AVAILABLE:
            logger.info(f"Generating interactive {method_str.upper()} plot for t={time_index}...")
            try:
                # Ensure cluster_id is string for categorical coloring in Plotly
                plot_df['cluster_id'] = plot_df['cluster_id'].astype(str)
                # Define columns for hover info
                hover_cols_base = ['node_id', 'cluster_id']
                available_static_cols = [col for col in ['line_id', 'building_function', 'lat', 'lon'] if col in plot_df.columns]
                hover_cols = hover_cols_base + available_static_cols

                fig_interactive = px.scatter(plot_df, x='dim1', y='dim2',
                                             color='cluster_id', # Use string version for discrete colors
                                             hover_data=hover_cols,
                                             title=f"Interactive {method_str.upper()} of Embeddings (t={time_index}, {timestamp_title})",
                                             labels={'dim1': f'{method_str.upper()} Dim 1', 'dim2': f'{method_str.upper()} Dim 2', 'cluster_id': 'Cluster ID'},
                                             category_orders={'cluster_id': sorted(plot_df['cluster_id'].unique())} # Sort legend
                                             )
                fig_interactive.update_traces(marker=dict(size=6, opacity=0.8)) # Slightly larger markers
                fig_interactive.update_layout(legend_title_text='Cluster ID', hovermode='closest')

                save_path_interactive = os.path.join(projection_plot_dir, f'{base_filename}_interactive.html')
                fig_interactive.write_html(save_path_interactive, include_plotlyjs='cdn') # Use CDN for smaller file size
                logger.info(f"Saved interactive {method_str.upper()} plot to: {save_path_interactive}")
            except Exception as e:
                logger.error(f"Failed to generate interactive {method_str.upper()} plot for t={time_index}: {e}")
        else:
            # Log this only once perhaps? Or just rely on initial check.
            logger.warning("Skipping interactive plot generation: Plotly library not found.")


# --- Standalone Test Block ---
if __name__ == '__main__':
    logger.info("--- Testing viz_embedding.py standalone ---")

    # Create dummy data
    num_nodes_test = 100
    num_features_test = 64
    num_clusters_test = 5
    time_index_test = 24 # Example time index

    dummy_embeddings = np.random.rand(num_nodes_test, num_features_test)
    dummy_labels = np.random.randint(0, num_clusters_test, num_nodes_test)
    dummy_node_ids = [f'Node_{i:03d}' for i in range(num_nodes_test)]
    dummy_static_data = {
        'building_id': dummy_node_ids,
        'line_id': [f'L{i % 5:03d}' for i in range(num_nodes_test)],
        'building_function': ['residential' if i % 2 == 0 else 'non_residential' for i in range(num_nodes_test)],
        'lat': np.random.rand(num_nodes_test) * 0.1 + 52.0,
        'lon': np.random.rand(num_nodes_test) * 0.1 + 4.0,
    }
    dummy_static_df = pd.DataFrame(dummy_static_data).set_index('building_id')

    # Dummy config
    config_test = {
        'SCENARIO_NAME': 'Embedding_Viz_Test',
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Embedding_Viz_Test'),
        'SEED': 42,
        'NUM_CLUSTERS_K': num_clusters_test,
        'SCALE_EMBEDDINGS_BEFORE_TSNE': True,
        'VIZ_INTERACTIVE_PLOTS': PLOTLY_AVAILABLE, # Enable if installed
        'TIME_SERIES_FREQUENCY': '15min',
        # 'VIZ_TSNE_N_ITER': 500 # Example of overriding TSNE param
    }
    os.makedirs(os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'plots', 'embedding_projections'), exist_ok=True)


    # Test TSNE plot
    plot_embedding_projection(config_test, dummy_embeddings, dummy_labels, dummy_node_ids,
                              dummy_static_df, time_index_test, method='tsne')

    # Test UMAP plot (if available)
    if UMAP_AVAILABLE:
        plot_embedding_projection(config_test, dummy_embeddings, dummy_labels, dummy_node_ids,
                                 dummy_static_df, time_index_test, method='umap')
    else:
         logger.info("Skipping standalone UMAP test as library is not installed.")

    logger.info("--- Standalone test complete ---")
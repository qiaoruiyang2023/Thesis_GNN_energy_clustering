# src/graph_construction.py

import pandas as pd
import numpy as np
import os
import logging
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist # For Euclidean distance if needed for k-NN
import joblib # To save/load scaler if needed for k-NN features

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees). Vectorized numpy version.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c # Radius of earth in kilometers.
    return km

def build_adjacency_matrix(nodes: list, edges: list) -> np.ndarray:
    """Builds an adjacency matrix from a list of nodes and edges."""
    n = len(nodes)
    adj = np.zeros((n, n), dtype=int)
    # Ensure node_ids used in edges are strings if nodes list contains strings
    nodes_str = [str(node) for node in nodes]
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes_str)}

    num_valid_edges = 0
    for u, v in edges:
        u_str, v_str = str(u), str(v) # Use string representation for lookup
        if u_str in node_to_idx and v_str in node_to_idx:
            u_idx, v_idx = node_to_idx[u_str], node_to_idx[v_str]
            if u_idx != v_idx: # Avoid self-loops unless intended
                adj[u_idx, v_idx] = 1
                adj[v_idx, u_idx] = 1 # Assuming undirected graph
                num_valid_edges +=1 # Count pairs added
        # else: # Optional logging for debugging
        #     logging.debug(f"Skipping edge ({u}, {v}): Node not found in index map.")

    # The number of matrix entries set to 1 will be 2 * num_valid_edges
    logging.debug(f"Built adjacency matrix with {num_valid_edges} unique undirected edges.")
    return adj

def adj_matrix_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Converts an adjacency matrix to a PyG edge_index tensor (COO format)."""
    row, col = np.where(adj > 0)
    # Ensure row and col are numpy arrays before stacking
    row = np.asarray(row)
    col = np.asarray(col)
    # Filter out self-loops if created accidentally (though build_adj_matrix tries to prevent this)
    mask = row != col
    row, col = row[mask], col[mask]
    # Stack and convert to tensor
    edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    return edge_index

# --- Scenario-Specific Builders ---

def _build_knn_graph(static_features_df: pd.DataFrame, config: dict) -> torch.Tensor:
    """Builds a k-NN graph based on static features."""
    k = config.get('KNN_K', 10)
    metric = config.get('KNN_METRIC', 'euclidean')
    
    # 直接使用lat和lon作为特征
    features_data = static_features_df[['lat', 'lon']].values
    logging.info(f"Building k-NN graph (k={k}, metric='{metric}') using lat/lon coordinates.")

    if features_data.shape[0] <= k:
        logging.warning(f"Number of nodes ({features_data.shape[0]}) is less than or equal to k ({k}). "
                        f"k-NN graph might behave unexpectedly. Reducing k to {features_data.shape[0] - 1}")
        k = max(1, features_data.shape[0] - 1) # k must be < n_samples

    if features_data.shape[0] < 2 :
         logging.warning("Less than 2 nodes, returning empty edge index.")
         return torch.empty((2, 0), dtype=torch.long)


    # kneighbors_graph returns a sparse matrix representation of the graph
    # mode='connectivity' gives unweighted graph
    adj_sparse = kneighbors_graph(features_data, n_neighbors=k, mode='connectivity',
                                  metric=metric, include_self=False, n_jobs=-1) # Use multiple cores if available
    # Convert to dense numpy array to use our helper function
    adj_matrix = adj_sparse.toarray()
    # Make symmetric: If A is a neighbor of B, make B a neighbor of A
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    edge_index = adj_matrix_to_edge_index(adj_matrix)
    return edge_index

def _build_feeder_graph(static_features_df: pd.DataFrame, config: dict) -> torch.Tensor:
    """Builds a graph connecting buildings on the same feeder line."""
    logging.info("Building feeder graph (connecting buildings with the same line_id).")
    if 'line_id' not in static_features_df.columns:
        raise ValueError("'line_id' column is required in static features for feeder graph.")

    nodes = static_features_df.index.tolist() # Original building IDs
    edges = []
    # Group by line_id and create edges within each group (clique)
    for line, group in static_features_df.groupby('line_id'):
        if line == 'UNKNOWN_LINE' or pd.isna(line):
            logging.debug(f"Skipping buildings with unknown line_id: {group.index.tolist()}")
            continue
        line_nodes = group.index.tolist()
        # Create all pairs within the group (if more than one node)
        if len(line_nodes) >= 2:
            for i in range(len(line_nodes)):
                for j in range(i + 1, len(line_nodes)):
                    edges.append((line_nodes[i], line_nodes[j])) # Use original building IDs

    adj_matrix = build_adjacency_matrix(nodes, edges)
    edge_index = adj_matrix_to_edge_index(adj_matrix)
    return edge_index

def _build_distance_graph(static_features_df: pd.DataFrame, config: dict) -> torch.Tensor:
    """Builds a graph connecting buildings within a distance threshold."""
    threshold = config.get('DISTANCE_THRESHOLD_KM', 0.5)
    logging.info(f"Building distance graph (connecting buildings within {threshold} km).")
    if not all(col in static_features_df.columns for col in ['lat', 'lon']):
         raise ValueError("'lat' and 'lon' columns required for distance graph.")

    nodes = static_features_df.index.tolist()
    coords = static_features_df[['lat', 'lon']].values
    n = len(nodes)
    if n < 2:
         logging.warning("Less than 2 nodes, returning empty edge index.")
         return torch.empty((2, 0), dtype=torch.long)

    # Calculate pairwise distances using Haversine
    lat = coords[:, 0]
    lon = coords[:, 1]
    # Create broadcastable arrays for vectorized calculation
    lat1, lat2 = np.meshgrid(lat, lat, indexing='ij')
    lon1, lon2 = np.meshgrid(lon, lon, indexing='ij')
    dist_matrix = haversine_np(lon1, lon1, lat2, lon2) # Corrected lon1 -> lon1

    # Create adjacency matrix based on threshold
    adj_matrix = (dist_matrix <= threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0) # No self-loops

    edge_index = adj_matrix_to_edge_index(adj_matrix)
    return edge_index

def _build_combined_graph(static_features_df: pd.DataFrame, config: dict) -> torch.Tensor:
    """Builds a graph connecting buildings if on same feeder AND within distance."""
    threshold = config.get('DISTANCE_THRESHOLD_KM', 0.5)
    logging.info(f"Building combined graph (same line_id AND distance <= {threshold} km).")
    if 'line_id' not in static_features_df.columns or not all(col in static_features_df.columns for col in ['lat', 'lon']):
         raise ValueError("'line_id', 'lat', 'lon' columns required for combined graph.")

    nodes = static_features_df.index.tolist()
    if len(nodes) < 2:
         logging.warning("Less than 2 nodes, returning empty edge index.")
         return torch.empty((2, 0), dtype=torch.long)

    node_to_idx = {node_id: i for i, node_id in enumerate(nodes)} # Map original ID to 0..N-1 index
    coords = static_features_df[['lat', 'lon']].values
    edges = [] # List of pairs of original node IDs

    # Iterate through feeders first
    for line, group in static_features_df.groupby('line_id'):
         if line == 'UNKNOWN_LINE' or pd.isna(line):
             continue
         line_nodes = group.index.tolist() # Original node IDs in this group
         if len(line_nodes) < 2:
             continue

         # Get integer indices and coordinates for nodes in this group
         line_node_indices = [node_to_idx[node_id] for node_id in line_nodes]
         line_coords = coords[line_node_indices]

         # Calculate pairwise distances ONLY within the group
         lat = line_coords[:, 0]
         lon = line_coords[:, 1]
         lat1, lat2 = np.meshgrid(lat, lat, indexing='ij')
         lon1, lon2 = np.meshgrid(lon, lon, indexing='ij')
         dist_matrix_group = haversine_np(lon1, lon1, lat2, lon2) # Corrected lon1 -> lon1

         # Add edges if distance threshold is met
         for i in range(len(line_nodes)):
             for j in range(i + 1, len(line_nodes)):
                 if dist_matrix_group[i, j] <= threshold:
                     # Add edge using original node IDs
                     edges.append((line_nodes[i], line_nodes[j]))

    # Build final adjacency matrix and edge index using original node IDs and the mapping
    adj_matrix = build_adjacency_matrix(nodes, edges)
    edge_index = adj_matrix_to_edge_index(adj_matrix)
    return edge_index

# --- Main Builder Function ---

def build_graph(config: dict) -> Data:
    """
    Builds and saves the graph based on the scenario specified in the config.
    Loads processed static features.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
              Includes x (static features), edge_index, pos, node_ids, line_ids.
    """
    scenario_name = config.get('SCENARIO_NAME', 'Unknown Scenario')
    logging.info(f"--- Starting Graph Construction for Scenario: {scenario_name} ---")

    processed_dir = config['PROCESSED_DATA_DIR']
    static_features_path = os.path.join(processed_dir, 'processed_static_features.csv')

    # --- Load Processed Static Data ---
    try:
        # Ensure index is treated as string to match potential dict keys later
        static_features_df = pd.read_csv(static_features_path, index_col='building_id', dtype={'building_id': str})
        static_features_df.index = static_features_df.index.astype(str)
        logging.info(f"Loaded processed static features from: {static_features_path} (Shape: {static_features_df.shape})")
    except FileNotFoundError:
        logging.error(f"Processed static features file not found: {static_features_path}. Run data_processing first.")
        raise
    except Exception as e:
        logging.error(f"Error loading processed static features: {e}")
        raise

    if static_features_df.empty:
        raise ValueError("Loaded static features are empty. Cannot build graph.")

    # --- Node Order and Mapping ---
    # Use the index of the DataFrame as the canonical node order
    node_ids = static_features_df.index.tolist() # Original building IDs (strings)
    num_nodes = len(node_ids)
    logging.info(f"Graph will contain {num_nodes} nodes.")

    # --- Select Graph Builder ---
    scenario_type = config.get('SCENARIO_TYPE')
    logging.info(f"Building graph using SCENARIO_TYPE: {scenario_type}")

    if scenario_type == 'A':
        edge_index = _build_knn_graph(static_features_df, config)
    elif scenario_type == 'B':
        edge_index = _build_feeder_graph(static_features_df, config)
    elif scenario_type == 'C':
        edge_index = _build_distance_graph(static_features_df, config)
    elif scenario_type == 'D':
        edge_index = _build_combined_graph(static_features_df, config)
    else:
        raise ValueError(f"Unsupported SCENARIO_TYPE in config: {scenario_type}")

    logging.info(f"Graph construction complete. Number of nodes: {num_nodes}, Number of edges: {edge_index.shape[1]}")

    # --- Prepare Node Features (x) and Positions (pos) ---
    # Select feature columns (exclude metadata like lat, lon, line_id)
    feature_cols = [col for col in static_features_df.columns if col not in ['lat', 'lon', 'line_id']]
    if not feature_cols:
        logging.warning("No feature columns identified in static_features_df after excluding metadata. Node features (x) will be empty.")
        x = torch.empty((num_nodes, 0), dtype=torch.float)
    else:
        logging.info(f"Using {len(feature_cols)} static features for graph nodes.")
        x = torch.tensor(static_features_df[feature_cols].values, dtype=torch.float)

    # Include positions if available
    if 'lat' in static_features_df.columns and 'lon' in static_features_df.columns:
        # Use numpy array first for potential NaN check
        pos_np = static_features_df[['lon', 'lat']].values.astype(np.float32)
        if np.isnan(pos_np).any():
            logging.warning("NaN values found in 'lat' or 'lon' columns. Position data (pos) might be incomplete.")
            # Option: impute pos or handle downstream - for now, create tensor anyway
        pos = torch.tensor(pos_np, dtype=torch.float)
    else:
        pos = None
        logging.warning("'lat' or 'lon' columns not found. Position data (pos) will not be included in the graph.")

    # --- Create PyG Data Object ---
    graph_data = Data(x=x, edge_index=edge_index, pos=pos)
    graph_data.num_nodes = num_nodes
    # Store mapping and line_ids for later use if needed
    graph_data.node_ids = node_ids # List of original building IDs (strings)
    if 'line_id' in static_features_df.columns:
         # Ensure line_ids are stored as a list, handling potential NaNs converted to strings
         graph_data.line_ids = static_features_df['line_id'].fillna('UNKNOWN_LINE').astype(str).tolist()


    # --- Save Graph Object ---
    suffix = '_train' if 'train' in processed_dir else '_test'
    graph_filename = f"graph_{scenario_name}{suffix}.pt"
    graph_save_path = os.path.join(processed_dir, graph_filename)
    if config.get('SAVE_GRAPH_OBJECT', True):
        try:
            torch.save(graph_data, graph_save_path)
            logging.info(f"Saved graph object to: {graph_save_path}")
        except Exception as e:
            logging.error(f"Error saving graph object: {e}")


    logging.info("--- Graph Construction Finished ---")
    return graph_data

# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    logging.info("Testing graph construction script standalone...")
    # Dummy config for testing standalone
    config = {
        'SCENARIO_TYPE': 'D', # Change this to test A, B, C
        'SCENARIO_NAME': 'Scenario_D_Combined_Constraint', # Change accordingly
        'PROCESSED_DATA_DIR': os.path.join('..', 'data', 'processed'), # Adjust relative path
        'RESULTS_DIR': os.path.join('..', 'results'), # Needed for consistency if config used elsewhere
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Scenario_D_Combined_Constraint'),# Adjust
        'KNN_K': 10, # For Scenario A
        'DISTANCE_THRESHOLD_KM': 0.5, # For Scenario C, D
        'SAVE_GRAPH_OBJECT': True
    }
    os.makedirs(config['PROCESSED_DATA_DIR'], exist_ok=True)
    os.makedirs(config['SCENARIO_RESULTS_DIR'], exist_ok=True)

    try:
        # Ensure processed static features exist before running
        static_features_path = os.path.join(config['PROCESSED_DATA_DIR'], 'processed_static_features.csv')
        if not os.path.exists(static_features_path):
             print(f"\nError: Processed static features file not found: {static_features_path}. Run data_processing.py first.")
        else:
            graph = build_graph(config)
            print("\n--- Graph Object Summary ---")
            print(graph)
            print(f"Number of nodes: {graph.num_nodes}")
            print(f"Number of edges: {graph.num_edges}")
            if hasattr(graph, 'node_ids'):
                 print(f"Number of node IDs stored: {len(graph.node_ids)}")
                 print(f"First 5 Node IDs: {graph.node_ids[:5]}")
            if hasattr(graph, 'line_ids'):
                 print(f"Number of line IDs stored: {len(graph.line_ids)}")
                 print(f"First 5 Line IDs: {graph.line_ids[:5]}")
            if graph.pos is not None:
                 print(f"Position tensor shape: {graph.pos.shape}")

    except FileNotFoundError:
         print("\nError: Processed static features file not found. Run data_processing.py first.")
    except Exception as e:
        logging.exception(f"An error occurred during graph construction test: {e}")
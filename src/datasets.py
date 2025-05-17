# src/datasets.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpatioTemporalGraphDataset(Dataset):
    """
    PyTorch Dataset for loading spatio-temporal graph data.

    Loads a static graph structure (nodes, edges, static features) and a time series
    of dynamic node features. Each item returned by __getitem__ represents a
    sequence of dynamic features for ALL nodes over a defined time window (`seq_len`),
    along with the static graph information.
    """
    def __init__(self, config: dict, mode: str = None):
        """
        Args:
            config (dict): Configuration dictionary containing paths and parameters.
            mode (str): 'train' or 'test' - determines which data directory to use. If None, uses config['MODE'].
        """
        self.config = config
        self.mode = mode if mode is not None else config.get('MODE', 'train')
        self.processed_dir = config['PROCESSED_DATA_DIR']
        time_config = config.get('TIME_SERIES_CONFIG', {})
        self.seq_len = time_config.get('MAX_SEQUENCE_LENGTH', 96)  # 默认96个时间点（24小时）

        # --- Load Static Graph Structure ---
        scenario_name = config.get('SCENARIO_NAME', 'default_scenario')
        
        # 在基线模式下不使用后缀
        if config.get('BASELINE_MODE', False):
            suffix = ''
        else:
            # 根据模式选择正确的文件名后缀
            suffix = '_train' if self.mode == 'train' else '_test'
        
        # 加载图对象
        graph_path = os.path.join(self.processed_dir, f'graph_{scenario_name}{suffix}.pt')
        self.graph = torch.load(graph_path)
        logging.info(f"Loaded {self.mode} graph object from: {graph_path}")
        
        # 设置图属性
        self.x_static = self.graph.x.clone().detach() if isinstance(self.graph.x, torch.Tensor) else torch.tensor(self.graph.x, dtype=torch.float)
        self.edge_index = self.graph.edge_index.clone().detach() if isinstance(self.graph.edge_index, torch.Tensor) else torch.tensor(self.graph.edge_index, dtype=torch.long)
        self.pos = self.graph.pos.clone().detach() if hasattr(self.graph, 'pos') and self.graph.pos is not None else None
        self.node_ids = self.graph.node_ids if hasattr(self.graph, 'node_ids') else list(range(self.graph.num_nodes))
        self.num_nodes = self.graph.num_nodes
        
        # 加载动态特征
        dynamic_features_path = os.path.join(self.processed_dir, f'processed_dynamic_features{suffix}.parquet')
        dynamic_df = pd.read_parquet(dynamic_features_path)
        logging.info(f"Loaded {self.mode} processed dynamic features from: {dynamic_features_path}")

        # --- Process and Align Dynamic Features ---
        # Ensure dynamic features are sorted by time for each node
        dynamic_df['timestamp'] = pd.to_datetime(dynamic_df['timestamp'], format='%H:%M:%S') # Ensure datetime for sorting
        dynamic_df.sort_values(by=['building_id', 'timestamp'], inplace=True)

        # Pivot to get time steps as columns temporarily for easier reshaping
        pivot = dynamic_df.pivot(index='building_id', columns='timestamp', values='net_load')

        # Align node order with the graph object
        pivot = pivot.reindex(self.node_ids)

        # Check for missing nodes after reindexing
        if pivot.isnull().values.any():
            missing_nodes = pivot[pivot.isnull().any(axis=1)].index.tolist()
            logging.warning(f"NaNs detected after aligning dynamic features with graph nodes."
                          f" Possibly missing dynamic data for some nodes in graph: {missing_nodes[:5]}...")
            pivot.fillna(0, inplace=True)
            logging.info("Filled missing dynamic feature entries with 0.")

        # Reshape into (num_nodes, num_timesteps, num_dynamic_features)
        self.num_timesteps = pivot.shape[1]
        self.num_dynamic_features = 1  # 只有net_load一个特征

        try:
            self.dynamic_features = torch.tensor(pivot.values, dtype=torch.float).reshape(
                self.num_nodes, self.num_timesteps, self.num_dynamic_features
            )
        except Exception as e:
            logging.error(f"Error reshaping dynamic features. Check pivot table structure and column names. Error: {e}")
            logging.error(f"Pivot columns example: {pivot.columns[:10]}")
            raise

        logging.info(f"Processed dynamic features tensor shape: {self.dynamic_features.shape}")

        # --- Calculate number of possible sequences ---
        # 确保序列长度不超过时间序列长度
        self.seq_len = min(self.seq_len, self.num_timesteps)
        
        # 计算可能的序列数量
        if self.num_timesteps <= self.seq_len:
            logging.warning(f"Time series length ({self.num_timesteps}) is less than or equal to sequence length ({self.seq_len}). Using full sequence.")
            self.num_sequences = 1
        else:
            self.num_sequences = self.num_timesteps - self.seq_len + 1
            
        # 确保至少有一个序列
        self.num_sequences = max(1, self.num_sequences)
        logging.info(f"Dataset initialized for mode '{self.mode}'. Number of sequences: {self.num_sequences}")

        print(f"[{self.mode}] dynamic_features shape: {self.dynamic_features.shape}")
        print(f"[{self.mode}] time steps: {self.num_timesteps}, expected: {self.seq_len}")

    def __len__(self):
        """Returns the number of possible sequences."""
        return self.num_sequences  # 直接返回计算好的序列数量

    def __getitem__(self, idx):
        """
        Returns a data sample for a given sequence index.

        Args:
            idx (int): The index of the starting time step for the sequence.

        Returns:
            torch_geometric.data.Data: A PyG Data object containing:
                - x: Static node features [num_nodes, num_static_features]
                - edge_index: Graph connectivity [2, num_edges]
                - dynamic_seq: Sequence of dynamic features [num_nodes, seq_len, num_dynamic_features]
                - pos: Node positions [num_nodes, 2] (optional)
                - node_ids: List of original node IDs (metadata)
                - start_time_idx: Starting time index of the sequence (metadata)
        """
        if not (0 <= idx < self.__len__()):
            raise IndexError(f"Index {idx} out of range for {self.__len__()} sequences.")

        # 如果时间序列长度小于或等于序列长度，使用整个序列
        if self.num_timesteps <= self.seq_len:
            dynamic_seq = self.dynamic_features
        else:
            t_start = idx
            t_end = min(idx + self.seq_len, self.num_timesteps)
            dynamic_seq = self.dynamic_features[:, t_start:t_end, :]
            # 如果序列长度不足，用零填充
            if t_end - t_start < self.seq_len:
                padding = torch.zeros((self.num_nodes, self.seq_len - (t_end - t_start), self.num_dynamic_features))
                dynamic_seq = torch.cat([dynamic_seq, padding], dim=1)

        # Create a *new* Data object for each item
        data = Data(
            x=self.x_static,            # Static features (shared across time)
            edge_index=self.edge_index, # Graph structure (shared across time)
            dynamic_seq=dynamic_seq     # The dynamic feature sequence for this window
        )
        if self.pos is not None:
            data.pos = self.pos

        # Add metadata
        data.node_ids = self.node_ids
        data.start_time_idx = idx

        return data

# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # Assuming config_loader and previous steps are done
    # from config_loader import load_config
    # config = load_config('scenario_B_config.yaml') # Example

    # Dummy config for testing standalone
    config = {
        'SCENARIO_NAME': 'Scenario_B_Feeder_Constraint', # Or change to test others
        'PROCESSED_DATA_DIR': 'data/processed/',
        'TIME_SERIES_CONFIG': {
            'MAX_SEQUENCE_LENGTH': 96  # Example: 24 hours
        },
        # Add dummy paths/values for other expected keys if needed
    }
    # Adjust data path for standalone execution if needed
    config['PROCESSED_DATA_DIR'] = os.path.join('..', config['PROCESSED_DATA_DIR'])

    try:
        # Ensure processed files exist
        graph_path = os.path.join(config['PROCESSED_DATA_DIR'], f"graph_{config['SCENARIO_NAME']}_train.pt")
        dynamic_features_path = os.path.join(config['PROCESSED_DATA_DIR'], 'processed_dynamic_features_train.parquet')

        if not os.path.exists(graph_path) or not os.path.exists(dynamic_features_path):
             print("Error: Processed graph or dynamic features file not found. Run previous steps first.")
        else:
            dataset = SpatioTemporalGraphDataset(config)
            print(f"Dataset length (number of sequences): {len(dataset)}")

            # Get the first sample
            first_sample = dataset[0]
            print("\n--- First Sample ---")
            print(first_sample)
            print(f"Static features shape (X): {first_sample.x.shape}")
            print(f"Edge index shape: {first_sample.edge_index.shape}")
            print(f"Dynamic sequence shape: {first_sample.dynamic_seq.shape}")
            if hasattr(first_sample, 'pos') and first_sample.pos is not None:
                print(f"Position shape: {first_sample.pos.shape}")
            if hasattr(first_sample, 'node_ids'):
                 print(f"Number of nodes (from node_ids): {len(first_sample.node_ids)}")
            if hasattr(first_sample, 'start_time_idx'):
                 print(f"Start time index: {first_sample.start_time_idx}")


            # Test with DataLoader (batch_size=1 processes the full graph sequence)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # Shuffle False for sequential processing order
            print("\n--- DataLoader Test ---")
            for i, batch in enumerate(dataloader):
                 print(f"Batch {i}:")
                 print(batch)
                 # Access data within the batch (DataLoader might wrap it)
                 print(f" Batch Static X shape: {batch.x.shape}")
                 print(f" Batch Dynamic Seq shape: {batch.dynamic_seq.shape}")
                 if i == 0: # Just check the first batch
                     break

    except FileNotFoundError:
         print("\nError: Processed data files not found. Run data_processing.py and graph_construction.py first.")
    except Exception as e:
        logging.exception(f"An error occurred during dataset testing: {e}")
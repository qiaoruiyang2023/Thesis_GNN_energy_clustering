# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv # Using GATv2Conv as it's generally preferred
# from torch_geometric.nn import GATConv # Alternative if needed
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpatioTemporalGNNEmbedder(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for learning node embeddings.

    Combines optional LSTM layers for temporal feature extraction per node
    and optional GATv2 layers for capturing graph structure and spatial
    dependencies at each time step. Allows ablation studies via config flags.
    """
    def __init__(self, config: dict, static_feature_dim: int, dynamic_feature_dim: int):
        """
        Initializes the SpatioTemporalGNNEmbedder model.

        Args:
            config (dict): Configuration dictionary with model hyperparameters and flags.
            static_feature_dim (int): Dimensionality of static node features.
            dynamic_feature_dim (int): Dimensionality of dynamic node features per time step.
        """
        super(SpatioTemporalGNNEmbedder, self).__init__()
        logging.info("Initializing SpatioTemporalGNNEmbedder...")

        # --- Get Hyperparameters ---
        self.lstm_hidden_dim = config.get('LSTM_HIDDEN_DIM', 128)
        self.lstm_layers = config.get('LSTM_LAYERS', 1)
        self.gnn_hidden_dim = config.get('GNN_HIDDEN_DIM', 128)
        self.gnn_layers = config.get('GNN_LAYERS', 2)
        self.gnn_heads = config.get('GNN_HEADS', 4)
        self.embedding_dim = config.get('EMBEDDING_DIM', 64) # Final output dimension
        self.dropout_rate = config.get('DROPOUT_RATE', 0.3)

        # --- Ablation Study Flags ---
        self.use_lstm = config.get('MODEL_USE_LSTM', True)
        self.use_gnn = config.get('MODEL_USE_GNN', True)
        self.use_static = config.get('MODEL_USE_STATIC_FEATURES', True)
        logging.info(f"Model Configuration: Use LSTM={self.use_lstm}, Use GNN={self.use_gnn}, Use Static Features={self.use_static}")


        self.static_feature_dim = static_feature_dim
        self.dynamic_feature_dim = dynamic_feature_dim

        # --- Define Layers Conditionally ---

        # 1. Temporal Encoder (LSTM) - Optional
        temporal_feature_output_dim = self.dynamic_feature_dim # Default if LSTM is off
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.dynamic_feature_dim,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_layers,
                batch_first=True, # Expect input shape [batch_size=num_nodes, seq_len, features]
                dropout=self.dropout_rate if self.lstm_layers > 1 else 0
            )
            temporal_feature_output_dim = self.lstm_hidden_dim # Update output dim if LSTM is used
            logging.info(f"Initialized LSTM: input={self.dynamic_feature_dim}, hidden={self.lstm_hidden_dim}, layers={self.lstm_layers}")
        else:
            self.lstm = None
            logging.info("LSTM component is disabled.")


        # 2. Spatial Encoders (GATv2 Layers) - Optional
        self.gnn_layers_list = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        gnn_output_feature_dim = 0 # Placeholder, updated below

        if self.use_gnn:
            # Determine GNN input dimension based on flags
            gnn_input_dim = 0
            if self.use_static:
                gnn_input_dim += self.static_feature_dim
            # Add dimension of temporal features (either LSTM output or raw dynamic)
            gnn_input_dim += temporal_feature_output_dim

            if gnn_input_dim == 0:
                raise ValueError("GNN input dimension is zero. Cannot use GNN without static or temporal features enabled.")

            current_gnn_layer_input_dim = gnn_input_dim
            for i in range(self.gnn_layers):
                # Output channels per head
                # If it's the last layer and heads=1, maybe don't multiply? PyG handles concat=True output dim.
                gnn_layer_out_channels = self.gnn_hidden_dim

                self.gnn_layers_list.append(
                    GATv2Conv(current_gnn_layer_input_dim,
                              gnn_layer_out_channels,
                              heads=self.gnn_heads,
                              dropout=self.dropout_rate,
                              concat=True) # Output dim = heads * out_channels
                )
                # Input features for BatchNorm = heads * out_channels
                gnn_output_feature_dim = self.gnn_heads * gnn_layer_out_channels
                self.batch_norms.append(nn.BatchNorm1d(gnn_output_feature_dim))
                logging.info(f"Initialized GATv2Conv Layer {i+1}: in={current_gnn_layer_input_dim}, out_per_head={gnn_layer_out_channels}, heads={self.gnn_heads}")
                # Input for the next GNN layer is the output dimension of the current one
                current_gnn_layer_input_dim = gnn_output_feature_dim
        else:
            logging.info("GNN component is disabled.")
            # If GNN is off, the features going to projection depend only on static/temporal flags
            gnn_output_feature_dim = 0 # GNN doesn't produce output
            # In ablation without GNN, we typically project directly from the temporal component only
            # (LSTM output or raw dynamic features) to isolate GNN's effect.
            # If static features were desired without GNN, the architecture might need rethinking.


        # 3. Final Output Projection Layer
        # Determine the input dimension for the projection layer
        if self.use_gnn:
            projection_input_dim = gnn_output_feature_dim # Use output of last GNN layer
        else:
            # If GNN is off, project directly from the temporal features
            projection_input_dim = temporal_feature_output_dim

        if projection_input_dim == 0:
             raise ValueError("Input dimension for final projection is zero. Check model component flags.")

        self.output_projection = nn.Linear(projection_input_dim, self.embedding_dim)
        logging.info(f"Initialized Output Projection Layer: in={projection_input_dim}, out={self.embedding_dim}")

        # 4. Dropout Layer (used after GNN layers if active)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass of the model, respecting ablation flags.

        Args:
            data (torch_geometric.data.Data or torch_geometric.data.Batch): Input data object. Expected attributes:
                - x (torch.Tensor): Static node features [num_nodes, static_feature_dim]
                - edge_index (torch.Tensor): Graph connectivity [2, num_edges]
                - dynamic_seq (torch.Tensor): Sequence of dynamic features [num_nodes, seq_len, dynamic_feature_dim]

        Returns:
            torch.Tensor: Node embeddings over the sequence [num_nodes, seq_len, embedding_dim]
        """
        x_static = data.x
        edge_index = data.edge_index
        dynamic_seq = data.dynamic_seq # Shape: [num_nodes, seq_len, dynamic_feature_dim]

        num_nodes, seq_len, _ = dynamic_seq.shape

        # 1. Get Temporal Features (LSTM output or raw dynamic features)
        if self.use_lstm and self.lstm is not None:
            lstm_out, _ = self.lstm(dynamic_seq) # [N, T, lstm_hidden_dim]
        else:
            # If LSTM is disabled, use raw dynamic features as the temporal representation
            lstm_out = dynamic_seq # [N, T, dynamic_feature_dim]

        # Initialize list to store embeddings for each time step
        embeddings_over_time = []

        # 2. Process each time step
        for t in range(seq_len):
            # Get temporal features for the current time step
            temporal_features_t = lstm_out[:, t, :] # Shape: [N, temporal_feature_output_dim]

            # Determine input features for GNN or Projection based on flags
            if self.use_gnn:
                # Prepare GNN input
                if self.use_static:
                    gnn_input_t = torch.cat([x_static, temporal_features_t], dim=1)
                else:
                    gnn_input_t = temporal_features_t # Use only temporal if static disabled

                # Pass through GNN layers
                gnn_out_t = gnn_input_t
                for i in range(self.gnn_layers):
                    gnn_out_t = self.gnn_layers_list[i](gnn_out_t, edge_index)
                    gnn_out_t = self.batch_norms[i](gnn_out_t)
                    gnn_out_t = F.relu(gnn_out_t) # Or F.elu / F.leaky_relu
                    if i < self.gnn_layers - 1: # Apply dropout to hidden GNN layers only
                        gnn_out_t = self.dropout(gnn_out_t)
                # gnn_out_t is the final GNN output for this timestep
                features_for_projection = gnn_out_t

            else: # GNN is disabled
                # Use only the temporal features for the final projection
                features_for_projection = temporal_features_t

            # 3. Project to final embedding dimension
            final_embedding_t = self.output_projection(features_for_projection) # Shape: [N, embedding_dim]

            embeddings_over_time.append(final_embedding_t)

        # Stack embeddings from all time steps
        # Output shape: [num_nodes, seq_len, embedding_dim]
        final_embeddings = torch.stack(embeddings_over_time, dim=1)

        return final_embeddings

# Example Usage (within training script or for testing)
if __name__ == '__main__':
    # Dummy config and data for testing model initialization and forward pass
    # Test different ablation settings
    base_config = {
        'LSTM_HIDDEN_DIM': 64, 'LSTM_LAYERS': 1,
        'GNN_HIDDEN_DIM': 64, 'GNN_LAYERS': 2, 'GNN_HEADS': 4,
        'EMBEDDING_DIM': 32, 'DROPOUT_RATE': 0.2,
        # Ablation Flags - change these to test
        'MODEL_USE_LSTM': True,
        'MODEL_USE_GNN': True,
        'MODEL_USE_STATIC_FEATURES': True,
    }
    num_nodes = 50
    num_static_features = 15
    num_dynamic_features = 1 # Just net_load
    seq_len = 24
    num_edges = 200

    # Create dummy static features and edge index
    x_static = torch.randn(num_nodes, num_static_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Create dummy dynamic sequence
    dynamic_seq = torch.randn(num_nodes, seq_len, num_dynamic_features)

    # Create dummy PyG Data object
    from torch_geometric.data import Data # Import locally for test
    data = Data(x=x_static, edge_index=edge_index, dynamic_seq=dynamic_seq)

    print(f"\n--- Testing Model with Config: ---")
    print(f"USE_LSTM={base_config['MODEL_USE_LSTM']}, USE_GNN={base_config['MODEL_USE_GNN']}, USE_STATIC={base_config['MODEL_USE_STATIC_FEATURES']}")

    try:
        # Initialize model
        model = SpatioTemporalGNNEmbedder(base_config,
                                           static_feature_dim=num_static_features,
                                           dynamic_feature_dim=num_dynamic_features)
        print("\n--- Model Architecture ---")
        print(model)

        # Test forward pass
        model.train() # Set model to training mode (enables dropout if used)
        output_embeddings = model(data)
        print("\n--- Forward Pass Test ---")
        print(f"Input dynamic_seq shape: {dynamic_seq.shape}")
        print(f"Output embeddings shape: {output_embeddings.shape}")
        # Check output shape consistency
        expected_shape = (num_nodes, seq_len, base_config['EMBEDDING_DIM'])
        assert output_embeddings.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, Got {output_embeddings.shape}"
        print("Forward pass successful! Output shape is correct.")

        # Test backward pass (requires a dummy loss)
        target = torch.randn_like(output_embeddings) # Dummy target
        loss = F.mse_loss(output_embeddings, target)
        loss.backward()
        print("Backward pass successful (dummy loss)!")
        # Check gradients for a parameter (e.g., first LSTM weight if LSTM is used)
        if model.lstm is not None and len(list(model.lstm.parameters())) > 0:
             print(f"LSTM weights grad exists: {list(model.lstm.parameters())[0].grad is not None}")
        elif model.use_gnn and len(model.gnn_layers_list) > 0 and len(list(model.gnn_layers_list[0].parameters())) > 0:
             print(f"First GNN layer weights grad exists: {list(model.gnn_layers_list[0].parameters())[0].grad is not None}")
        elif len(list(model.output_projection.parameters())) > 0:
            print(f"Projection layer weights grad exists: {list(model.output_projection.parameters())[0].grad is not None}")


    except Exception as e:
        logging.exception(f"An error occurred during model testing with config {base_config}: {e}")

    # --- Example: Test an ablation setting ---
    print("\n--- Testing Model with GNN Disabled: ---")
    ablation_config = base_config.copy()
    ablation_config['MODEL_USE_GNN'] = False
    try:
        model_ablated = SpatioTemporalGNNEmbedder(ablation_config,
                                                   static_feature_dim=num_static_features,
                                                   dynamic_feature_dim=num_dynamic_features)
        print(model_ablated)
        output_ablated = model_ablated(data)
        expected_shape_ablated = (num_nodes, seq_len, ablation_config['EMBEDDING_DIM'])
        assert output_ablated.shape == expected_shape_ablated, f"Ablated shape mismatch! Expected {expected_shape_ablated}, Got {output_ablated.shape}"
        print("Forward pass successful for GNN ablation!")
    except Exception as e:
         logging.exception(f"An error occurred during model ablation testing: {e}")
# src/loss_fn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContrastiveLoss(nn.Module):
    """
    Triplet Margin Loss for learning graph embeddings.

    Encourages connected nodes (anchor-positive) to have smaller distances
    in the embedding space than unconnected nodes (anchor-negative) by at least a margin.
    """
    def __init__(self, config: dict):
        """
        Initializes the ContrastiveLoss module.

        Args:
            config (dict): Configuration dictionary containing parameters like:
                           - CONTRASTIVE_MARGIN (float): The margin for the triplet loss.
                           - LOSS_EMBEDDING_SOURCE (str): 'last' to use last time step embeddings,
                                                          'mean' to use mean over time. (Default: 'last')
                           - NEGATIVE_SAMPLING_STRATEGY (str): 'random' (Default), 'hard' (TODO)
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = config.get('CONTRASTIVE_MARGIN', 1.0)
        self.embedding_source = config.get('LOSS_EMBEDDING_SOURCE', 'last')
        self.sampling_strategy = config.get('NEGATIVE_SAMPLING_STRATEGY', 'random')
        logging.info(f"Initialized ContrastiveLoss (Triplet Margin) with margin={self.margin}, "
                     f"embedding_source='{self.embedding_source}', sampling='{self.sampling_strategy}'")

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Calculates the triplet margin loss.

        Args:
            embeddings (torch.Tensor): Node embeddings output by the GNN model.
                                       Shape: [num_nodes, seq_len, embedding_dim]
            edge_index (torch.Tensor): Graph connectivity. Shape: [2, num_edges]

        Returns:
            torch.Tensor: The calculated average triplet loss (scalar).
        """

        # --- Select Embeddings ---
        if self.embedding_source == 'last':
            # Use embeddings from the last time step
            emb_t = embeddings[:, -1, :] # Shape: [num_nodes, embedding_dim]
        elif self.embedding_source == 'mean':
            # Use mean embeddings over the sequence length
            emb_t = embeddings.mean(dim=1) # Shape: [num_nodes, embedding_dim]
        else:
            logging.warning(f"Unknown LOSS_EMBEDDING_SOURCE '{self.embedding_source}', defaulting to 'last'.")
            emb_t = embeddings[:, -1, :]

        num_nodes = emb_t.shape[0]
        device = emb_t.device

        # Ensure edge_index is on the same device
        edge_index = edge_index.to(device)

        # --- Sample Triplets (Anchor, Positive, Negative) ---
        # For simplicity and reasonable efficiency, iterate through edges to get (anchor, positive)
        # pairs and randomly sample a negative for each.

        # Get source (anchor) and target (positive) nodes from edges
        anchors = edge_index[0]
        positives = edge_index[1]

        # --- Random Negative Sampling ---
        if self.sampling_strategy == 'random':
             # Sample one random negative for each positive pair (edge)
             # Ensure negative is not the anchor or the positive node itself
             num_edges = anchors.size(0)
             # Generate random indices for negatives
             neg_indices = torch.randint(0, num_nodes, (num_edges,), device=device)

             # Check and resample if neg_idx is anchor or positive
             needs_resample = (neg_indices == anchors) | (neg_indices == positives)
             # Keep resampling until valid negatives are found for all problematic cases
             # Note: In very dense graphs or small graphs, this loop could potentially be long,
             # but statistically it's usually fast. A max iteration count could be added.
             while torch.any(needs_resample):
                  resample_indices = torch.where(needs_resample)[0]
                  neg_indices[resample_indices] = torch.randint(0, num_nodes, (len(resample_indices),), device=device)
                  needs_resample = (neg_indices == anchors) | (neg_indices == positives)

             negatives = neg_indices # The final sampled negative indices

        # TODO: Implement 'hard' negative mining if needed later
        # elif self.sampling_strategy == 'hard':
        #     # Find the negative node (not connected to anchor) that has the smallest distance to anchor
        #     pass # Requires computing many distances or using specialized libraries
        else:
             raise ValueError(f"Unsupported NEGATIVE_SAMPLING_STRATEGY: {self.sampling_strategy}")


        # --- Retrieve Embeddings for Triplets ---
        emb_anchor = emb_t[anchors]     # Shape: [num_edges, embedding_dim]
        emb_positive = emb_t[positives] # Shape: [num_edges, embedding_dim]
        emb_negative = emb_t[negatives] # Shape: [num_edges, embedding_dim]

        # --- Calculate Triplet Loss ---
        # Use Euclidean distance (p=2)
        pos_dist = F.pairwise_distance(emb_anchor, emb_positive, p=2)
        neg_dist = F.pairwise_distance(emb_anchor, emb_negative, p=2)

        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)

        # Return the average loss over all triplets (edges)
        return loss.mean()

# Example Usage (within training script or for testing)
if __name__ == '__main__':
    # Dummy config and data for testing
    config = {
        'CONTRASTIVE_MARGIN': 1.0,
        'LOSS_EMBEDDING_SOURCE': 'last', # 'last' or 'mean'
        'NEGATIVE_SAMPLING_STRATEGY': 'random'
    }
    num_nodes = 50
    seq_len = 10
    embedding_dim = 32
    num_edges = 150

    # Dummy embeddings and edge index
    embeddings = torch.randn(num_nodes, seq_len, embedding_dim, requires_grad=True) # Set requires_grad=True for backward test
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Initialize loss
    loss_fn = ContrastiveLoss(config)

    # Calculate loss
    try:
        loss_value = loss_fn(embeddings, edge_index)
        print(f"\n--- Loss Calculation Test ---")
        print(f"Calculated Loss: {loss_value.item()}")
        assert loss_value >= 0 # Loss should be non-negative

        # Test backward pass
        loss_value.backward()
        print("Backward pass successful!")
        # Check if gradients exist for embeddings
        assert embeddings.grad is not None
        print(f"Gradient shape w.r.t embeddings: {embeddings.grad.shape}")


    except Exception as e:
        logging.exception(f"An error occurred during loss function testing: {e}")
# src/training.py

import torch
import torch.optim as optim
# from torch.utils.data import DataLoader # Ensure this is commented out/removed
from torch_geometric.loader import DataLoader # Ensure this is the DataLoader being used
import os
import logging
import time
from tqdm import tqdm # For progress bar
import torch.nn as nn
import numpy as np # Added for standalone seed setting
import random      # Added for standalone seed setting


# Import from our source modules
# Ensure these paths are correct relative to where this script is or adjust sys.path
try:
    from models import SpatioTemporalGNNEmbedder # Assuming models.py is in the same directory or src/
    from loss_fn import ContrastiveLoss         # Assuming loss_fn.py is in the same directory or src/
    from datasets import SpatioTemporalGraphDataset # Assuming datasets.py is in the same directory or src/
except ImportError as e:
     # Basic logging might not be configured yet if run standalone very early
     print(f"[ERROR] Error importing project modules in training.py: {e}")
     print("[ERROR] Ensure training.py is run in an environment where 'models', 'loss_fn', 'datasets' are accessible.")
     # Depending on execution context, might need:
     # import sys
     # sys.path.append(os.path.dirname(__file__)) # Add src directory to path if needed

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                loss_fn: nn.Module, device: torch.device, epoch: int, num_epochs: int) -> float:
    """
    Performs a single training epoch.

    Args:
        model: The PyTorch model to train.
        dataloader: The PyG DataLoader providing training data batches (Data or Batch objects).
        optimizer: The optimizer for updating model weights.
        loss_fn: The loss function to calculate training loss.
        device: The device (CPU or CUDA) to run training on.
        epoch (int): Current epoch number (for logging).
        num_epochs (int): Total number of epochs (for progress bar).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = 0 # Initialize batch count

    # Progress bar setup using the PyG DataLoader
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, unit="batch")

    for batch in pbar:
        num_batches += 1 # Increment batch count
        try:
            # PyG DataLoader handles moving Data/Batch objects correctly
            batch = batch.to(device)
        except Exception as e_move:
            logging.error(f"Error moving batch to device: {e_move}. Skipping batch.")
            continue # Skip this batch

        optimizer.zero_grad()

        try:
            # Forward pass - Model expects a PyG Data or Batch object
            embeddings = model(batch) # Output Shape: [num_nodes_in_batch, seq_len, embedding_dim]

            # Calculate loss
            # Loss function expects embeddings and edge_index from the batch object
            # Ensure batch has edge_index attribute (should for PyG Data/Batch)
            if not hasattr(batch, 'edge_index'):
                 logging.error("Batch object missing 'edge_index'. Cannot compute loss. Skipping batch.")
                 continue
            loss = loss_fn(embeddings, batch.edge_index)
            # 加入数值稳定项
            loss = torch.clamp(loss, min=1e-7, max=1e7)
            # 打印中间值以帮助调试
            logging.info(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.6f}, Embeddings Mean: {embeddings.mean().item():.6f}, Embeddings Std: {embeddings.std().item():.6f}")

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}") # Update progress bar

        except RuntimeError as e_mem:
             if "CUDA out of memory" in str(e_mem):
                 logging.error(f"CUDA out of memory during training step (Epoch {epoch+1}). Try reducing batch size if > 1, sequence length, or model size.")
                 # Optionally: torch.cuda.empty_cache() # Might help, but often doesn't solve fundamental OOM
                 raise e_mem # Reraise the error to stop training
             else:
                 logging.error(f"Runtime error during training step: {e_mem}. Skipping batch.")
                 continue
        except Exception as e_train:
            logging.error(f"Error during training step (forward/loss/backward): {e_train}. Skipping batch.")
            # Optionally raise e_train after logging if you want the script to stop
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    pbar.close() # Close progress bar for the epoch
    return avg_loss

def evaluate_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    """
    Evaluate model performance on the test set.

    Args:
        model: PyTorch model to evaluate
        dataloader: PyG DataLoader providing test data
        loss_fn: Loss function for computing loss
        device: Device to run evaluation on (CPU or CUDA)

    Returns:
        float: Average loss on the test set
    """
    model.eval()  # Set to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # No gradient computation
        for batch in dataloader:
            try:
                batch = batch.to(device)
                embeddings = model(batch)
                
                if not hasattr(batch, 'edge_index'):
                    logging.error("Batch object missing 'edge_index'. Cannot compute loss. Skipping batch.")
                    continue
                    
                loss = loss_fn(embeddings, batch.edge_index)
                loss = torch.clamp(loss, min=1e-7, max=1e7)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logging.error(f"Error during evaluation step: {e}. Skipping batch.")
                continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def train_model(config: dict, model: SpatioTemporalGNNEmbedder, dataset: SpatioTemporalGraphDataset) -> tuple:
    """
    训练时空GNN模型的主函数。

    Args:
        config (dict): 配置字典
        model (SpatioTemporalGNNEmbedder): 初始化的模型实例
        dataset (SpatioTemporalGraphDataset): 初始化的数据集实例

    Returns:
        tuple: 包含每个epoch的训练损失和测试损失的列表
    """
    logging.info(f"--- 开始模型训练 - 场景: {config.get('SCENARIO_NAME', 'Unknown')} ---")
    start_time = time.time()

    # --- 设置设备 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        logging.info(f"CUDA可用。在GPU {gpu_index}上训练: {gpu_name}。")
    else:
        device = torch.device("cpu")
        logging.info("CUDA不可用。在CPU上训练。")
    model.to(device)

    # --- 设置PyG DataLoader ---
    batch_size = config.get('TRAINING_BATCH_SIZE', 1)
    num_workers = config.get('DATALOADER_WORKERS', 0)
    if batch_size != 1:
        logging.warning(f"配置的BATCH_SIZE为{batch_size}。确保模型/损失函数能正确处理批次。")

    # 创建训练集和测试集的DataLoader
    # 由于数据集长度已经很小（只有1个序列），我们不再进行划分
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 使用相同的数据集作为测试集
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    logging.info(f"PyG DataLoader创建完成 - 训练集: batch_size={batch_size}, shuffle=True, num_workers={num_workers}, pin_memory={train_loader.pin_memory}")
    logging.info(f"PyG DataLoader创建完成 - 测试集: batch_size={batch_size}, shuffle=False, num_workers={num_workers}, pin_memory={test_loader.pin_memory}")

    # --- 设置优化器 ---
    lr = config.get('LEARNING_RATE', 0.0001)
    wd = config.get('WEIGHT_DECAY', 0.0001)
    optimizer_name = config.get('OPTIMIZER', 'AdamW').lower()

    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        logging.warning(f"不支持的优化器'{optimizer_name}', 默认使用AdamW。")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    logging.info(f"优化器: {optimizer_name.capitalize()}, 学习率: {lr}, 权重衰减: {wd}")

    # --- 设置损失函数 ---
    loss_fn_name = config.get('LOSS_FUNCTION', 'ContrastiveLoss')
    if loss_fn_name == 'ContrastiveLoss':
        loss_fn = ContrastiveLoss(config).to(device)
    else:
        raise ValueError(f"配置中指定的LOSS_FUNCTION不支持: {loss_fn_name}")
    logging.info(f"损失函数: {loss_fn_name}")

    # --- 训练循环 ---
    num_epochs = config.get('EPOCHS', 50)
    logging.info(f"开始训练, 共{num_epochs}个epoch, 设备: {device}")

    training_losses = []
    test_losses = []

    try:
        for epoch in range(num_epochs):
            # 训练一个epoch
            avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, num_epochs)
            training_losses.append(avg_train_loss)
            logging.info(f"Epoch {epoch+1}/{num_epochs} - 平均训练损失: {avg_train_loss:.6f}")

            # 在测试集上评估
            avg_test_loss = evaluate_epoch(model, test_loader, loss_fn, device)
            test_losses.append(avg_test_loss)
            logging.info(f"Epoch {epoch+1}/{num_epochs} - 平均测试损失: {avg_test_loss:.6f}")

    except KeyboardInterrupt:
        logging.warning("用户中断训练(KeyboardInterrupt)。保存当前模型状态。")
    except Exception as e:
        logging.exception(f"训练循环中发生错误: {e}")
    finally:
        # --- 保存最终模型 ---
        if config.get('SAVE_MODEL', True):
            model_save_dir = os.path.join(config['SCENARIO_RESULTS_DIR'], 'model')
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, 'final_model.pt')
            try:
                model.to('cpu')
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"保存最终模型状态到: {model_save_path}")
            except Exception as e_save:
                logging.error(f"保存模型时出错: {e_save}")

        end_time = time.time()
        training_duration = end_time - start_time
        logging.info(f"--- 模型训练完成。持续时间: {training_duration:.2f}秒 ---")

    # 返回训练损失和测试损失
    return training_losses, test_losses

def loss_fn(embeddings, dynamic_features, config):
    """
    Compute loss for training the GNN embedder.
    Args:
        embeddings: Node embeddings from the GNN
        dynamic_features: Dynamic features (e.g., time series data)
        config: Configuration dictionary
    Returns:
        loss: Combined loss value
    """
    # 1. 计算动态特征与嵌入之间的余弦相似度
    # 将 dynamic_features 从 [N, T, D] 转换为 [N, D*T] 以便与 embeddings 对齐
    N, T, D = dynamic_features.shape
    dynamic_flat = dynamic_features.reshape(N, T * D)
    # 计算余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(embeddings, dynamic_flat, dim=1)
    # 动态行为对齐项：希望嵌入与动态特征相似，因此用 1 - cos_sim 作为损失
    dynamic_alignment_loss = 1 - cos_sim.mean()
    # 从配置中获取权重，默认为 0.1
    dynamic_weight = config.get('DYNAMIC_ALIGNMENT_WEIGHT', 0.1)
    # 2. 原有的损失项（例如，对比损失、重构损失等）
    # 这里假设原有损失为 0，实际应用中请替换为你的原始损失计算
    original_loss = torch.tensor(0.0, device=embeddings.device)
    # 3. 组合损失
    loss = original_loss + dynamic_weight * dynamic_alignment_loss
    return loss

# Example Usage (within main.py or for testing)
if __name__ == '__main__':
    # This block requires successful execution of previous steps:
    # config_loader, data_processing, graph_construction, datasets, models, loss_fn
    # And saved processed data/graph files.

    logging.info("--- Testing training script standalone ---")

    # --- CUDA Check ---
    if torch.cuda.is_available():
        print(f"Is CUDA available? {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available.")

    # 1. Load Config (replace with actual loader or use dummy)
    # Using a dummy config here for demonstration
    config_test = {
        'SCENARIO_NAME': 'Scenario_Standalone_Test',
        'PROCESSED_DATA_DIR': os.path.join('..', 'data', 'processed'),
        'RESULTS_DIR': os.path.join('..', 'results'),
        'SCENARIO_RESULTS_DIR': os.path.join('..', 'results', 'Scenario_Standalone_Test'),
        'MODEL_NAME': 'GATLSTMEmbedder',
        'INPUT_SEQ_LEN': 12,
        'EPOCHS': 5,
        'LEARNING_RATE': 0.005,
        'OPTIMIZER': 'AdamW',
        'WEIGHT_DECAY': 0.0001,
        'LOSS_FUNCTION': 'ContrastiveLoss',
        'CONTRASTIVE_MARGIN': 1.0,
        'LOSS_EMBEDDING_SOURCE': 'last',
        'NEGATIVE_SAMPLING_STRATEGY': 'random',
        'SAVE_MODEL': True,
        'DATALOADER_WORKERS': 0, # Set based on your system, 0 often safest for testing
        'TRAINING_BATCH_SIZE': 1, # Keep as 1 for this dataset type
        'LSTM_HIDDEN_DIM': 32, 'LSTM_LAYERS': 1, 'GNN_HIDDEN_DIM': 32,
        'GNN_LAYERS': 1, 'GNN_HEADS': 2, 'EMBEDDING_DIM': 16, 'DROPOUT_RATE': 0.1,
        'SEED': 42
    }
     # Set seed if needed for standalone
    random_seed = config_test.get('SEED', 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

     # Ensure results dir exists
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
             config_test['SCENARIO_NAME'] = 'Scenario_B_Feeder_Constraint' # Fallback
             # Recheck paths after fallback
             graph_path = os.path.join(config_test['PROCESSED_DATA_DIR'], f"graph_{config_test['SCENARIO_NAME']}.pt")
             if not os.path.exists(graph_path) or not os.path.exists(dyn_path):
                  raise FileNotFoundError(f"Required processed data not found for fallback scenario either: {graph_path} or {dyn_path}")


        logging.info(f"--- Standalone: Initializing Dataset using config for {config_test['SCENARIO_NAME']} ---")
        dataset = SpatioTemporalGraphDataset(config_test)

        # 3. Initialize Model (get feature dims from dataset/graph)
        static_dim = dataset.x_static.shape[1]
        dynamic_dim = dataset.dynamic_features.shape[2]
        model_instance = SpatioTemporalGNNEmbedder(config_test, static_feature_dim=static_dim, dynamic_feature_dim=dynamic_dim)
        logging.info(f"--- Standalone: Model Initialized ---")

        # 4. Run Training and capture losses
        logging.info(f"--- Standalone: Starting Training ---")
        epoch_losses, test_losses = train_model(config_test, model_instance, dataset)

        print("\n--- Standalone Training Test Finished ---")
        if epoch_losses: # Check if training actually ran
            print(f"Average Training Losses per Epoch: {[f'{l:.6f}' for l in epoch_losses]}")
        else:
            print("Training did not complete any epochs successfully.")
        if config_test.get('SAVE_MODEL', True):
             print(f"Final model state dict saved in: {os.path.join(config_test['SCENARIO_RESULTS_DIR'], 'model')}")

    except FileNotFoundError as e:
        print(f"\nError: Prerequisite file not found. Ensure data_processing.py and graph_construction.py ran successfully. Details: {e}")
    except Exception as e:
        logging.exception(f"An error occurred during training standalone test: {e}")
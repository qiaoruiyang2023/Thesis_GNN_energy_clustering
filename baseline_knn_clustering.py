import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import pandas as pd
import numpy as np
import torch
import logging
from sklearn.cluster import KMeans
from src.evaluation import run_evaluation
from src.visualization import run_visualization
from src.data_processing import preprocess_dynamic_features, preprocess_static_features, id_column

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    # 1. 读取配置
    config_path = 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/configs/base_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 补充结果目录，保证与GNN-LSTM一致
    config['SCENARIO_RESULTS_DIR'] = os.path.join(
        config['RESULTS_DIR'], 'baseline_knn', 'Scenario_A_No_Constraint_KNN'
    )

    raw_dir = config['RAW_DATA_DIR']
    static_file = os.path.join(raw_dir, config['BUILDINGS_DEMO_FILE'])
    assignments_file = os.path.join(raw_dir, config['BUILDING_ASSIGNMENTS_FILE'])
    timeseries_file = os.path.join(raw_dir, config['TIME_SERIES_LOADS_FILE'])

    # 2. 读取原始数据
    buildings_df = pd.read_csv(static_file, dtype={id_column: str})
    assignments_df = pd.read_csv(assignments_file, dtype={id_column: str, 'line_id': str})
    timeseries_df = pd.read_csv(timeseries_file, dtype={id_column: str, 'Energy': str})

    # 3. 动态特征处理
    processed_dynamic_df, buildings_with_ts = preprocess_dynamic_features(timeseries_df, config)
    # 4. 静态特征处理
    processed_static_df = preprocess_static_features(buildings_df, assignments_df, buildings_with_ts, config)

    # 5. 对齐building顺序
    processed_static_df = processed_static_df.loc[buildings_with_ts]
    N = len(buildings_with_ts)
    # 只保留数值型特征（去掉building_id和line_id）
    static_feature_cols = [col for col in processed_static_df.columns if col not in ['building_id', 'line_id']]
    static_features = processed_static_df[static_feature_cols].values  # [N, S]
    S = static_features.shape[1]
    # 动态特征 [N, T]
    T = processed_dynamic_df['timestamp'].nunique()
    all_times = sorted(processed_dynamic_df['timestamp'].unique(), key=lambda x: str(x))
    dynamic_matrix = processed_dynamic_df.pivot(index='building_id', columns='timestamp', values='net_load').loc[buildings_with_ts, all_times].values  # [N, T]
    dynamic_features = dynamic_matrix[..., np.newaxis]  # [N, T, 1]

    # 6. 拼接特征
    combined_features = np.zeros((N, T, S+1))
    for t in range(T):
        combined_features[:, t, :S] = static_features
        combined_features[:, t, S:] = dynamic_features[:, t, :]

    # 7. 聚类
    n_clusters = config.get('NUM_CLUSTERS_K', 6)
    all_labels = []
    cluster_assignments = {}
    logging.info(f"开始KMeans聚类 (n_clusters={n_clusters})...")
    for t in range(T):
        features_t = combined_features[:, t, :]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_t = kmeans.fit_predict(features_t)
        cluster_assignments[t] = {str(bid): int(label) for bid, label in zip(buildings_with_ts, labels_t)}
        all_labels.append(labels_t)
    all_labels = np.stack(all_labels, axis=1)  # [N, T]
    logging.info("聚类完成")

    # 8. 保存聚类结果
    save_dir = config['SCENARIO_RESULTS_DIR']
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(os.path.join(save_dir, 'cluster_labels_timeseries.csv'), all_labels, delimiter=',')

    # 保存详细分布
    cluster_df = pd.DataFrame()
    for t in range(T):
        df_t = pd.DataFrame({
            'building_id': buildings_with_ts,
            'time_index': t,
            'cluster': all_labels[:, t]
        })
        cluster_df = pd.concat([cluster_df, df_t], ignore_index=True)
    cluster_df.to_csv(os.path.join(save_dir, 'cluster_assignments.csv'), index=False)

    # 9. 评估与可视化
    features_expanded = torch.tensor(combined_features, dtype=torch.float32)
    class DummyDataset:
        def __init__(self, mode, node_ids):
            self.mode = mode
            self.node_ids = node_ids
    dataset = DummyDataset(mode='train', node_ids=list(buildings_with_ts))
    run_evaluation(config, cluster_assignments, features_expanded, dataset)
    run_visualization(config, cluster_assignments, features_expanded, None, dataset)
    logging.info("基线聚类实验完成")

if __name__ == '__main__':
    main() 
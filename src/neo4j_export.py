import pandas as pd
from neo4j import GraphDatabase
import logging
import time
from neo4j.exceptions import ServiceUnavailable, AuthError
import os

class Neo4jExporter:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="jora1920", database="final", max_retries=3):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_retries = max_retries
        self.logger = logging.getLogger("Neo4jExporter")
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                # Verify connection and database existence
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1")
                self.logger.info(f"Successfully connected to Neo4j database '{self.database}'")
                return
            except (ServiceUnavailable, AuthError) as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts: {str(e)}")
                    raise
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Operation failed after {self.max_retries} attempts: {str(e)}")
                    raise
                self.logger.warning(f"Operation attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)
                # Reconnect if needed
                if not self.driver:
                    self._connect()

    def create_building_nodes(self, static_features_df):
        """
        Create building nodes with static features.
        static_features_df: DataFrame with 'building_id' as index or column.
        """
        self.logger.info(f"Starting to export {len(static_features_df)} building nodes...")
        with self.driver.session(database=self.database) as session:
            for idx, row in static_features_df.iterrows():
                building_id = row.get('building_id', idx)
                props = {k: v for k, v in row.items() if pd.notnull(v)}
                props['building_id'] = str(building_id)
                self._execute_with_retry(session.write_transaction, self._create_building_node, props)
                if idx % 100 == 0:  # Log progress every 100 nodes
                    self.logger.info(f"Processed {idx} building nodes...")

    @staticmethod
    def _create_building_node(tx, props):
        query = """
        MERGE (b:Building {building_id: $building_id})
        SET b += $props
        """
        tx.run(query, building_id=props['building_id'], props=props)

    def create_cluster_relationships(self, cluster_assign_df):
        """
        Create relationships between buildings and clusters for each time step.
        cluster_assign_df: DataFrame with columns ['building_id', 'time_index', 'cluster_id']
        """
        self.logger.info(f"Starting to export {len(cluster_assign_df)} cluster relationships...")
        with self.driver.session(database=self.database) as session:
            for idx, row in cluster_assign_df.iterrows():
                self._execute_with_retry(session.write_transaction, self._create_cluster_rel, 
                                      str(row['building_id']), int(row['time_index']), int(row['cluster_id']))
                if idx % 1000 == 0:  # Log progress every 1000 relationships
                    self.logger.info(f"Processed {idx} cluster relationships...")

    @staticmethod
    def _create_cluster_rel(tx, building_id, time_index, cluster_id):
        query = """
        MERGE (b:Building {building_id: $building_id})
        MERGE (c:Cluster {cluster_id: $cluster_id, time_index: $time_index})
        MERGE (b)-[:IN_CLUSTER {time_index: $time_index}]->(c)
        """
        tx.run(query, building_id=building_id, cluster_id=cluster_id, time_index=time_index)

    def add_energy_info_to_buildings(self, dynamic_features_df):
        """
        Add time series energy info to buildings as properties or relationships.
        dynamic_features_df: DataFrame with ['building_id', 'time_index', ...energy columns...]
        """
        self.logger.info(f"Starting to export {len(dynamic_features_df)} energy records...")
        with self.driver.session(database=self.database) as session:
            for idx, row in dynamic_features_df.iterrows():
                building_id = str(row['building_id'])
                time_index = int(row['time_index'])
                props = {k: v for k, v in row.items() if k not in ['building_id', 'time_index'] and pd.notnull(v)}
                self._execute_with_retry(session.write_transaction, self._add_energy_info, 
                                      building_id, time_index, props)
                if idx % 1000 == 0:  # Log progress every 1000 records
                    self.logger.info(f"Processed {idx} energy records...")

    @staticmethod
    def _add_energy_info(tx, building_id, time_index, props):
        for k, v in props.items():
            tx.run(f"""
                MATCH (b:Building {{building_id: $building_id}})
                SET b.energy_{k}_{time_index} = $value
            """, building_id=building_id, value=v)

    def export_all(self, static_features_df, cluster_assign_df, dynamic_features_df):
        try:
            self.logger.info(f"Starting Neo4j export process to database '{self.database}'...")
            self.logger.info("Exporting static features...")
            self.create_building_nodes(static_features_df)
            self.logger.info("Exporting cluster relationships...")
            self.create_cluster_relationships(cluster_assign_df)
            self.logger.info("Exporting energy info...")
            self.add_energy_info_to_buildings(dynamic_features_df)
            self.logger.info("Export to Neo4j completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during Neo4j export: {str(e)}")
            raise
        finally:
            self.close()

# 添加主函数使文件可以独立运行
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # 静态特征文件路径
        static_features_path = 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/data/raw/buildings_demo.csv'
        # 聚类结果文件路径
        cluster_assign_path = 'D:/GEO2020_Thesis/energy_cluster_gnn_v02/energy_cluster_gnn/results/Scenario_A_No_Constraint_KNN/cluster_assignments.csv'

        # 加载静态特征
        if not os.path.exists(static_features_path):
            raise FileNotFoundError(f"静态特征文件不存在: {static_features_path}")
        static_features_df = pd.read_csv(static_features_path)
        # 如果有 building_id 字段，确保为字符串类型
        if 'building_id' in static_features_df.columns:
            static_features_df['building_id'] = static_features_df['building_id'].astype(str)
        logger.info(f"已加载静态特征: {static_features_df.shape}")

        # 加载聚类结果
        if not os.path.exists(cluster_assign_path):
            raise FileNotFoundError(f"聚类结果文件不存在: {cluster_assign_path}")
        cluster_assign_df = pd.read_csv(cluster_assign_path)
        if 'building_id' in cluster_assign_df.columns:
            cluster_assign_df['building_id'] = cluster_assign_df['building_id'].astype(str)
        logger.info(f"已加载聚类结果: {cluster_assign_df.shape}")

        # 动态特征可以传空DataFrame
        dynamic_features_df = pd.DataFrame()

        # 创建 Neo4j 导出器实例
        logger.info("初始化 Neo4j 导出器...")
        exporter = Neo4jExporter(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="jora1920",
            database="final"
        )

        # 执行导出
        logger.info("开始导出数据到 Neo4j...")
        exporter.export_all(static_features_df, cluster_assign_df, dynamic_features_df)
        logger.info("数据导出完成！")

    except Exception as e:
        logger.error(f"导出过程中发生错误: {str(e)}")
        raise

# Example usage (to be called in your main pipeline):
# from src.neo4j_export import Neo4jExporter
# exporter = Neo4jExporter(
#     uri="bolt://localhost:7687",  # 如果需要，修改为您的 Neo4j 地址
#     user="neo4j",                 # 如果需要，修改为您的用户名
#     password="your_password",     # 如果需要，修改为您的密码
#     database="final"             # 指定目标数据库
# )
# exporter.export_all(static_features_df, cluster_assign_df, dynamic_features_df) 
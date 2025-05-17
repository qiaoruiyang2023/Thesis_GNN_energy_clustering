# src/data_processing.py

import pandas as pd
import numpy as np
import os
import logging
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib # Added import for potential future saving of scalers/encoders

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
id_column = 'building_id' # Define globally for consistency

def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw data files specified in the config."""
    raw_dir = config['RAW_DATA_DIR']
    try:
        # Load main files, ensuring building_id is string
        buildings_df = pd.read_csv(os.path.join(raw_dir, config['BUILDINGS_DEMO_FILE']), dtype={id_column: str})
        assignments_df = pd.read_csv(os.path.join(raw_dir, config['BUILDING_ASSIGNMENTS_FILE']), dtype={id_column: str, 'line_id': str})
        timeseries_df = pd.read_csv(os.path.join(raw_dir, config['TIME_SERIES_LOADS_FILE']), dtype={id_column: str, 'Energy': str})

        # Ensure consistent formatting (strip whitespace)
        buildings_df[id_column] = buildings_df[id_column].str.strip()
        assignments_df[id_column] = assignments_df[id_column].str.strip()
        timeseries_df[id_column] = timeseries_df[id_column].str.strip()
        # Strip line_id too, just in case
        if 'line_id' in assignments_df.columns:
             assignments_df['line_id'] = assignments_df['line_id'].astype(str).str.strip()
        if 'line_id' in buildings_df.columns:
             buildings_df['line_id'] = buildings_df['line_id'].astype(str).str.strip()


        logging.info(f"Loaded buildings demo data: {buildings_df.shape} (Columns: {buildings_df.columns.tolist()})")
        logging.info(f"Loaded building assignments data: {assignments_df.shape} (Columns: {assignments_df.columns.tolist()})")
        logging.info(f"Loaded time series data: {timeseries_df.shape}")

        return buildings_df, assignments_df, timeseries_df
    except FileNotFoundError as e:
        logging.error(f"Error loading raw data file: {e}")
        raise
    except ValueError as e:
        logging.error(f"Error reading CSV, potentially missing column specified in dtype: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        raise

def normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes time column formats (e.g., H:MM:SS -> HH:MM:SS)."""
    time_columns = {}
    for col in df.columns:
        if isinstance(col, str) and re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", col):
            parts = list(map(int, col.split(':')))
            if len(parts) == 2: # HH:MM
                normalized_time = f"{parts[0]:02d}:{parts[1]:02d}:00"
            elif len(parts) == 3: # HH:MM:SS
                 normalized_time = f"{parts[0]:02d}:{parts[1]:02d}:{parts[2]:02d}"
            else:
                normalized_time = col # Should not happen with regex, but safer
            time_columns[col] = normalized_time
        else:
            time_columns[col] = col # Keep non-time columns as is
    df.rename(columns=time_columns, inplace=True)
    return df

def preprocess_dynamic_features(timeseries_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """优化的动态特征预处理函数"""
    logging.info("Starting dynamic feature processing...")
    
    # 添加输入数据检查
    logging.info(f"Input timeseries_df shape: {timeseries_df.shape}")
    logging.info(f"Sample of building IDs before processing: {timeseries_df[id_column].head().tolist()}")
    
    # 1. 预先过滤所需的列
    energy_types_needed = ['total_electricity', 'generation']
    if config.get('NET_LOAD_DEFINITION') == 'detailed':
        energy_types_needed.extend(['heating', 'cooling', 'facility', 'battery_charge', 'battery_discharge'])
    
    # 2. 一次性转换ID列类型，保持原始ID
    timeseries_df = timeseries_df.copy()
    timeseries_df[id_column] = timeseries_df[id_column].astype(str)
    
    # 3. 预先过滤时间列
    time_cols = [col for col in timeseries_df.columns 
                if re.match(r"^\d{2}:\d{2}:\d{2}$", col)]
    time_cols = sorted(time_cols)
    
    # 4. 使用更高效的pivot方法
    ts_filtered = timeseries_df[timeseries_df['Energy'].isin(energy_types_needed)]
    
    # 使用groupby代替pivot_table，保持原始ID
    pivot_groups = ts_filtered.groupby(['building_id', 'Energy'])[time_cols].first()
    pivot_df = pivot_groups.unstack(level='Energy')
    
    # 获取唯一的建筑ID列表，保持原始ID
    buildings_with_ts = pivot_df.index.unique()
    num_buildings = len(buildings_with_ts)
    num_timesteps = len(time_cols)
    
    logging.info(f"Number of buildings: {num_buildings}")
    logging.info(f"Number of timesteps: {num_timesteps}")
    logging.info(f"Sample of unique building IDs: {buildings_with_ts[:5].tolist()}")
    
    # 5. 计算net_load
    net_loads = np.zeros((num_buildings, num_timesteps))
    for t_idx, t in enumerate(time_cols):
        total_elec_col = (t, 'total_electricity')
        gen_col = (t, 'generation')
        battery_col = (t, 'battery_charge')
        
        # 确保所有需要的列都存在
        total_elec = pivot_df.get(total_elec_col, 0)
        generation = pivot_df.get(gen_col, 0)
        battery = pivot_df.get(battery_col, 0)
        
        net_loads[:, t_idx] = total_elec - generation - battery
    
    # 6. 创建最终的DataFrame
    # 创建重复的building_id和timestamp
    building_ids = np.repeat(buildings_with_ts, num_timesteps)
    timestamps = np.tile(pd.to_datetime(time_cols, format='%H:%M:%S').time, num_buildings)
    net_loads_flat = net_loads.flatten()
    
    # 确保所有数组长度相同
    assert len(building_ids) == len(timestamps) == len(net_loads_flat), \
        f"Array length mismatch: buildings={len(building_ids)}, times={len(timestamps)}, loads={len(net_loads_flat)}"
    
    processed_dynamic_df = pd.DataFrame({
        'building_id': building_ids,
        'timestamp': timestamps,
        'net_load': net_loads_flat
    })
    
    logging.info(f"Final processed_dynamic_df shape: {processed_dynamic_df.shape}")
    
    return processed_dynamic_df, buildings_with_ts


# Modified preprocess_static_features
def preprocess_static_features(buildings_df: pd.DataFrame, assignments_df: pd.DataFrame, buildings_with_ts: np.ndarray, config: dict) -> pd.DataFrame:
    """
    Processes static building features:
    1. Filters buildings_df to match buildings present in time series data.
    2. Ensures 'line_id' column is present (using buildings_df preferentially).
    3. Selects relevant features.
    4. Handles missing values (imputation).
    5. Scales numerical features.
    6. Encodes categorical features.
    """
    logging.info("Starting static feature processing...")

    # 添加更多日志输出来诊断问题
    logging.info(f"Buildings in static data: {buildings_df[id_column].nunique()} unique IDs")
    logging.info(f"Buildings in time series: {len(buildings_with_ts)} unique IDs")
    logging.info(f"Sample of building IDs in static data: {buildings_df[id_column].head().tolist()}")
    logging.info(f"Sample of building IDs in time series: {buildings_with_ts[:5].tolist()}")

    # Ensure building_id columns are clean strings for matching
    buildings_df[id_column] = buildings_df[id_column].astype(str).str.strip()
    assignments_df[id_column] = assignments_df[id_column].astype(str).str.strip()
    buildings_with_ts_str = [str(bid).strip() for bid in buildings_with_ts]

    # 检查ID格式
    logging.info("ID format check:")
    logging.info(f"Static data ID example: '{buildings_df[id_column].iloc[0]}' (type: {type(buildings_df[id_column].iloc[0])})")
    logging.info(f"Time series ID example: '{buildings_with_ts_str[0]}' (type: {type(buildings_with_ts_str[0])})")

    # 检查交集
    common_ids = set(buildings_df[id_column]).intersection(set(buildings_with_ts_str))
    logging.info(f"Number of common building IDs: {len(common_ids)}")

    # --- Filter buildings ---
    static_df = buildings_df[buildings_df[id_column].isin(buildings_with_ts_str)].copy()
    logging.info(f"Filtered static features to {static_df.shape[0]} buildings matching time series.")

    if static_df.empty:
        # 提供更详细的错误信息
        error_msg = (
            "No matching buildings found between static data and time series data.\n"
            f"Static data building IDs: {buildings_df[id_column].unique()[:5]}\n"
            f"Time series building IDs: {buildings_with_ts_str[:5]}"
        )
        raise ValueError(error_msg)

    # --- Ensure 'line_id' column exists ---
    # Check if line_id came from buildings_demo.csv
    if 'line_id' not in static_df.columns:
        logging.warning(f"'line_id' column not found in {config['BUILDINGS_DEMO_FILE']}. "
                        f"Attempting to merge from {config['BUILDING_ASSIGNMENTS_FILE']}.")
        # Filter assignments for relevant buildings
        assignments_filtered = assignments_df[assignments_df[id_column].isin(buildings_with_ts_str)].copy()
        if not assignments_filtered.empty and 'line_id' in assignments_filtered.columns:
            assignments_simple = assignments_filtered[[id_column, 'line_id']].drop_duplicates(subset=[id_column], keep='first')
            # Merge ONLy the line_id column
            static_df = pd.merge(static_df, assignments_simple[[id_column, 'line_id']], on=id_column, how='left')
            logging.info("Merged 'line_id' from assignments file.")
        else:
            logging.error(f"'line_id' also not found or empty in {config['BUILDING_ASSIGNMENTS_FILE']}. Cannot proceed with scenarios requiring line_id.")
            # Depending on requirements, you might raise an error or create a dummy column
            # raise ValueError("line_id column missing from both input files.")
            logging.warning("Creating dummy 'UNKNOWN_LINE' column as line_id was missing.")
            static_df['line_id'] = 'UNKNOWN_LINE'
    else:
        logging.info(f"'line_id' column found in {config['BUILDINGS_DEMO_FILE']}, using this.")
        # Ensure it's string type
        static_df['line_id'] = static_df['line_id'].astype(str)

    # Handle potential NaNs in the line_id column (either from source or failed merge)
    if static_df['line_id'].isnull().any():
        num_missing = static_df['line_id'].isnull().sum()
        logging.warning(f"Found {num_missing} missing 'line_id' values after processing sources. Filling with 'UNKNOWN_LINE'.")
        static_df['line_id'].fillna('UNKNOWN_LINE', inplace=True)

    # Strip whitespace just in case
    static_df['line_id'] = static_df['line_id'].str.strip()

    # --- Feature Selection ---
    numerical_cols_config = config.get('STATIC_NUMERICAL_COLS', [
        'lat', 'lon', 'peak_load_kW', 'solar_capacity_kWp',
        'battery_capacity_kWh', 'battery_power_kW', 'area', 'height'
    ])
    categorical_cols_config = config.get('STATIC_CATEGORICAL_COLS', [
        'has_solar', 'has_battery', 'label', 'age_range', 'building_function'
    ])

    # Ensure selected columns actually exist in the static_df
    all_available_cols = static_df.columns.tolist()
    numerical_cols = [col for col in numerical_cols_config if col in all_available_cols]
    categorical_cols = [col for col in categorical_cols_config if col in all_available_cols]

    # Warn if configured columns are missing
    missing_num = set(numerical_cols_config) - set(numerical_cols)
    missing_cat = set(categorical_cols_config) - set(categorical_cols)
    if missing_num: logging.warning(f"Configured numerical columns not found in data: {missing_num}")
    if missing_cat: logging.warning(f"Configured categorical columns not found in data: {missing_cat}")

    logging.info(f"Using numerical features: {numerical_cols}")
    logging.info(f"Using categorical features: {categorical_cols}")

    # Select columns to keep (ID, line_id for graph, and selected features)
    cols_to_keep = [id_column, 'line_id'] + numerical_cols + categorical_cols
    # Ensure unique columns and that they exist
    cols_to_keep_unique = list(dict.fromkeys(cols_to_keep))
    cols_to_keep_final = [col for col in cols_to_keep_unique if col in static_df.columns]

    static_features = static_df[cols_to_keep_final].copy()

    # --- Handle Missing Values ---
    # Numerical Imputation (only if numerical_cols exist)
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        static_features[numerical_cols] = num_imputer.fit_transform(static_features[numerical_cols])
        logging.info("Imputed missing numerical static features using median.")

    # Categorical Imputation (only if categorical_cols exist)
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        static_features[categorical_cols] = cat_imputer.fit_transform(static_features[categorical_cols])
        logging.info("Imputed missing categorical static features using most frequent.")
        # Convert boolean columns back to bool after imputation if necessary
        for col in ['has_solar', 'has_battery']:
            if col in static_features.columns and col in categorical_cols:
                 # Check dtype after imputation, astype might fail if column is not object/string
                 if pd.api.types.is_object_dtype(static_features[col]) or pd.api.types.is_string_dtype(static_features[col]):
                     # Convert 'True'/'False' strings or similar back to bool
                     static_features[col] = static_features[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(False).astype(bool)
                 else:
                     try:
                         static_features[col] = static_features[col].astype(bool)
                     except Exception as e_bool:
                         logging.warning(f"Could not convert imputed column '{col}' back to bool: {e_bool}")

    # --- Scale Numerical Features ---
    scaler_type = config.get('FEATURE_SCALING_METHOD', 'StandardScaler')
    scaler = None
    if numerical_cols: # Only apply if numerical columns exist
        if scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            logging.info(f"Skipping numerical feature scaling (method: {scaler_type}).")

        if scaler:
            # Scale only the identified numerical columns
            scaled_data = scaler.fit_transform(static_features[numerical_cols])
            static_features[numerical_cols] = scaled_data
            # Consider saving the scaler:
            # scaler_path = os.path.join(config['PROCESSED_DATA_DIR'], 'static_scaler.joblib')
            # joblib.dump(scaler, scaler_path)
            # logging.info(f"Saved scaler to {scaler_path}")
            logging.info(f"Scaled numerical features using {scaler_type}.")

    # --- Encode Categorical Features ---
    # Exclude boolean-like columns if they were handled separately
    ohe_cols = [col for col in categorical_cols if col not in ['has_solar', 'has_battery']]
    if ohe_cols:
        try:
            # Use handle_unknown='ignore' to prevent errors if new categories appear in future data (e.g., test set)
            # Use sparse_output=False to get a dense numpy array directly
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(static_features[ohe_cols])
            # Get feature names for the new columns
            encoded_feature_names = encoder.get_feature_names_out(ohe_cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=static_features.index)

            # Drop original categorical columns and join encoded ones
            static_features = static_features.drop(columns=ohe_cols)
            static_features = pd.concat([static_features, encoded_df], axis=1)
            # Consider saving the encoder:
            # encoder_path = os.path.join(config['PROCESSED_DATA_DIR'], 'static_encoder.joblib')
            # joblib.dump(encoder, encoder_path)
            # logging.info(f"Saved encoder to {encoder_path}")
            logging.info(f"OneHotEncoded categorical features: {ohe_cols}.")
        except Exception as e_enc:
            logging.error(f"Error during OneHotEncoding for columns {ohe_cols}: {e_enc}")
            raise # Reraise error as this is critical for feature dimensions


    # Convert boolean features to float (0.0 / 1.0) for the model input
    for col in ['has_solar', 'has_battery']:
        if col in static_features.columns:
            # Ensure the column is actually boolean before converting to float
            if static_features[col].dtype == 'bool':
                 static_features[col] = static_features[col].astype(float)
            else:
                 logging.warning(f"Column '{col}' is not boolean type after imputation/encoding, skipping conversion to float.")


    logging.info(f"Final processed static features shape: {static_features.shape} (Columns: {static_features.columns.tolist()})")
    # Set index AFTER all processing on columns is done
    static_features.set_index(id_column, inplace=True)
    return static_features


def process_data(config: dict):
    """优化的主数据处理函数"""
    logging.info("--- Starting Data Processing Pipeline ---")
    
    # 检查输入数据
    timeseries_path = os.path.join(config['RAW_DATA_DIR'], config['TIME_SERIES_LOADS_FILE'])
    logging.info(f"Reading time series data from: {timeseries_path}")
    
    # 1. 只加载必要的列
    buildings_df = pd.read_csv(
        os.path.join(config['RAW_DATA_DIR'], config['BUILDINGS_DEMO_FILE']),
        usecols=lambda x: x in ['building_id', 'lat', 'lon', 'line_id'] + 
                         config.get('STATIC_NUMERICAL_COLS', []) +
                         config.get('STATIC_CATEGORICAL_COLS', []),
        dtype={'building_id': str}
    )
    logging.info(f"Loaded buildings data with shape: {buildings_df.shape}")
    
    assignments_df = pd.read_csv(
        os.path.join(config['RAW_DATA_DIR'], config['BUILDING_ASSIGNMENTS_FILE']),
        usecols=['building_id', 'line_id'],
        dtype={'building_id': str, 'line_id': str}
    )
    logging.info(f"Loaded assignments data with shape: {assignments_df.shape}")
    
    # 读取时间序列数据
    timeseries_df = pd.read_csv(timeseries_path, dtype={'building_id': str})
    logging.info(f"Loaded time series data with shape: {timeseries_df.shape}")
    
    # 处理数据
    processed_dynamic_df, buildings_with_ts = preprocess_dynamic_features(timeseries_df, config)
    processed_static_df = preprocess_static_features(buildings_df, assignments_df, buildings_with_ts, config)
    
    # 使用更高效的数据划分方法
    train_size = int(len(buildings_with_ts) * (1 - config.get('TEST_SIZE', 0.2)))
    train_buildings = np.random.choice(buildings_with_ts, size=train_size, replace=False)
    test_buildings = np.setdiff1d(buildings_with_ts, train_buildings)
    
    logging.info(f"Split data into {len(train_buildings)} training and {len(test_buildings)} test buildings")
    
    # 使用布尔索引进行数据划分
    train_mask = processed_dynamic_df['building_id'].isin(train_buildings)
    train_dynamic_df = processed_dynamic_df[train_mask]
    test_dynamic_df = processed_dynamic_df[~train_mask]
    
    train_static_df = processed_static_df[processed_static_df.index.isin(train_buildings)]
    test_static_df = processed_static_df[processed_static_df.index.isin(test_buildings)]
    
    # 修改保存路径
    if config.get('SAVE_PROCESSED_DATA', True):
        # 创建目录
        train_dir = os.path.join(config['PROCESSED_DATA_DIR'], 'train')
        test_dir = os.path.join(config['PROCESSED_DATA_DIR'], 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # 保存训练集
        train_dynamic_df.to_parquet(os.path.join(train_dir, 'processed_dynamic_features_train.parquet'))
        train_static_df.to_parquet(os.path.join(train_dir, 'processed_static_features_train.parquet'))
        
        # 保存测试集
        test_dynamic_df.to_parquet(os.path.join(test_dir, 'processed_dynamic_features_test.parquet'))
        test_static_df.to_parquet(os.path.join(test_dir, 'processed_static_features_test.parquet'))
        
        logging.info("Saved processed data to disk")
    
    return {
        'train': (train_static_df, train_dynamic_df),
        'test': (test_static_df, test_dynamic_df)
    }


# Example Usage (within main.py or for testing) - No changes needed here
if __name__ == '__main__':
    # Dummy config for testing standalone
    # Ensure the config reflects the actual columns available, especially
    # if 'line_id' is NOT in BUILDINGS_DEMO_FILE, the code will try to merge it.
    # If it IS in BUILDINGS_DEMO_FILE, the code will use that one.
    config_test = {
        'RAW_DATA_DIR': 'data/raw/', # Adjust relative path if running directly
        'PROCESSED_DATA_DIR': 'data/processed/',
        'BUILDINGS_DEMO_FILE': 'buildings_demo.csv',
        'BUILDING_ASSIGNMENTS_FILE': 'building_assignments.csv',
        'TIME_SERIES_LOADS_FILE': 'time_series_loads_50b_renamed.csv', # Use renamed file for testing
        'NET_LOAD_DEFINITION': 'total_electricity',
        'IMPUTATION_METHOD': 'zero', # Example: 'zero', 'mean', 'ffill'
        'FEATURE_SCALING_METHOD': 'StandardScaler', # Example: 'StandardScaler', 'MinMaxScaler', None
        'STATIC_NUMERICAL_COLS': ['lat', 'lon', 'peak_load_kW', 'solar_capacity_kWp', 'battery_capacity_kWh', 'battery_power_kW', 'area', 'height'],
        'STATIC_CATEGORICAL_COLS': ['has_solar', 'has_battery', 'label', 'age_range', 'building_function']
    }
     # Adjust raw data path for standalone execution if needed
    # Assuming script is run from project root for standalone test
    config_test['RAW_DATA_DIR'] = os.path.join('..', config_test['RAW_DATA_DIR'])
    config_test['PROCESSED_DATA_DIR'] = os.path.join('..', config_test['PROCESSED_DATA_DIR'])
    # Create dummy processed dir if it doesn't exist
    os.makedirs(config_test['PROCESSED_DATA_DIR'], exist_ok=True)

    print(f"--- Running Standalone Test for data_processing.py ---")
    print(f"Using config: {config_test}")

    try:
        # Create dummy input files if they don't exist for testing
        # (This part is simplified, real testing would need representative dummy data)
        def create_dummy_csv(path, columns):
            if not os.path.exists(path):
                print(f"Creating dummy file: {path}")
                dummy_data = {}
                for i, col in enumerate(columns):
                    if 'id' in col: dummy_data[col] = [f'ID_{j}' for j in range(5)]
                    elif 'lat' in col: dummy_data[col] = np.random.rand(5) * 90
                    elif 'lon' in col: dummy_data[col] = np.random.rand(5) * 180
                    elif 'kW' in col or 'kWh' in col: dummy_data[col] = np.random.rand(5) * 100
                    elif 'has' in col: dummy_data[col] = [True, False, True, False, True]
                    else: dummy_data[col] = [f'Cat_{i}_{j}' for j in range(5)]
                pd.DataFrame(dummy_data).to_csv(path, index=False)

        # create_dummy_csv(os.path.join(config_test['RAW_DATA_DIR'], config_test['BUILDINGS_DEMO_FILE']), ['building_id', 'lat', 'lon', 'peak_load_kW', 'has_solar', 'line_id'])
        # create_dummy_csv(os.path.join(config_test['RAW_DATA_DIR'], config_test['BUILDING_ASSIGNMENTS_FILE']), ['building_id', 'line_id', 'distance_km'])
        # create_dummy_csv(os.path.join(config_test['RAW_DATA_DIR'], config_test['TIME_SERIES_LOADS_FILE']), ['building_id', 'Energy', '00:00:00', '00:15:00'])


        processed_data = process_data(config_test)
        print("\n--- Standalone Test Results ---")
        print("\nProcessed Static Features Head:")
        print(processed_data['train'][0].head())
        print("\nProcessed Dynamic Features Head:")
        print(processed_data['train'][1].head())
        print(f"\nStatic features shape: {processed_data['train'][0].shape}")
        print(f"Dynamic features shape: {processed_data['train'][1].shape}")
        if not processed_data['train'][1].empty:
            print(f"Number of unique buildings in dynamic data: {processed_data['train'][1]['building_id'].nunique()}")
        else:
            print("Dynamic features DataFrame is empty.")

    except FileNotFoundError:
         print("\nError: Raw data files not found. Make sure the paths in the config are correct"
               " and the script is run from the intended directory (e.g., project root via main.py). "
               "If running data_processing.py directly, adjust paths relative to the script location.")
    except Exception as e:
        logging.exception(f"An error occurred during data processing standalone test: {e}")


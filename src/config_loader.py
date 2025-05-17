# src/config_loader.py

import yaml
import os
import collections.abc
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_dict(d, u):
    """
    Recursively update a dictionary.
    Args:
        d (dict): Dictionary to be updated.
        u (dict): Dictionary with updates.
    Returns:
        dict: Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(config_filename: str, config_dir: str = 'configs') -> dict:
    """
    Load configuration from YAML files with proper encoding handling.
    
    Args:
        config_filename (str): Name of the scenario config file
        config_dir (str): Directory containing config files
    
    Returns:
        dict: Combined configuration dictionary
    """
    try:
        # 首先加载基础配置
        base_config_path = os.path.join(config_dir, 'base_config.yaml')
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
            logging.info(f"Loaded base configuration from: {os.path.abspath(base_config_path)}")

        # 然后加载场景特定配置
        scenario_config_path = os.path.join(config_dir, config_filename)
        with open(scenario_config_path, 'r', encoding='utf-8') as f:
            scenario_config = yaml.safe_load(f)
            logging.info(f"Loaded scenario configuration from: {os.path.abspath(scenario_config_path)}")

        # 合并配置
        config = {**base_config, **scenario_config}

        # 确保必要的目录路径存在
        for key in ['PROCESSED_DATA_DIR', 'RESULTS_DIR']:
            if key in config:
                os.makedirs(config[key], exist_ok=True)

        # 添加场景特定的结果目录
        scenario_name = config.get('SCENARIO_NAME', 'Unknown_Scenario')
        config['SCENARIO_RESULTS_DIR'] = os.path.join(config['RESULTS_DIR'], scenario_name)
        os.makedirs(config['SCENARIO_RESULTS_DIR'], exist_ok=True)

        return config

    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise
    except UnicodeDecodeError as e:
        logging.error(f"Encoding error reading configuration file: {e}")
        logging.error("Please ensure all configuration files are saved with UTF-8 encoding")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    try:
        # Example: Load configuration for Scenario B
        scenario_file = 'scenario_B_config.yaml'
        config = load_config(scenario_file)

        print("\n--- Merged Configuration ---")
        # Pretty print the config
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))

        print(f"\nScenario Name: {config.get('SCENARIO_NAME')}")
        print(f"Scenario Type: {config.get('SCENARIO_TYPE')}")
        print(f"Results will be saved in: {config.get('SCENARIO_RESULTS_DIR')}")

    except Exception as e:
        print(f"An error occurred: {e}")
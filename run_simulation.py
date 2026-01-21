import os
import json
import logging
from datetime import datetime
from src.data.data_preprocessor import DataPreprocessor
from src.market.market_simulator import MarketSimulator
from src.b2e.b2erounding import B2ERounding, wait_for_write_tasks
from src.utils.logger import Logger

def setup_logging(config):
    log_dir = config['output']['logPath']
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, config['experiment_name']) 
    os.makedirs(log_dir, exist_ok=True)
    log_file1 = os.path.join(log_dir, config['experiment_name'] + ".log") 
    Logger.setup(log_file = log_file1, console_output=config['output']["console_output"])  # 或 False
    return Logger.get_logger()

def main():
    with open('config/simulation_config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    logger = setup_logging(config)
    logger.info("Starting simulation")
    
    preprocessor = DataPreprocessor(config)
    if not os.path.exists(os.path.join(config['raw_data_path'], 'raw_epochs.pkl')):
        logger.info("Preprocessing raw data")
        preprocessor.preprocess_raw_data()
    logger.info(config)
    # experiment_name = "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    # experiment_name = "two_pools"
    experiment_name = config['experiment_name']
    num_epochs = config['simulation']['num_epochs']
    logger.info(f"Creating new experiment: {experiment_name} with {num_epochs} epochs")
    preprocessor.create_experiment(experiment_name, num_epochs, use_same_data = config['simulation']['use_same_data'], clear_data = False)
    
    simulator = MarketSimulator(config, B2ERounding, wait_for_write_tasks, experiment_name)
    
    logger.info("Starting market simulation")
    simulator.run_simulation(num_epochs)
     
    logger.info("Simulation completed")
    final_state = simulator.get_market_state()
    logger.info("Final market state:")
    logger.info(json.dumps(final_state, indent=2))
    

if __name__ == "__main__":
    main()
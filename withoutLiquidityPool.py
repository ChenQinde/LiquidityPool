import os
import json
import shutil
from src.market.b2etest import B2ETester
from src.b2e.b2erounding import B2ERounding, wait_for_write_tasks
import pandas as pd

def read_broker_data(csv_path):
    """
    Read and process broker data, ensuring correct data types
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure necessary columns exist
    required_columns = [
        'ID', 
        'Number of CTXs served', 
        'revenue rate(Fee * feeRatio / brokerAmount)', 
        'Balance', 
        'Usage'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file is missing required column: {col}")
    
    # Convert data types
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df['revenue rate(Fee * feeRatio / brokerAmount)'] = pd.to_numeric(
        df['revenue rate(Fee * feeRatio / brokerAmount)'], 
        errors='coerce'
    )
    
    # Sort by balance in descending order
    df = df.sort_values(by='Balance', ascending=False)
    
    return df

def copy_data_to_without_folder(base_path, source_folder, num_items=300):
    """
    Copy data from the original folder to the withoutpool/without_foldername directory
    Only copy data1.txt files, not brokerBalance.txt
    """
    source_path = os.path.join(base_path, source_folder)
    
    # Create fixed withoutpool parent directory
    withoutpool_path = os.path.join(base_path, "withoutpool")
    if not os.path.exists(withoutpool_path):
        os.makedirs(withoutpool_path)
        print(f"Created parent directory: {withoutpool_path}")
    
    # Create without_foldername subdirectory under withoutpool
    target_folder = f"without_{source_folder}"
    target_path = os.path.join(withoutpool_path, target_folder)
    
    # Create target folder (if it doesn't exist)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Created directory: {target_path}")
    
    # Copy data1.txt file for each item
    copied_count = 0
    for i in range(num_items):
        item_folder = f"item{i}"
        source_item_path = os.path.join(source_path, item_folder)
        target_item_path = os.path.join(target_path, item_folder)
        
        # Create target item folder
        if not os.path.exists(target_item_path):
            os.makedirs(target_item_path)
        
        # Copy data1.txt file
        source_file = os.path.join(source_item_path, "data1.txt")
        target_file = os.path.join(target_item_path, "data1.txt")
        
        # Copy Ctx.csv file
        source_file1 = os.path.join(source_item_path, "Ctx.csv")
        target_file1 = os.path.join(target_item_path, "Ctx.csv")
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            shutil.copy2(source_file1, target_file1)
            copied_count += 1
            # Print progress every 50 files to reduce console output
            if copied_count % 50 == 0 or copied_count == 1:
                print(f"Copied: {copied_count}/{num_items} files")
        else:
            print(f"Warning: Source file not found {source_file}")
    
    print(f"Copy completed: Total {copied_count}/{num_items} files copied")
    return target_path

def run_b2e_on_folder(folder_path, num_items=300, broker_balance_path=None, additional_brokers=None):
    """
    Run B2E algorithm on each item in the specified folder
    
    Parameters:
    folder_path - Path to the folder to process
    num_items - Number of items to process
    broker_balance_path - Path to broker funds file
    additional_brokers - List of additional broker accounts to add, format: [(broker_id, balance),...]
    """
    # Test configuration
    config = {
        "output_path": folder_path,  # Output path
        "b2e": {
            "varType": "LINEAR",  # Variable type
            "iterNum": 1,  # Number of iterations
            "alpha": 1,  # Alpha parameter
            "feeRatio": 0.1  # Fee ratio
        }
    }
    
    # Progress statistics
    successful_runs = 0
    failed_runs = 0
    
    # Run B2E algorithm for each item
    for i in range(num_items):
        item_folder = f"item{i}"
        item_path = os.path.join(folder_path, item_folder)
        data_file = os.path.join(item_path, "data1.txt")
        
        
        # Update output path
        config["output_path"] = item_path
        
        # Create tester
        tester = B2ETester(config, B2ERoundingClass=B2ERounding)
        
        # Load data
        tester.load_data(data_file)
        
        # If broker funds file is provided, load it
        if broker_balance_path and os.path.exists(broker_balance_path):
            tester.load_broker_funds_from_file(broker_balance_path)
            
            # Add additional broker accounts (if any)
            if additional_brokers:
                tester.add_broker_fund(additional_brokers)
                print(f"Added {len(additional_brokers)} additional broker accounts")
        
        print(item_path)
        # Execute single B2E test
        tester.run_single_test()
        
        # Save results
        tester.save_results()
        
        # Wait for write tasks to complete
        
        # Print statistics
        broker_result_path = os.path.join(item_path, "Broker_result.csv")
        if os.path.exists(broker_result_path):
            df = read_broker_data(broker_result_path)
            
            # Print data statistics
            print(f"\n[{item_folder}] Data statistics:")
            print(f"Total number of Brokers: {len(df)}")
            print(f"Maximum balance: {df['Balance'].max()/1e18:.2e} Ether")
            print(f"Minimum balance: {df['Balance'].min()/1e18:.2e} Ether")
            print(f"Average revenue rate: {df['revenue rate(Fee * feeRatio / brokerAmount)'].mean()*100:.4f}%")
        
        successful_runs += 1
        print(f"Successfully processed {item_folder} ({successful_runs}/{num_items})")
        
    
    wait_for_write_tasks()
    print(f"\nProcessing completed! Success: {successful_runs}, Failed: {failed_runs}")

def main():
    # Set base path - using relative path from project root
    base_path = os.path.join("src", "data", "processed_data")
    
    # Original folder names
    # original_folder = "2pools_20w_300_diff_final_balance_medium1"
    original_folder = "2pools_20w_300_diff_final_balance_fully1"
    
    # Number of items
    num_items = 300
    
    # Copy data to new folder
    new_folder_path = copy_data_to_without_folder(base_path, original_folder, num_items)
    
    # Specify path to broker balance file - using relative path
    broker_balance_path = os.path.join("config", "brokerBalance_fully_balanced.txt")
    
    if not os.path.exists(broker_balance_path):
        broker_balance_path = None
        print("Warning: brokerBalance.txt file not found, broker funds data will not be loaded")
    
    # Optional: Add additional broker accounts, format: [(broker_id, balance),...]
    # Example: [("ExtraBroker1", 1000000000000000000), ("ExtraBroker2", 2000000000000000000)]
    # additional_brokers = [("LiquidityPool1", 1e18), ("LiquidityPool2", 1e18)]
    additional_brokers = []
    
    # Run B2E algorithm on the new folder
    print(f"\nStarting B2E algorithm on {new_folder_path}...")
    run_b2e_on_folder(new_folder_path, num_items, broker_balance_path, additional_brokers)

if __name__ == "__main__":
    main()
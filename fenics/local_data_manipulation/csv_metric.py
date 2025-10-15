import pandas as pd
import os

def make_pandas_df(data) -> pd.DataFrame:
    df = pd.DataFrame(data) 
    return df

def make_csv(df, node_id: int, keyword: str, output_dir):
    
    metrics_folder = f'{output_dir}/metrics'

    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)

    df.to_csv(f'{metrics_folder}/node_{node_id}_metrics_{keyword}.csv', index=False)

def delete_files_by_prefix(directory_path: str, prefix: str):
    #Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"\nError: Directory not found or is inaccessible at path: {directory_path}")
        print("Please check the path and try again.")
        return

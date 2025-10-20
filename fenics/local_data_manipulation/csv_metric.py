import pandas as pd
import os

from fenics.local_data_manipulation.yaml_maker import get_output_dir

def make_pandas_df(data) -> pd.DataFrame:
    df = pd.DataFrame(data) 
    return df

def concat_pandas_df(df1, df2) -> pd.DataFrame:
    df = pd.concat([df1, df2], axis=1)
    return df

def make_csv(df, node_id: int):
    
    output_dir = get_output_dir()
    metrics_folder = f'{output_dir}/metrics'

    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)

    df.to_csv(f'{metrics_folder}/node_{node_id}_metrics.csv', index=False)

def load_csv(node_id):

    output_dir = get_output_dir()
    metrics_folder = f'{output_dir}/metrics'

    df = pd.read_csv(f'{metrics_folder}/node_{node_id}_metrics.csv')
    result_dict = df.to_dict('list')
    return result_dict

def delete_files_by_prefix(directory_path: str, prefix: str):
    #Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"\nError: Directory not found or is inaccessible at path: {directory_path}")
        print("Please check the path and try again.")
        return

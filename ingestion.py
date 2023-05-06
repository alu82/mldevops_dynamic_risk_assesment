import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    dfs = []
    current_dir = os.getcwd()
    record_file_path = f"{current_dir}/{output_folder_path}/ingestedfiles.txt"
    clean_output_folder(f"{current_dir}/{output_folder_path}")
    for file in os.listdir(f"{current_dir}/{input_folder_path}"):
        file_format = file[-4:]
        if file_format == ".csv":
            file_path = f"{current_dir}/{input_folder_path}/{file}"
            df = pd.read_csv(file_path)
            dfs.append(df)
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"{file}\n")
    complete_df = pd.concat(dfs)
    complete_df = complete_df.drop_duplicates(ignore_index=True)
    complete_df.to_csv(f"{current_dir}/{output_folder_path}/finaldata.csv") 
    
def clean_output_folder(output_folder):
    # remove all files we write later to avoid inconsistencies
    if os.path.exists(f"{output_folder}/finaldata.csv"):
        os.remove(f"{output_folder}/finaldata.csv")
    if os.path.exists(f"{output_folder}/ingestedfiles.txt"):
        os.remove(f"{output_folder}/ingestedfiles.txt")
    
    
if __name__ == '__main__':
    merge_multiple_dataframe()


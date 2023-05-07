
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Function to get model predictions
def model_predictions(testdata_path=None):
    #read the deployed model and a test dataset, calculate predictions
    with open(f"{os.getcwd()}/{prod_deployment_path}/trainedmodel.pkl", 'rb') as file:
        model = pickle.load(file)
    
    if testdata_path is None:
        testdata_path = f"{os.getcwd()}/{test_data_path}/testdata.csv"
        
    testdata = pd.read_csv(testdata_path)
    X=testdata.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    if 'exited' in testdata.columns:
        y=testdata['exited'].values.reshape(-1, 1).ravel()
    else:
        y = None
        
    predicted = model.predict(X)
    return y, predicted

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(f"{os.getcwd()}/{dataset_csv_path}/finaldata.csv")
    df_numeric = df.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees', 'exited']]
    means = df_numeric.mean()
    medians = df_numeric.median()
    std = df_numeric.std()
    return means, medians, std

##################Function to get percentage of missing data
def missing_data():
    #calculate summary statistics here
    df = pd.read_csv(f"{os.getcwd()}/{dataset_csv_path}/finaldata.csv")
    nas = list(df.isna().sum()/len(df))
    return nas

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    timings = []
    cmds = [
        f"python3 {os.getcwd()}/ingestion.py",
        f"python3 {os.getcwd()}/training.py"
    ]
    for cmd in cmds:
        starttime = timeit.default_timer()
        os.system(cmd)
        endtime = timeit.default_timer()
        timing = endtime - starttime
        timings.append(timing)
        
    return timings

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    return outdated


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    

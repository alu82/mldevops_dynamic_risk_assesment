import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import pandas as pd
import apicalls

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
ingested_folder_path = os.path.join(config['output_folder_path'])

##################Check and read new data
#first, read ingestedfiles.txt
ingested_files_path = f"{os.getcwd()}/{prod_deployment_path}/ingestedfiles.txt"
ingested_files = []
if os.path.exists(ingested_files_path):
    with open(ingested_files_path, 'r') as f:
        ingested_files = f.read().splitlines()
print(f"Ingested files: {ingested_files}")

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = []
for input_file in os.listdir(f"{os.getcwd()}/{input_folder_path}"):
    file_format = input_file[-4:]
    if file_format == ".csv" and input_file not in ingested_files:
        new_files.append(input_file)
print(f"New files: {new_files}")

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_files)>0:
    ingestion.merge_multiple_dataframe()
    
    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    latest_score_path = f"{os.getcwd()}/{prod_deployment_path}/latestscore.txt"
    latest_score = 0.0
    if os.path.exists(ingested_files_path):
        with open(latest_score_path, 'r') as f:
            latest_score = float(f.read().splitlines()[0])
    print(f"latest f1 score: {latest_score}")
    
    ingested_file_path = f"{os.getcwd()}/{ingested_folder_path}/finaldata.csv"
    new_score = scoring.score_model(ingested_file_path)
    print(f"new f1 score: {new_score}")
    
    
    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if new_score < latest_score:
        print("model drift detected, training new model")
        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        training.train_model()
        deployment.store_model_into_pickle()

        ##################Diagnostics and reporting
        #run reporting.py and api calls for the redeployed model
        reporting.score_model()
        apicalls.call_api() # server has to run
        print("new model trained.")
        







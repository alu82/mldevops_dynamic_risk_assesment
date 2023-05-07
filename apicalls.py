import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path']) 

# requests.get('http://127.0.0.1:8000').content 
#Call each API endpoint and store the responses
test_file_path = f"{os.getcwd()}/{test_data_path}/testdata.csv"
response1 = requests.post(f"{URL}/prediction", data={"data_location":test_file_path}).content 
response2 = requests.get(f"{URL}/scoring").content 
response3 = requests.get(f"{URL}/summarystats").content 
response4 = requests.get(f"{URL}/diagnostics").content 

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
record_file_path = f"{os.getcwd()}/{output_model_path}/apireturns.txt"
with open(record_file_path, 'w') as record_file:
    for response in responses:
        record_file.write(f"{response.decode('utf-8')}\n")


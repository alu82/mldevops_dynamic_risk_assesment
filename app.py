from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    data_location = request.form.get('data_location')
    _, predictions = model_predictions(data_location)
    return str(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    f1score = score_model()
    return str(f1score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():        
    summary = dataframe_summary()
    return str(list(summary))

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():    
    #check timing and percent NA values and outdated packages
    timings = execution_time()
    nas = missing_data()
    outdated = outdated_packages_list()
    return str([timings, nas, outdated])

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

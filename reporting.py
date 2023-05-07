import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import ConfusionMatrixDisplay
import diagnostics



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    y, y_pred = diagnostics.model_predictions()
    
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()
    plt.savefig(f"{os.getcwd()}/{output_model_path}/confusionmatrix.png")


if __name__ == '__main__':
    score_model()

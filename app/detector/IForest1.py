import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import os
import sys
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
# from TSB_UAD.utils.slidingWindows import find_length, printResult
from TSB_UAD.utils.slidingWindows import find_length
# from TSB_UAD.utils.visualisation import plotFig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.detector.Detector import plotFig
from app.detector.Detector import data_preprocess
from sklearn.preprocessing import MinMaxScaler

from TSB_UAD.models.iforest import IForest

def iforest1(df):
    # Data Preprocessing
    # print(df)
    name = "timeseries"
    max_length = 10000

    data = df[:max_length,0].astype(float)

    label = df[:max_length,1]

        
    slidingWindow = find_length(data)
    # X_data = Window(window = slidingWindow).convert(data).to_numpy()
    # print(X_data)

    # data_train = data[:int(0.1*len(data))]
    # data_test = data

    # X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    # X_test = Window(window = slidingWindow).convert(data_test).to_numpy()

    modelName='IForest1'
    clf = IForest()
    # x = X_data
    
    
    clf.fit(data)
    score = clf.decision_scores_

    # Post processing
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    
    # score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        

    # save result as figure
    plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName)
    # plt.savefig(modelName+'.png')
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    fig_dir = os.path.join(parent_dir, 'fig')
    output_file = f"{fig_dir}//{modelName}.png"
    plt.savefig(output_file, dpi=300, format="png")
    plt.close()
    return label, score
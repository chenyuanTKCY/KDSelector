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

from TSB_UAD.models.pca import PCA

def pca(df):
    # Data Preprocessing
    data, label, slidingWindow, name, X_data, X_train, X_test, data_train, data_test = data_preprocess(df)

    modelName='PCA'
    clf = PCA()
    x = X_data
    clf.fit(x)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName)



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
    

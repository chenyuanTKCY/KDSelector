from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd
# import altair as alt
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
import sys
import os
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
# from ..vus.utils.metrics import metricor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TSBUAD.TSB_UAD.vus.utils.metrics import metricor




def data_preprocess(df):
    # Data Preprocessing

    
    # df = pd.read_csv(filepath, header=None).dropna().to_numpy()

    
    name = "timeseries"
    max_length = 40000

    data = df[:max_length,0].astype(float)
    
    label = df[:max_length,1]
    
        
    slidingWindow = find_length(data)
    X_data = Window(window = slidingWindow).convert(data).to_numpy()
    
    data_train = data[:int(0.1*len(data))]
    data_test = data

    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
    return data, label, slidingWindow, name, X_data, X_train, X_test, data_train, data_test

    
def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None):
    grader = metricor()
    range_anomaly = grader.range_convers_new(label)
    
    max_length = len(score)
    if plotRange is None:
        plotRange = [0, max_length]
    
    
    fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
    
    
    f3_ax1 = fig3.add_subplot(gs[0, :-1])
    f3_ax1.plot(data[:max_length], 'k')
    for r in range_anomaly:
        if r[0] == r[1]:
            f3_ax1.plot(r[0], data[r[0]], 'r.')
        else:
            f3_ax1.plot(range(r[0], r[1] + 1), data[r[0]:r[1] + 1], 'r')
    f3_ax1.set_xlim(plotRange)
    
    
    f3_ax2 = fig3.add_subplot(gs[1, :-1])
    f3_ax2.plot(score[:max_length])
    f3_ax2.hlines(np.mean(score) + 3 * np.std(score), 0, max_length, linestyles='--', color='red')
    f3_ax2.set_ylabel('score')
    f3_ax2.set_xlim(plotRange)
    
    
    f3_ax3 = fig3.add_subplot(gs[2, :-1])
    index = label + 2 * (score > (np.mean(score) + 3 * np.std(score)))
    cf = lambda x: 'k' if x == 0 else ('r' if x == 1 else ('g' if x == 2 else 'b'))
    color = np.vectorize(cf)(index[:max_length])
    f3_ax3.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    f3_ax3.legend(
        handles=[
            mpatches.Patch(color='black', label='TN'),
            mpatches.Patch(color='red', label='FN'),
            mpatches.Patch(color='green', label='FP'),
            mpatches.Patch(color='blue', label='TP'),
        ],
        loc='best'
    )
    f3_ax3.set_xlim(plotRange)
    
     
        
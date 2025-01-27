########################################################################
# @author : Chenyuan Zhang
# @when : Winter Semester 2024/2025
# @where : Harbin
# @file : MLClassifier
#
########################################################################

import streamlit as st 
import pandas as pd
import util.download as dl
from time import perf_counter
import os
import sys
import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import eval_feature_based as efb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import train_feature_based as tfb

def train(path,method):

    temp_dir = "app/results"
    ou_path = path
    # st.write(os.listdir(ou_path))
    if os.path.isdir(ou_path) and len(os.listdir(ou_path)) > 0:
        path = os.path.join(ou_path, [file for file in os.listdir(ou_path) if file.endswith('.csv')][0])
        classifier = method
        split = st.slider("Training/Validation Split", min_value=0.0, max_value=1.0, step=0.01, value=0.7)

        
        path_save = os.path.join(temp_dir, "weights")
        os.makedirs(path_save, exist_ok=True)
        self_save_done_training = os.path.join(temp_dir, "done_training")
        os.makedirs(self_save_done_training, exist_ok=True)

        train_time = 0
        
        if st.button(" Start Training"):
            st.info(" Training started... Please wait while the model is being trained.")
            
            tic = perf_counter()
            tfb.train_feature_based(path, classifier, split, eval_model=True, path_save=path_save, self_save_done_training=self_save_done_training)
            train_time = perf_counter() - tic

            st.session_state["classifier_training"] = path_save
            st.success(" Training completed!")
        
            pred_save = os.path.join(temp_dir, "raw_predictions")
            latest_file = dl.find_latest_file(pred_save)
            if latest_file:
                st.session_state['latest_file'] = latest_file
            if 'latest_file' in st.session_state:
                latest_file = st.session_state['latest_file']
                try:
                    df = pd.read_csv(latest_file)  
                    st.dataframe(df, width=800, height=300)
                except Exception as e:
                    st.error(f"Error reading the file: {e}")
            st.write(f"training time: {float(train_time):.2f} seconds")
    else:
        st.warning(" No dataset files found in the selected folder.", icon="⚠️")


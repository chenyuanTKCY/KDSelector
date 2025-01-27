########################################################################
# @author : Chenyuan Zhang
# @when : Winter Semester 2024/2025
# @where : Harbin
# @file : deep_model
#
########################################################################


import matplotlib.pyplot as plt
import streamlit as st
import csv
import sys
import os
import time
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.download as dl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import train_deep_model as tdm
import eval_deep_model as edm
from pathlib import Path
from util.helper import extract_file
import shutil


def find_latest_csv(directory):
    
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")]
    if not files:
        return None  
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def is_folder_empty(folder_path):
    if os.path.exists(folder_path) and os.listdir(folder_path):
        return False  
    return True  


def csv_to_dict(file_path):
    result_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)  
        key_column, value_column = reader.fieldnames[:2]  
        for row in reader:
            key = row[key_column]       
            value = row[value_column]
            result_dict[key] = value   
    return result_dict

def train(path,model,params):
    
    
    path_save = st.session_state["temp_dir"]
    train_dir_path = os.path.join(path_save, "training")
    os.makedirs(train_dir_path, exist_ok=True)
    os.makedirs(path_save, exist_ok=True)
    os.makedirs(os.path.join(path_save, "runs"), exist_ok=True)
    os.makedirs(os.path.join(path_save, "weights"), exist_ok=True)
    os.makedirs(os.path.join(path_save, "done_training"), exist_ok=True)
    
    with st.expander(" Training Settings", expanded=True):
        split = st.slider("Training/Validation Split", min_value=0.0, max_value=1.0, step=0.01, value=0.7)
        epochs = st.slider(" Number of epochs", min_value=1, max_value=100, step=1, value=10)
        batch = st.number_input(" Batch size", min_value=1, max_value=100, step=1, value=64)
    output_dim = None
    lambda_CL = None
    temperature = None
    temperature_soft = None
    alpha = None
    nbits = None
    nbins = None

    with st.expander(" Performance-Informed Selector Learning (PISL) ", expanded=True):
        switchPISL = st.toggle("Open PISL", value=False)
        if switchPISL:
            alpha = st.slider("Relative importance of soft and hard labels $\\alpha$", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
            temperature_soft = st.slider("Temperature $t_{soft}$for soft labels", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    with st.expander(" Meta-Knowledge Integration (MKI) ", expanded=True):
        switch3 = st.toggle("Open MKI", value=False)
        if switch3:
        
            text_dir = Path("./app/text")
            if text_dir.exists():
                shutil.rmtree(text_dir)  
            text_dir.mkdir(parents=True, exist_ok=True)  
            uploaded_texts = st.file_uploader(
            "Upload a text file containing the names of the files to be processed (.zip and .rar supported)",
            type=["zip", "rar"],
            accept_multiple_files=True
            )
            if uploaded_texts:
                for uploaded_text in uploaded_texts:
                    
                    st.warning(f"Processing file: {uploaded_text.name}")

                    
                    local_file_path = text_dir / uploaded_text.name
                    with open(local_file_path, "wb") as f:
                        f.write(uploaded_text.getbuffer())

                    
                    result = extract_file(local_file_path, text_dir)
                    
                   
                    if local_file_path.exists():
                        local_file_path.unlink()     
            output_dim = st.slider("Dimension of feature projection ($H$)", min_value=1, max_value=100, step=1, value=64)
            lambda_CL = st.slider("Importance of $$\mathcal{L}_{MKI} (\lambda)$$",  min_value=0.0, max_value=1.0, step=0.01, value=0.77)
            temperature = st.slider("Temperature $t$ for InfoNCE", min_value=0.0, max_value=1.0, step=0.01, value=0.25)
            
    with st.expander(" Pruning-based Acceleration (PA) ", expanded=True):
        switch1 = st.toggle("Open PA", value=False)
        if switch1:
            nbits = st.slider("Number of bits for LSH", min_value=1, max_value=32, step=1, value=14)
            nbins = st.slider("Number of bins", min_value=1, max_value=32, step=1, value=8)
    
    if st.button(" Start Learning"):
        
        st.write("Processing...")
        time.sleep(3)
        st.info(" Training started... Please wait while the model is being trained.")

        
        metrics_df = pd.DataFrame(columns=["n_epochs", "training_time", "acc", "val_acc", "loss", "val_loss", "top_1","top_2", "top_3", "top_4","train_time", "val_time"])

        
        progress_bar = st.progress(0)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        col1.markdown("<h3 style='text-align: center;'> Loss over Epochs</h3>", unsafe_allow_html=True)
        loss_chart = col1.empty()

        col2.markdown("<h3 style='text-align: center;'> Accuracy over Epochs</h3>", unsafe_allow_html=True)
        accuracy_chart = col2.empty()

        col3.markdown("<h3 style='text-align: center;'> Top-K Validation Accuracy</h3>", unsafe_allow_html=True)
        topk_chart_placeholder = col3.empty()
        
        col4.markdown("<h3 style='text-align: center;'> Train and Validation Time</h3>", unsafe_allow_html=True)
        time_chart_placeholder = col4.empty()
        current_epoch = 0

        
        for temp_results in tdm.train_deep_model(path, model, split, batch, params, epochs, output_dim, nbits = nbits,nbins = nbins,lambda_CL = lambda_CL, temperature = temperature, temperature_soft = temperature_soft,alpha = alpha,path_save=path_save,switch1=switch1, switch3=switch3):
            
            current_epoch += 1

            
            progress = current_epoch / epochs
            progress_bar.progress(progress)

            
            results_df = pd.DataFrame([temp_results])
            metrics_df = pd.concat([metrics_df, results_df], ignore_index=True)
            metrics_df.index = range(1, len(metrics_df) + 1)  
            
            loss_chart.line_chart(metrics_df[["loss", "val_loss"]])
            accuracy_chart.line_chart(metrics_df[["acc", "val_acc"]])

            
            topk_chart_placeholder.line_chart(metrics_df[["top_1","top_2", "top_3", "top_4"]])
            
            
            time_chart_placeholder.line_chart(metrics_df[["train_time", "val_time"]])
            # st.write(f"Epoch {current_epoch}/{epochs}")
            # st.write(f"Training time: {temp_results['training_time']:.2f} seconds")
            # st.write(f"Validation time: {temp_results['val_time']:.2f} seconds")
        
        st.success(" Training completed!")
        progress_bar.progress(1.0)
        
        if "deep_training" not in st.session_state:
            st.session_state["deep_training"] = path_save
        # st.write(f"Results saved in: {path_save}")
        
        
        pred_save = os.path.join(path_save, "raw_predictions")
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
        else:
            st.warning("No CSV file found in the directory.")
            
        train_element_path = "app/results/done_training"
        
        latest_csv_file = find_latest_csv(train_element_path)
        
        result_dict = csv_to_dict(latest_csv_file)
        st.write(f"training time: {float(result_dict['training_time']):.2f} seconds")
        st.write(f"validation time: {float(result_dict['val_time']):.2f} seconds")
        st.write(f"validation accuracy:{float(result_dict['val_acc']):.2f}")


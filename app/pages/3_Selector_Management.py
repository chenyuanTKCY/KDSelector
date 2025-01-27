import streamlit as st
import os
import re
import time
import shutil
import base64
import json
from pathlib import Path
import datetime
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.upload as up

def delete_contents_in_subfolders(folder_path):
    if os.path.exists(folder_path):
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):  
                for content in os.listdir(subfolder_path):
                    content_path = os.path.join(subfolder_path, content)
                    if os.path.isfile(content_path):  
                        os.remove(content_path)
                    elif os.path.isdir(content_path):  
                        shutil.rmtree(content_path)
    else:
        st.warning(f"The folder '{folder_path}' does not exist.")
                    
st.markdown("<h1 style='text-align: center; color: #000000;'>Selector Management Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #000000;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Below is the list of learned NN-based selectors :</p>", unsafe_allow_html=True)


models = [ "convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu",'knn', 'svc_linear', 'decision_tree', 'random_forest', 'mlp', 'ada_boost', 'bayes', 'qda']


# Markdown table header
# markdown_output = ["| id | Selector | Window Size $L$ | $t_{soft}$ | $ \ alpha$ | Output_Dim $H$ | $ \lambda $ | temperature for InfoNCE | nbits for LSH | nbins $p$ |",
                #    "|------|-------|-------------|-----------|-----------|------------|------------|-----------|--------|-------|"]
markdown_output = [
    "| id | Selector | Window Size $L$ | $t_{soft}$ | $ \\alpha$ | Output_Dim $H$ | $ \\lambda $ | Temperature for InfoNCE | nbits for LSH | nbins $p$ |",
    "|:---:|:--------:|:---------------:|:-----------:|:----------:|:--------------:|:------------:|:------------------------:|:-------------:|:----------:|"
]

# Dictionary to store unmatched model folders
unmatched_output = ["| id | Selector | Window Size $L$ |", "|----|-------|-------------|"]

# Dictionary to store matched model folders
matched_models = {}
directory = './app/results/weights'
result_path = './app/results'
# Initialize index for rows
index = 1

# Traverse the directory
for folder in os.listdir(directory):
    folder_path = os.path.join(directory, folder)
    
    # Ensure it's a folder
    if os.path.isdir(folder_path):
        # Check if folder name starts with any model name in the models list
        for model in models:
            if folder.startswith(model):
                matched_models[model] = folder_path
                window_size = int(re.search(r'\d+', folder_path).group())
                # Enter the folder and list all files
                files = os.listdir(folder_path)
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    
                    # Ensure it's a file
                    if os.path.isfile(file_path):
                        # Match the filename using the given pattern
                        pattern = r"(?P<temperature_soft>-?\d+(\.\d+)?)_(?P<alpha>-?\d+(\.\d+)?)_(?P<output_dim>-?\d+)_(?P<lambda_CL>-?\d+(\.\d+)?)_(?P<temperature>-?\d+(\.\d+)?)_(?P<nbits>-?\d+)_(?P<nbins>-?\d+)_model"
                        match = re.match(pattern, file)
                        
                        if match:
                            # Extract variables
                            temperature_soft = match.group("temperature_soft")
                            alpha = match.group("alpha")
                            output_dim = match.group("output_dim")
                            lambda_CL = match.group("lambda_CL")
                            temperature = match.group("temperature")
                            nbits = match.group("nbits")
                            nbins = match.group("nbins")
                            # Add the row to the Markdown output
                            markdown_output.append(
                                f"| {index} |{model} | {window_size}| {temperature_soft} | {alpha} | {output_dim} | {lambda_CL} | {temperature} | {nbits} | {nbins} |"
                            )
                            index += 1
                        else:
                            
                            unmatched_output.append(f"| {index} | {model} | {window_size} |")
                break  # A folder only matches one model

# Display the Markdown table if any entries exist
if len(markdown_output) > 2:
    st.markdown("\n".join(markdown_output))
else:
    st.warning("No matched selector files.")


if len(unmatched_output) > 2:
    st.markdown("<p style='text-align: center;'>Below is the list of learned Non-NN-based selectors </p>", unsafe_allow_html=True)
    st.markdown("\n".join(unmatched_output))


with st.expander("Upload your configurations and weights", expanded=True):
    
    params_file = st.file_uploader("Upload your parameters (JSON)", type=["json"])
    
    uploaded_weights = st.file_uploader("Upload your weights")

    
    if params_file is not None:
        params = json.load(params_file)
        model = params.get("model", "")
        window_size = params.get("window_size", 128)
        temperature_soft = params.get("temperature_soft",None)
        alpha = params.get("alpha",None)
        output_dim = params.get("output_dim",None)
        lambda_CL = params.get("lambda_CL",None)
        temperature = params.get("temperature",None)
        nbits = params.get("nbits",None)
        nbins = params.get("nbins",None)
        if model in ["convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu"]:
            if uploaded_weights is not None:
                if st.button("upload"):
                    directory_path = os.path.join(directory, f"{model}_{window_size}")
                    os.makedirs(directory_path,exist_ok=True)
                    new_filename = f"{temperature_soft}_{alpha}_{output_dim}_{lambda_CL}_{temperature}_{nbits}_{nbins}_model"
                    for folder in os.listdir(directory):
                        folder_path = os.path.join(directory, folder)
                        if folder.startswith(model) and folder.endswith(str(window_size)):
                            
                            with open(folder_path + '/' + new_filename, 'wb') as f:
                                f.write(uploaded_weights.getbuffer())
        else:
            if uploaded_weights is not None:
                if st.button("upload"):
                    directory_path = os.path.join(directory, f"{model}_{window_size}")
                    os.makedirs(directory_path,exist_ok=True)
                    for folder in os.listdir(directory):
                        folder_path = os.path.join(directory, folder)
                        if folder.startswith(model) and folder.endswith(str(window_size)):
                            
                            with open(folder_path + '/' + model + '.pkl', 'wb') as f:
                                f.write(uploaded_weights.getbuffer())


st.write("Delete all results")
if st.button("clear results"):
    delete_contents_in_subfolders(result_path)
    st.success("All results have been deleted.")
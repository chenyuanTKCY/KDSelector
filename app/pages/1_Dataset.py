import streamlit as st
import zipfile
import rarfile
import shutil
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.helper import extract_file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.metrics_loader import MetricsLoader
from utils.config import detector_names
import create_windows_dataset as cws
import generate_features as gf

st.markdown("""
    <h1 style='text-align: center; color: #000000;'> Dataset Uploader</h1>
    <p style='text-align: center;'>Upload your dataset and in zipped or rar format for processing.</p>
    <hr style='border:1px solid #000000;'>
""", unsafe_allow_html=True)

current_dir = Path("./app/data")
st.session_state["temp_dir"] = str(current_dir)

if current_dir.exists():
    shutil.rmtree(current_dir)  
current_dir.mkdir(parents=True, exist_ok=True)  


uploaded_files = st.file_uploader(
    "Upload dataset (.zip and .rar supported)",
    type=["zip", "rar"],
    accept_multiple_files=True
)


  

if uploaded_files:
    
    for uploaded_file in uploaded_files:
        
        st.warning(f"Processing file: {uploaded_file.name}")
        
        
        local_file_path = current_dir / uploaded_file.name
        with open(local_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        result = extract_file(local_file_path, current_dir)
        # st.write(result)
        
       
        if local_file_path.exists():
            local_file_path.unlink()  
        


if not os.listdir(current_dir):
    st.warning(" No timeseries uploaded yet. Please upload your metrics to proceed.")
else:
    
    trail_name = "TSB"
    selected_data_names = st.multiselect("Select Time Series", os.listdir(current_dir))
    selected_metric_detail = st.selectbox(" Select Metric Detail", ["AUC_PR", "AUC_ROC", "VUS_PR", "VUS_ROC"])
    
    
    window_size = st.slider("📏 Select Window Length", min_value=100, max_value=1000, value=128)

    
   
    all_folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]

    
    if st.button(" Start Processing", key="process"):
        st.markdown("<h2 style='color: #000000;'> Data Windowing</h2>", unsafe_allow_html=True)
        st.info(" Processing started... Please wait while the datasets are being processed.")
        cws.create_tmp_dataset(trail_name, "/mnt/data/user3/MSAD_system/app/TSB", 
                               str(current_dir), 
                               "./data/TSB/metrics", 
                               window_size, selected_metric_detail)

        # st.success(" Data Windowing  Processing Completed.")
        

        
        trail_file_name = f"{trail_name}_{window_size}"
        trail_path = os.path.join("/mnt/data/user3/MSAD_system/app","TSB", trail_file_name)

        
        progress_bar = st.progress(0)
        status_text = st.empty()

        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.markdown(f"<p style='text-align: center;'> Processing... {progress}% complete.</p>", unsafe_allow_html=True)

        # status_text.markdown("<p style='text-align: center;'> Generating features...</p>", unsafe_allow_html=True)

        
        gf.generate_features(trail_path, progress_callback=update_progress)
        st.success(" Processing Completed.")
        st.session_state["temp_dir"] = current_dir
        



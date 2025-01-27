import zipfile
import streamlit as st
from zipfile import ZipFile
import rarfile  
import os
import pandas as pd
import sys
def upload_dataset(stri, request,reqdir):
    
    uploaded_file = st.file_uploader(f" Upload a zipped or rar folder of your {request}", type=["zip", "rar"], key=stri)

    if uploaded_file is not None:
       
        file_name = uploaded_file.name
        file_path = os.path.join(reqdir, file_name)
        st.write(reqdir)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_dir = reqdir
        
        if request != "train":
            
            # data_dir = os.path.join(reqdir, "data")
            os.makedirs(data_dir, exist_ok=True)

            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    if request == "data":
                        zip_ref.extractall(data_dir)
                    else:
                        zip_ref.extractall(reqdir)
                st.success(f" Zip file extracted successfully!")
            
            elif file_name.endswith(".rar"):
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    if request == "data":
                        rar_ref.extractall(data_dir)
                    else:
                        rar_ref.extractall(reqdir)
                st.success(f" Rar file extracted successfully!")
                
        elif request == "train":
            afdir = os.path.join(reqdir, file_name.split(".zip")[0])
            os.makedirs(afdir, exist_ok=True)
            if file_name.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(afdir)
                st.success(f" Zip file extracted successfully!")
            
            elif file_name.endswith(".rar"):
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(afdir)
                st.success(f" Rar file extracted successfully!")
        
        extracted_files = os.listdir(reqdir)
        # st.markdown("### Extracted Files")
        # st.write(extracted_files)

        
        for file in extracted_files:
            if file.endswith('.csv'):
                file_path = os.path.join(reqdir, file)
                st.markdown(f"#### 🔍 Previewing {file}")
                df = pd.read_csv(file_path)
                st.dataframe(df.head())

        st.success(" File uploaded and processed successfully!")

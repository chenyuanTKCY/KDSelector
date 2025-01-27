import streamlit as st
import sys
import os
import json
import re
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.deep_model as dp
import util.MLClassifier as ml

def methodConfiguration():
    
    params = None
    with st.expander("Selector Configuration", expanded=True):
        # non title
        model = st.selectbox("", [ "convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu",'knn', 'svc_linear', 'decision_tree', 'random_forest', 'mlp', 'ada_boost', 'bayes', 'qda'])
        if model in [ "convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu"]:
            params = os.path.join('./models/configuration', max([f for f in os.listdir('./models/configuration') if f.startswith(model)], key=len))
            # read the json file and presentation
            with open(params, 'r') as f:
                params_detail = json.load(f)
            st.write("Params detail:")
            st.write(params_detail)
        if model in ["sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu"]:
            model = "sit"
        return model, params


st.markdown("<h1 style='text-align: center; color: #000000;'> Selector Learning</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #000000;'>", unsafe_allow_html=True)

st.session_state['temp_dir'] = "app/results" 
data_path = "./app/TSB/TSB_128"  


model, params = methodConfiguration()
if model in [ "convnet", "inception_time",  "resnet", "sit"]:
    dp.train(data_path,model,params) 
else:
    ml.train(data_path,model)  
        

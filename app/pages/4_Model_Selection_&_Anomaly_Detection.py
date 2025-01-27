import streamlit as st
import pandas as pd
import os
import sys
import io
from pathlib import Path
import shutil
import numpy as np
import plotly.graph_objs as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.helper import add_rect
from util.run_model import run_model
from util.run_detector import run_detector
from time import perf_counter, process_time
st.session_state['temp_dir'] = "app/results" 

st.markdown("<h1 style='text-align: center; color: #000000;'> Model Selection and Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #000000;'>", unsafe_allow_html=True)



uploaded_ts = st.file_uploader("Upload your time series")
if uploaded_ts is not None:
    
    ts_data_raw = pd.read_csv(uploaded_ts, header=None).dropna().to_numpy()
    ts_data = ts_data_raw[:, 0].astype(float)
    ts_data = ts_data[:min(len(ts_data), 40000)] 

    # Run model on the uploaded data
    pred_detector, counter_dict, voting_time = run_model(ts_data)
    if pred_detector is not None:
        st.markdown("Voting results:")
        st.bar_chart(counter_dict)
        st.markdown(f"The TSAD model to select is {pred_detector}")
        st.write(f"The time taken for selecting is: {voting_time:.2f} seconds")
        # # Plot time series and detected anomalies
        trace_scores_upload = [go.Scattergl(x=list(range(len(ts_data))), y=ts_data,
                                            mode='lines', line=dict(color='blue', width=3),
                                            name="Time series", yaxis='y2')]
        tic = perf_counter()
        if len(ts_data_raw[0]) > 1:
            label,score = run_detector(pred_detector, ts_data_raw)
            detecting_time = perf_counter() - tic
            # label_data = ts_data_raw[:, 1]
            label_data = label[:min(len(label), 40000)]
            anom = add_rect(label, ts_data)
            
                    
            anom1 = [np.nan if value is None else value for value in anom]
            
            # Convert the 'anom' array to a DataFrame
            anom_df = pd.DataFrame(anom1, columns=["Anomaly"])

            # Convert the DataFrame to a CSV buffer in binary format using BytesIO
            csv_buffer = io.BytesIO()
            anom_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)  # Rewind the buffer to the beginning
            
            # Add a download button for the CSV file
            st.download_button(
                label="Download Anomalies as CSV",
                data=csv_buffer,
                file_name="anom_data.csv",
                mime="text/csv"
            )
            trace_scores_upload.append(go.Scattergl(x=list(range(len(ts_data))), y=anom,
                                                    mode='lines', line=dict(color='red', width=3),
                                                    name="Anomalies", yaxis='y2'))
            #   # Plot scores
            trace_scores_upload.append(go.Scattergl(x=list(range(len(ts_data))), y=score,
                                                    mode='lines', line=dict(color='green', width=2, dash='dot'),
                                                    name="Anomalies Scores", yaxis='y'))

        # Define layout for uploaded data plot
        layout_upload = go.Layout(
            yaxis=dict(domain=[0, 0.4], range=[0, 1]),
            yaxis2=dict(domain=[0.45, 1], range=[min(ts_data), max(ts_data)]),
            title="Uploaded time series snippet (40k points maximum)",
            template="simple_white",
            margin=dict(l=8, r=4, t=50, b=10),
            height=375,
            hovermode="x unified",
            xaxis=dict(range=[0, len(ts_data)])
        )

        # Create and display the plot
        fig = dict(data=trace_scores_upload, layout=layout_upload)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The time taken for detecting is: {detecting_time:.2f} seconds")
    else:
        st.warning("You have not trained the selector yet, please train the selector first")
else:
    st.warning("No time series uploaded yet.")

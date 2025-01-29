import streamlit as st
import base64


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file('app/fig/KD_Logo.png')
html_content = f"""
<div style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
    <img src='data:image/png;base64,{logo_base64}' alt='Logo' style='height: 75px; margin-bottom: 10px;'>
    <h1 style='text-align: center; color: #000000;'>Welcome to KDSelector</h1>
</div>
<p style='text-align: center; font-size: 18px;'>A Knowledge-Enhanced and Data-Efficient Model Selector Learning Framework for Time Series Anomaly Detection.</p>
<hr style='border:1px solid #000000;'>
"""

st.markdown(html_content, unsafe_allow_html=True)

st.markdown(
    """
    <div style='padding: 10px; background-color: #E3F2FD; border-radius: 8px;'>
        <h3>👋 Welcome</h3>
        <p style='font-size: 16px;'>
            This system helps users automatically select the best time series anomaly detection method based on data features. 
            And you can start with the following steps:
            <ul style='font-size: 16px;'>
                <li><strong>Selector Learning</strong>
                </li>
                <li><strong>Model Selection</strong>
                </li>
                <li><strong>Anomaly Detection</strong>
                </li>
            </ul>
        </p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style='padding: 10px; background-color: #FFF3E0; border-radius: 8px; text-align: justify;'>
        <h3>About the System</h3>
        <p style='font-size: 16px;'>
            To fully leverage the detection performance of all TSAD models, we have designed a <strong>Performance-Informed Selector Learning (PISL) Module</strong>. This module transforms the performance scores of different TSAD models into selection probabilities, which serve as soft labels to enhance selector training. Additionally, we have introduced a <strong>Meta-Knowledge Integration (MKI) Module</strong> to extract knowledge from diverse metadata. Furthermore, we propose a novel <strong>Pruning-Based Acceleration (PA) Framework</strong> for NN-based selector training, which can prune more training samples at each epoch while maintaining nearly lossless model accuracy.
        </p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

st.image("app/fig/System_Overview.png", caption="System Overview", use_container_width=True)

st.markdown("""
    <div style='padding: 10px; background-color: #F1F8E9; border-radius: 8px;'>
        <h3>Key Features</h3>
        <ul style='font-size: 16px;'>
            <li><strong>Knowledge-Enhanced Learning</strong>: Integrate additional knowledge from historical data to improve selector accuracy.</li>
            <li><strong>Data-Efficient Training</strong>: Use pruning techniques to reduce training time without sacrificing accuracy.</li>
            <li><strong>Flexible Plugin Design</strong>: PISL, MKI, and PA modules are architecture-agnostic and can be easily integrated into various TSAD model selection tasks.</li>
            <li><strong>High-Performance System</strong>: The system supports multiple TSAD models and selectors, and provides 16 benchmark datasets for evaluation.</li>
            <li><strong>Significant Performance Gains</strong>: Achieve higher accuracy and faster training compared to existing solutions.</li>
            <li><strong>User-Friendly Interface</strong>: Visualize and evaluate selector performance, manage learned selectors, and run anomaly detection with ease.</li>
        </ul>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='padding: 10px; background-color: #E0F7FA; border-radius: 8px;'>
        <h3>📬 Communication</h3>
        <p style='font-size: 16px;'>For questions or feedback, feel free to reach out:</p>
        <p style='font-size: 16px;'>📧 <a href='mailto:zyliang@hit.edu.cn'>zyliang@hit.edu.cn</a></p>
        <p style='font-size: 16px;'>Our code is available at:</p>
        <p style='font-size: 16px;'><a href='http://github.com/chenyuanTKCY/KDSelector'>http://github.com/chenyuanTKCY/KDSelector</a></p>
    </div>
    <br>
""", unsafe_allow_html=True)


st.markdown("""
    <hr>
    <div style='text-align: center; color: #000000;'>
        <p>Made by HIT MDC</p>
    </div>
""", unsafe_allow_html=True)
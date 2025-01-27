import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import rarfile
import plotly.express as px




def extract_file(file_path, output_dir):
    """Extract file to the specified folder"""
    if file_path.suffix == ".zip":
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                return f"Successfully extracted {file_path.name}!"
        except zipfile.BadZipFile:
            return f"File {file_path.name} is not a valid ZIP file."
    elif file_path.suffix == ".rar":
        try:
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                rar_ref.extractall(output_dir)
                return f"Successfully extracted {file_path.name}!"
        except rarfile.BadRarFile:
            return f"File {file_path.name} is not a valid RAR file."
    else:
        return f"File {file_path.name} format is not supported."

def add_rect(label, data):
	"""
	Highlights anomalies in a time series plot.

	Args:
		label (list): List of labels indicating anomalies (1 for anomaly, 0 for normal).
		data (list): List of data points representing the time series.

	Returns:
		list: List containing highlighted anomalies in the time series plot.

	Example:
		highlighted_ts = add_rect([0, 1, 0, 0, 1], [1, 2, 3, 4, 5])
	"""

	# Initialize list for plotting anomalies
	anom_plt = [None] * len(data)

	# Create a copy of the original data
	ts_plt = data.copy()

	# Get the length of the time series
	len_ts = len(data)

	# Loop through labels and data to identify anomalies
	for i, lab in enumerate(label):
		if lab == 1:
			# Mark anomalies and neighboring points
			anom_plt[i] = data[i]
			anom_plt[min(len_ts - 1, i + 1)] = data[min(len_ts - 1, i + 1)]

	return anom_plt


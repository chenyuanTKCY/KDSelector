import streamlit as st
import numpy as np
from collections import Counter
from pathlib import Path
import torch
import os
import re
import joblib
from time import perf_counter, process_time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from utils.train_deep_model_utils import  json_file
from models.model.resnet import ResNetBaseline
from models.model.inception_time import InceptionModel
from models.model.convnet import ConvNet
from models.model.sit import SignalTransformer
from app.util.split_ts import split_ts
from app.util.norm import z_normalization
from utils.config import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC

# Detector (NOTE: run_model.py requires THIS specific order)
detector_names = [
	'AE', 
	'CNN', 
	'HBOS', 
	'IFOREST', 
	'IFOREST1', 
	'LOF', 
	'LSTM', 
	'MP', 
	'NORMA', 
	'OCSVM', 
	'PCA', 
	'POLY'
]
models = [ "convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu",'knn', 'svc_linear', 'decision_tree', 'random_forest', 'mlp', 'ada_boost', 'bayes', 'qda']
device = 'cpu'
def run_model(sequence):
	"""
	"""
	matched_models = {}
	directory = './app/results/weights'

	for folder in os.listdir(directory):
		folder_path = os.path.join(directory, folder)
		
		# Ensure it's a folder
		if os.path.isdir(folder_path):
			# Check if folder name starts with any model name in the models list
			for model in models:
				if folder.startswith(model):
					if os.listdir(folder_path):
						matched_models[model] = folder_path
	method = st.selectbox("Choose a learned selector", [ "convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu",'knn', 'svc_linear', 'decision_tree', 'random_forest', 'mlp', 'ada_boost', 'bayes', 'qda'])
	weights_path = None
	window_size = None
	if method in matched_models:
		weights = st.selectbox("Choose a set of learned weights",os.listdir(matched_models[method]))
		weights_path = os.path.join(matched_models[method], weights)
		window_size = int(re.search(r'\d+', weights_path).group())
	if weights_path is None:
		return None, None, None


	# Load model
	if method in ["convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu"]:
		if method in ["sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu"]:
			method = "sit"
  		
		model_parameters_file = os.path.join('./models/configuration', max([f for f in os.listdir('./models/configuration') if f.startswith(method)], key=len))
		model_parameters = json_file(model_parameters_file)
		# Create the model, load it on GPU and print it
		model = deep_models[method.lower()](**model_parameters).to(device)
	elif method in ['knn', 'svc_linear', 'decision_tree', 'random_forest', 'mlp', 'ada_boost', 'bayes', 'qda']:
		model = joblib.load(weights_path)
	
	tic = perf_counter()
	# deep_learning models process pipeline
	if method in ["convnet", "inception_time",  "resnet", "sit_conv_patch","sit_linear_patch","sit_stem_original","sit_stem_relu"]:
		# Load weights
		model.load_state_dict(torch.load(weights_path, map_location='cpu'))
		model.eval()
		model.to('cpu')
	# non-deep learning models process pipeline
	else:
		pass

	# Normalize
	sequence = z_normalization(sequence, decimals=5)
	# Split timeseries and load to cpu
	sequence = split_ts(sequence, window_size)[:, np.newaxis]
	sequence = torch.from_numpy(sequence).to('cpu')
	print(sequence)
	if method in ["convnet", "inception_time",  "resnet", "sit"]:
		# Generate predictions
		preds = model(sequence.float()).argmax(dim=1).tolist()
	# non-deep learning models process pipeline
	else:
		sequence_flattened = sequence.reshape(sequence.shape[0], -1)[:, :22]
		preds = model.predict(sequence_flattened).astype(int).tolist()
	
	# Majority voting
	counter = Counter(preds)
	most_voted = counter.most_common(1)
	detector = most_voted[0][0]
	
	counter = dict(counter)
	vote_summary = {detector:0 for detector in detector_names}
	for key in counter:
		vote_summary[detector_names[key]] = counter[key]
	voting_time = perf_counter() - tic
	return detector_names[detector], vote_summary, voting_time

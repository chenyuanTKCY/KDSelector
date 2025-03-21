########################################################################
#
# @adpater : Chenyuan Zhang Drongrui Cai
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : train_deep_model_utils
#
########################################################################


import os
from pathlib import Path
import pandas as pd
import copy
import argparse
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from text_process import process_text


import numpy as np
from tqdm import tqdm
from models.blocks import mlp
import matplotlib.pyplot as plt
from time import perf_counter, process_time
from datetime import datetime
from models.blocks import mlp
from InfoBatch.infobatch import InfoBatch
from models.blocks import bert
class ModelExecutioner:
	def __init__(
		self,
		model,
		model_name,
		output_dim,
		batch_size,  
		window_size,
		prune_ratio,
		nbits,
		nbins,
		device='cuda',
		criterion=nn.CrossEntropyLoss(reduction='none'),
		use_scheduler=False,
		temperature_soft = None,
		alpha = None,
		temperature=None,
		n_warmup_steps=4000,
		d_model=256,
		learning_rate=0.00001,
		runs_dir='runs',
		weights_dir='weights',
		lambda_CL=0.5,
		
		# T=0.1
	):
		self.model = model
		self.output_dim = output_dim  
		self.batch_size = batch_size  
		self.runs_dir = runs_dir
		self.weights_dir = weights_dir
		self.model_name = model_name
		self.device = device
		self.criterion = criterion.to(self.device)
		self.use_scheduler = use_scheduler
		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-9
		)
		self.n_warmup_steps = n_warmup_steps
		self.d_model = d_model
		self.training_time_epoch = 0
		self.epoch_best = 0
		self.learning_rate = learning_rate
		self.lambda_CL = lambda_CL
		self.temperature = temperature
		self.temperature_soft = temperature_soft
		self.alpha = alpha
		self.window_size = window_size
		self.prune_ratio = prune_ratio
		self.nbits = nbits
		self.nbins = nbins
		# self.T = T


	def train_one_epoch(self, epoch_index, tb_writer, train_data):
		
		assert isinstance(train_data, InfoBatch), "train_data should be an instance of InfoBatch"
		
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_data.sampler)
		all_loss = []
		all_acc = []

		loop = tqdm(
			enumerate(train_loader),
			total=len(train_loader),
			desc="Epoch [{}/{}]".format(epoch_index, self.n_epochs),
			leave=False,
			unit="batch",
			disable=not self.verbose
		)
		print(self.temperature_soft)
		print(self.alpha)
		print("----------")
		print(self.lambda_CL)
		print(self.temperature)
		print(self.output_dim)
		
		if self.output_dim is not None:
			mlp_ts = mlp.MLP(input_dim=int(self.window_size), output_dim=self.output_dim).to(self.device)
		for i, (inputs, labels, soft_labels, texts) in loop:
			inputs = inputs.to(self.device, dtype=torch.float32)
			labels = labels.to(self.device, dtype=torch.float32)
			soft_labels = soft_labels.to(self.device, dtype=torch.float32)
			if self.output_dim is not None:
				text_features = process_text(texts, self.device, output_dim=self.output_dim)

				inputs_ts = inputs.squeeze(1).to(self.device, dtype=torch.float32)
				x_ts = mlp_ts(inputs_ts)

				
				q = nn.functional.normalize(x_ts, dim=1)
				k = nn.functional.normalize(text_features, dim=1)
				logits = torch.einsum('nc,ck->nk', [q, k.t()])
				logits /= 0.1
				labels_CL = torch.arange(q.shape[0], dtype=torch.long).to(self.device)
				loss_CL = nn.CrossEntropyLoss()(logits, labels_CL)
			# Make predictions for this batch
			outputs = self.model(inputs.float()).to(self.device)
			# Compute the loss and the gradients
			loss_CE_hardlabel = self.criterion(outputs.float(), labels.long())
			# print(loss_CE_hardlabel)

			if self.temperature_soft is not None:

				soft_labels = torch.softmax(soft_labels / self.temperature_soft, dim=-1)
				loss_CE_softlabel = torch.nn.CrossEntropyLoss(reduction='none')(outputs, soft_labels)
				
				if self.output_dim is not None:
					loss = (1 - self.alpha) * loss_CE_hardlabel + self.alpha * loss_CE_softlabel + self.lambda_CL * loss_CL
				else:
					loss = (1 - self.alpha) * loss_CE_hardlabel + self.alpha * loss_CE_softlabel
			else:
				if self.output_dim is not None:
					loss = loss_CE_hardlabel + self.lambda_CL * loss_CL 
				else:
					loss  = loss_CE_hardlabel
				loss = loss.unsqueeze(0).expand(self.batch_size).clone()
				# print(f"values shape: {loss.shape}")
				# print(f"values :{loss}")

			
			# train_data.update(loss_InfoBtach)
			# print(loss.shape[0])
			train_data.update(loss)

			loss = loss.mean()

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# Compute the accuracy
			_, predictions = torch.max(outputs, 1)
			correct = (predictions == labels).sum().item()
			accuracy = correct / labels.size(0)

			all_loss.append(loss.item())
			all_acc.append(accuracy)

			if i % 10 == 9:
				loop.set_postfix(acc=np.mean(all_acc), loss=np.mean(all_loss))

		return np.mean(all_loss), np.mean(all_acc)

	def evaluate(self, dataloader):
		all_loss = []
		all_acc = []
		all_acc_top_k = []

		# Switch model to eval mode
		self.model.eval()

		# The loop through batches
		with torch.no_grad():
			loop = tqdm(
				enumerate(dataloader),
				total=len(dataloader),
				desc="  validation: ",
				unit="batch",
				leave=False,
				disable=not self.verbose,
			)
			for i, batch in loop:
				# Adjust the unpacking based on the number of items in the batch
				if len(batch) == 4:
					inputs, labels, soft_labels, texts = batch
				elif len(batch) == 3:
					inputs, labels, texts = batch
				else:
					inputs, labels = batch

				# Move data to the same device as model
				inputs = inputs.to(self.device, dtype=torch.float32)
				labels = labels.to(self.device, dtype=torch.float32)

				# Make predictions for this batch
				outputs = self.model(inputs.float()).to(self.device)

				# Compute the loss
				loss = self.criterion(outputs.float(), labels.long())

				# Compute top k accuracy
				acc_top_k = self.compute_topk_acc(outputs, labels, k=4)

				all_loss.append(loss.item())
				all_acc.append(acc_top_k[1])
				all_acc_top_k.append(acc_top_k)

				# Report on progress bar
				if i % 10 == 9:
					loop.set_postfix(
						val_loss=np.mean(all_loss),
						val_acc=np.mean(all_acc),
					)

		return np.mean(all_loss), np.mean(all_acc), all_acc_top_k

	# def train(self, n_epochs, training_loader, validation_loader, verbose=True):
	def train(self, n_epochs, train_data, validation_loader, verbose=True):
		
		# train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_data.sampler)
		timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
		writer = SummaryWriter(self.runs_dir + '/{}_{}'.format(self.model_name, timestamp))
		self.n_epochs = n_epochs
		self.verbose = verbose
		best_val_loss = np.Inf
		best_val_acc = 0
		best_model = None

		# Set up early stop
		early_stopper = EarlyStopper(patience=50, min_delta=0)

		# Set up scheduler
		if self.use_scheduler:
			self.scheduler = ScheduledOptim(
				self.optimizer,
				lr_mul=.75,
				d_model=self.d_model,
				n_warmup_steps=self.n_warmup_steps,
			)

		# Check if saving dirs exist (if not create them)
		model_path = os.path.join(
			self.weights_dir,
			self.model_name,
			'{}_{}_{}_{}_{}_{}_{}_{}_model'.format(self.temperature_soft,self.alpha,self.output_dim,self.lambda_CL,self.temperature,self.prune_ratio,self.nbits,self.nbins)
		)
		Path(os.path.join(self.weights_dir, self.model_name))\
		   .mkdir(parents=True, exist_ok=True)

		tic = perf_counter()
		all_train_time = 0
		all_val_time = 0
		for epoch in range(n_epochs):
			# Make sure gradient tracking is on and do a pass
			self.model.train(True)
			tic1 = perf_counter()
			# avg_loss, avg_acc = self.train_one_epoch(epoch, writer, training_loader)
			avg_loss, avg_acc = self.train_one_epoch(epoch, writer, train_data)

		
			train_time = perf_counter() - tic1

			all_train_time += train_time
			# We don't need gradients on to do reporting
			self.model.train(False)

			# Run model on validation data to evaluate
			avg_val_loss, avg_val_acc, val_topk_acc = self.evaluate(validation_loader)
		
			val_time = perf_counter() - tic1 - train_time
			all_val_time += val_time
			avg_val_top1 = np.mean([x[1] for x in val_topk_acc])
			avg_val_top2 = np.mean([x[2] for x in val_topk_acc])
			avg_val_top3 = np.mean([x[3] for x in val_topk_acc])
			avg_val_top4 = np.mean([x[4] for x in val_topk_acc])

			# Epoch reporting
			print(
				"Epoch [{}/{}] {:.2f}secs : acc: {:.3f}, val_acc: {:.3f}, loss: {:.3f}, val_loss: {:.3f}, top k val_acc: k=1: {:.3f} k=2: {:.3f} k=3: {:.3f} k=4: {:.3f}"\
				.format(epoch, n_epochs, perf_counter()-tic, avg_acc, avg_val_acc, avg_loss, avg_val_loss, avg_val_top1, avg_val_top2, avg_val_top3, avg_val_top4)
			)
   
			 # Collect the results for each epoch
			temp_results = {
				'n_epochs': epoch + 1,
				'training_time': perf_counter() - tic,
				'acc': avg_acc,
				'val_acc': avg_val_acc,
				'loss': avg_loss,
				'val_loss': avg_val_loss,
				'top_1': avg_val_top1,
				'top_2': avg_val_top2,
				'top_3': avg_val_top3,
				'top_4': avg_val_top4,
				'train_time': train_time,
				'val_time': val_time
			}
   
			# Log the running loss averaged per batch
			writer.add_scalars('Training vs. Validation Accuracy',
				{'Training': avg_acc, 'Validation': avg_val_acc},
				epoch + 1
			)
			writer.add_scalars('Training vs. Validation Loss',
				{'Training': avg_loss, 'Validation': avg_val_loss},
				epoch + 1
			)
			writer.flush()


			# Track best performance and save the model's state
			if avg_val_acc > best_val_acc:
				best_val_acc = avg_val_acc
				best_model = copy.deepcopy(self.model)
				torch.save(self.model.state_dict(), model_path)

			# Early stopping
			if (epoch > 3 and early_stopper.early_stop_acc(avg_val_acc)) or ((perf_counter()-tic) > 70000):
				break
			yield best_model,temp_results
		# Collect the results
		results = {
			'n_epochs': epoch + 1,
			# 'training_time': perf_counter()-tic,
			'training_time': all_train_time,
			'val_time': all_val_time,
			'acc': avg_acc,
			'val_acc': avg_val_acc,
			'loss': avg_loss,
			'val_loss': avg_val_loss,
			'top_2_val_acc': avg_val_top2,
			'top_3_val_acc': avg_val_top3,
			'top_4_val_acc': avg_val_top4,
		}

		return best_model, results

	def normal_train_one_epoch(self, epoch_index, tb_writer, training_loader):
		all_loss = []
		all_acc = []
		
		loop = tqdm(
			enumerate(training_loader),
			total=len(training_loader),
			desc="Epoch [{}/{}]".format(epoch_index, self.n_epochs),
			leave=False,
			unit="batch",
			disable=not self.verbose
		)
		print(self.temperature_soft)
		print(self.alpha)
		print("----------")
		print(self.lambda_CL)
		print(self.temperature)
		print(self.output_dim)
		
		if self.output_dim is not None:
			mlp_ts = mlp.MLP(input_dim=int(self.window_size), output_dim=self.output_dim).to(self.device)
		for i, (inputs, labels, soft_labels, texts) in loop:
			inputs = inputs.to(self.device, dtype=torch.float32)
			labels = labels.to(self.device, dtype=torch.float32)
			soft_labels = soft_labels.to(self.device, dtype=torch.float32)
			if self.output_dim is not None:
				text_features = process_text(texts, self.device, output_dim=self.output_dim)

				
				inputs_ts = inputs.squeeze(1).to(self.device, dtype=torch.float32)
				x_ts = mlp_ts(inputs_ts)

				
				q = nn.functional.normalize(x_ts, dim=1)
				k = nn.functional.normalize(text_features, dim=1)
				logits = torch.einsum('nc,ck->nk', [q, k.t()])
				logits /= 0.1
				labels_CL = torch.arange(q.shape[0], dtype=torch.long).to(self.device)
				loss_CL = nn.CrossEntropyLoss()(logits, labels_CL)
			# Make predictions for this batch
			outputs = self.model(inputs.float()).to(self.device)
			# Compute the loss and the gradients
			loss_CE_hardlabel = self.criterion(outputs.float(), labels.long())
			# print(loss_CE_hardlabel)

			if self.temperature_soft is not None:

				soft_labels = torch.softmax(soft_labels / self.temperature_soft, dim=-1)
				loss_CE_softlabel = torch.nn.CrossEntropyLoss(reduction='none')(outputs, soft_labels)
				
				if self.output_dim is not None:
					loss = (1 - self.alpha) * loss_CE_hardlabel + self.alpha * loss_CE_softlabel + self.lambda_CL * loss_CL
					loss = loss.mean()

				else:
					loss = (1 - self.alpha) * loss_CE_hardlabel + self.alpha * loss_CE_softlabel
					loss = loss.mean()
			else:
				if self.output_dim is not None:
					loss = loss_CE_hardlabel+ self.lambda_CL * loss_CL
				else:
					loss = loss_CE_hardlabel

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# Compute the accuracy
			_, predictions = torch.max(outputs, 1)
			correct = (predictions == labels).sum().item()
			accuracy = correct / labels.size(0)

			all_loss.append(loss.item())
			all_acc.append(accuracy)

			if i % 10 == 9:
				loop.set_postfix(acc=np.mean(all_acc), loss=np.mean(all_loss))

		return np.mean(all_loss), np.mean(all_acc)

	def normal_train(self, n_epochs, training_loader, validation_loader, verbose=True):
		timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
		writer = SummaryWriter(self.runs_dir + '/{}_{}'.format(self.model_name, timestamp))
		self.n_epochs = n_epochs
		self.verbose = verbose
		best_val_loss = np.Inf
		best_val_acc = 0
		best_model = None

		# Set up early stop
		early_stopper = EarlyStopper(patience=50, min_delta=0)

		# Set up scheduler
		if self.use_scheduler:
			self.scheduler = ScheduledOptim(
				self.optimizer,
				lr_mul=.75,
				d_model=self.d_model,
				n_warmup_steps=self.n_warmup_steps,
			)

		# Check if saving dirs exist (if not create them)
		model_path = os.path.join(
			self.weights_dir,
			self.model_name,
			'{}_{}_{}_{}_{}_{}_{}_{}_model'.format(self.temperature_soft,self.alpha,self.output_dim,self.lambda_CL,self.temperature,self.prune_ratio,self.nbits,self.nbins)
		)
		Path(os.path.join(self.weights_dir, self.model_name))\
		   .mkdir(parents=True, exist_ok=True)


		tic = perf_counter()
		all_train_time = 0
		all_val_time = 0
		for epoch in range(n_epochs):
			tic1 = perf_counter()
			# Make sure gradient tracking is on and do a pass
			self.model.train(True)
			# print("!!!!!!! Epoch {} !!!!!!")
			avg_loss, avg_acc = self.normal_train_one_epoch(epoch, writer, training_loader)
			train_time = perf_counter() - tic1
			all_train_time += train_time
			# We don't need gradients on to do reporting
			self.model.train(False)

			# Run model on validation data to evaluate
			avg_val_loss, avg_val_acc, val_topk_acc = self.evaluate(validation_loader)
			val_time = perf_counter() - train_time - tic1
			all_val_time += val_time
			avg_val_top1 = np.mean([x[1] for x in val_topk_acc])
			avg_val_top2 = np.mean([x[2] for x in val_topk_acc])
			avg_val_top3 = np.mean([x[3] for x in val_topk_acc])
			avg_val_top4 = np.mean([x[4] for x in val_topk_acc])

			# Epoch reporting
			print(
				"Epoch [{}/{}] {:.2f}secs : acc: {:.3f}, val_acc: {:.3f}, loss: {:.3f}, val_loss: {:.3f}, top k val_acc: k=1: {:.3f} k=2: {:.3f} k=3: {:.3f} k=4: {:.3f}"\
				.format(epoch, n_epochs, perf_counter()-tic, avg_acc, avg_val_acc, avg_loss, avg_val_loss, avg_val_top1, avg_val_top2, avg_val_top3, avg_val_top4)
			)

			 # Collect the results for each epoch
			temp_results = {
				'n_epochs': epoch + 1,
				'training_time': perf_counter() - tic,
				'acc': avg_acc,
				'val_acc': avg_val_acc,
				'loss': avg_loss,
				'val_loss': avg_val_loss,
				'top_1': avg_val_top1,
				'top_2': avg_val_top2,
				'top_3': avg_val_top3,
				'top_4': avg_val_top4,
				'train_time': train_time,
				'val_time': val_time
			}
			# Log the running loss averaged per batch
			writer.add_scalars('Training vs. Validation Accuracy',
				{'Training': avg_acc, 'Validation': avg_val_acc},
				epoch + 1
			)
			writer.add_scalars('Training vs. Validation Loss',
				{'Training': avg_loss, 'Validation': avg_val_loss},
				epoch + 1
			)
			writer.flush()

			# Track best performance and save the model's state
			if avg_val_acc > best_val_acc:
				best_val_acc = avg_val_acc
				best_model = copy.deepcopy(self.model)
				torch.save(self.model.state_dict(), model_path)

			# Early stopping
			if (epoch > 3 and early_stopper.early_stop_acc(avg_val_acc)) or ((perf_counter()-tic) > 70000):
				break

			yield best_model,temp_results
		# Collect the results
		results = {
			'n_epochs': epoch + 1,
			# 'training_time': perf_counter()-tic,
			'training_time': all_train_time,
			'val_time': all_val_time,
			'acc': avg_acc,
			'val_acc': avg_val_acc,
			'loss': avg_loss,
			'val_loss': avg_val_loss,
			'top_2_val_acc': avg_val_top2,
			'top_3_val_acc': avg_val_top3,
			'top_4_val_acc': avg_val_top4,
		}

		return best_model, results

	def compute_topk_acc(self, outputs, labels, k=4):
			'''Compute top k accuracy'''
			mean_acc_top_k = {k:[] for k in range(1, k+1)}

			_, y_pred = outputs.topk(k=k, dim=1)  # _, [B, n_classes] -> [B, maxk]
			y_pred = y_pred.t()
			target_reshaped = labels.view(1, -1).expand_as(y_pred)
			correct = (y_pred == target_reshaped)

			for k in range(1, k+1):
				ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
				flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
				tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
				topk_acc = tot_correct_topk / labels.size(0)  # topk accuracy for entire batch
				mean_acc_top_k[k].append(topk_acc.item())

			return mean_acc_top_k


	def torch_devices_info(self):
			print("----------------------------------------------------------------")
			print("Is there a GPU available: {}".format(torch.cuda.is_available()))
			print("Number of allocated devices: {}".format(torch.cuda.device_count()))
			curr_device_id = torch.cuda.current_device()
			print("Index of current device: {}".format(curr_device_id))
			print("Name of current divice: '{}'".format(torch.cuda.get_device_name(curr_device_id)))
			print("Memory allocated:", round(torch.cuda.memory_allocated(curr_device_id)/1024**3, 3), 'GB')
			print("Memory cached:   ", round(torch.cuda.memory_reserved(curr_device_id)/1024**3, 3), 'GB')
			print("----------------------------------------------------------------")

class EarlyStopper:
	def __init__(self, patience=5, min_delta=0.0001):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_val_loss = np.inf
		self.max_val_acc = 0

	def early_stop(self, val_loss):
		if val_loss < self.min_val_loss:
			self.min_val_loss = val_loss
			self.counter = 0
		elif val_loss > (self.min_val_loss - self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False

	def early_stop_acc(self, val_acc):
		if val_acc > self.max_val_acc:
			self.max_val_acc = val_acc
			self.counter = 0
		elif val_acc < (self.max_val_acc + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False


class ScheduledOptim():
	'''A simple wrapper class for learning rate scheduling
					https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
	'''

	def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
		self._optimizer = optimizer
		self.lr_mul = lr_mul
		self.d_model = d_model
		self.n_warmup_steps = n_warmup_steps
		self.n_steps = 0


	def step_and_update_lr(self):
		"Step with the inner optimizer"
		self._update_learning_rate()
		self._optimizer.step()


	def zero_grad(self):
		"Zero out the gradients with the inner optimizer"
		self._optimizer.zero_grad()


	def _get_lr_scale(self):
		d_model = self.d_model
		n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
		return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


	def _update_learning_rate(self):
		''' Learning rate scheduling per step '''

		self.n_steps += 1
		lr = self.lr_mul * self._get_lr_scale()

		for param_group in self._optimizer.param_groups:
			param_group['lr'] = lr

		return lr

	def plot_lr(self, steps=400000):
		lr = []
		tmp_n_steps = self.n_steps
		self.n_steps = 0

		for i in range(steps):
			lr.append(self._update_learning_rate())

		plt.figure(figsize=(10, 8))
		plt.grid(True)
		plt.title('Scheduler d_model = {}'.format(self.d_model))
		plt.plot(lr)
		plt.ylabel('Learning Rate')
		plt.xlabel('Train Step')
		plt.tight_layout()
		plt.show()

		self.n_steps = tmp_n_steps


def json_file(x):
	if not os.path.isfile(x):
		raise argparse.ArgumentTypeError("{} is not a file".format(x))

	try:
		with open(x) as f:
   			variables = json.load(f)
	except Exception as e:
		raise argparse.ArgumentTypeError("{} is not a json file".format(x))

	return variables
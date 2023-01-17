import torch 
from torch.utils.data import DataLoader
from torch import optim 
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np 
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import os
import argparse
import yaml 
from tqdm import tqdm 

from model import Model 
from dataset import SpokenDigitDataset, collate

def train(hp): 
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = Model(hp['n_mfcc'], hp['n_label'], hp['h'], hp['d'], hp['n_lstm']).to(device)
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
	dataset = SpokenDigitDataset(hp['dataset_path'], hp['sampling_rate'], hp['n_mfcc'])
	train_dataset, valid_dataset, test_dataset = dataset.split_dataset(hp['train_valid_test_split'])
	
	accuracy_history = {'train': np.zeros(hp['epochs']), 'valid': np.zeros(hp['epochs'])}	
	
	for epoch in tqdm(range(hp['epochs'])):

		model.train()
		no_audio, accuracy = 0, 0
		for batch, lengths, labels in DataLoader(train_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True): 
			batch = batch.to(device)
			lengths = lengths.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			y = model(batch, lengths)
			y_pred = torch.argmax(y, dim=1)
			loss = criterion(y, labels)
			loss.backward()
			optimizer.step()

			no_audio += len(batch)
			accuracy += (y_pred == labels).sum().item()
		accuracy /= no_audio
		accuracy_history['train'][epoch] = accuracy

		model.eval()
		no_audio, valid_accuracy = 0, 0
		with torch.no_grad(): 
			for batch, lengths, labels in DataLoader(valid_dataset, batch_size=hp['batch_size'], collate_fn=collate, shuffle=True):
				batch = batch.to(device)
				lengths = lengths.to(device)
				labels = labels.to(device)
				y = model(batch, lengths)
				y_pred = torch.argmax(y, dim=1)

				no_audio += len(batch)
				valid_accuracy += (y_pred == labels).sum().item()
			valid_accuracy /= no_audio
			accuracy_history['valid'][epoch] = valid_accuracy

	if not os.path.exists(hp['model_path']): 
		os.mkdir(hp['model_path'])
	torch.save(model.state_dict(), os.path.join(hp['model_path'],'model.pt'))

	plt.figure(figsize=(8, 5))
	plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.plot(accuracy_history['train'], c='r', label='training')
	plt.plot(accuracy_history['valid'], c='b', label='validation')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc='lower right')
	plt.savefig(os.path.join(hp['model_path'],'accuracy.png'))
	plt.clf()

	model.eval()
	batch, lengths, labels = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn=collate, shuffle=True)))
	batch = batch.to(device)
	lengths = lengths.to(device)
	labels = labels.to(device)
	y = model(batch, lengths)
	y_pred = torch.argmax(y, dim=1)
	cm = confusion_matrix(labels.to('cpu').numpy(), y_pred.detach().to('cpu').numpy(), labels=hp['labels'])
	acc = np.diag(cm).sum() / cm.sum()
	df = pd.DataFrame(cm, index=hp['labels'], columns=hp['labels'])
	plt.figure(figsize=(10,7))
	sns.heatmap(df, annot=True)
	plt.title('Confusion matrix for test dataset predictions', fontsize=14)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	b, t = plt.ylim()
	plt.ylim(b + 0.5, t - 0.5)
	plt.savefig(os.path.join(hp['model_path'],'confusion_matrix.png'))
	plt.clf()

	print(f'Train Dataset accuracy: {(accuracy_history["train"][-1])*100:.2f}')
	print(f'Validation Dataset accuracy: {(accuracy_history["valid"][-1]*100):.2f}')
	print(f'Test Dataset Accuracy: {(acc*100):.2f}')

if __name__ == '__main__': 

	parser = argparse.ArgumentParser()
	parser.add_argument('--hyperparameters', type=str, default='hyperparameters.yaml', help='The path for hyperparameters.yaml file')
	args = parser.parse_args()

	hyperparameters = yaml.safe_load(open(args.hyperparameters, 'r'))
	train(hyperparameters)


import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Model(nn.Module): 

	def __init__(self, n_mfcc, n_label, h, d, n_lstm): 
		super().__init__()
		self.lstm_layer = nn.LSTM(input_size=n_mfcc, hidden_size=h, num_layers=n_lstm, batch_first=True, bidirectional=True)
		self.lstm_layer_dropout = nn.Dropout()
		self.linear_layer = nn.Linear(in_features=h*2, out_features=d)
		self.linear_layer_relu = nn.ReLU()
		self.linear_layer_dropout = nn.Dropout()
		self.output_layer = nn.Linear(in_features=d, out_features=n_label)
		self.output_layer_logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x, lengths): 
		batch_size = len(x)
		x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True)
		x, (hn, cn) = self.lstm_layer(x)
		hn = self.lstm_layer_dropout(hn)
		hn = hn.transpose(1, 2).reshape(-1, batch_size).transpose(1, 0)
		hn = self.linear_layer_relu(self.linear_layer(hn))
		hn = self.linear_layer_dropout(hn)
		return self.output_layer_logsoftmax(self.output_layer(hn))


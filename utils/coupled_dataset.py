import torch
import numpy as np


class CoupledDataset(torch.utils.data.Dataset):

	def __init__(self, data1, data2, data_cond1=None, data_cond2=None):

		self.data_cond = None
		if data_cond1 is None:
			data_cond1 = np.zeros([data1.shape[0], 0])
		if data_cond2 is None:
			data_cond2 = np.zeros([data2.shape[0], 0])

		assert data_cond1.shape[0] == data1.shape[0]
		assert data_cond2.shape[0] == data1.shape[0]

		data_cond1 = data_cond1.astype(np.float32)
		data_cond2 = data_cond2.astype(np.float32)

		data1 = data1.astype(np.float32)
		data2 = data2.astype(np.float32)

		self.n_feat_cond1 = data_cond1.shape[1]
		self.n_feat_cond2 = data_cond2.shape[1]

		self.n_feat1 = data1.shape[1]
		self.n_feat2 = data2.shape[1]
		self.n_feat = self.n_feat1 + self.n_feat2

		self.data = np.concatenate([data1, data2, data_cond1, data_cond2], axis=1)

		self.shapes = data1.shape,data2.shape


	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.data[idx, :]




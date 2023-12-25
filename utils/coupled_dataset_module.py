import numpy as np
import pandas as pd
from os.path import join, exists
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.coupled_dataset import CoupledDataset
import torch

class CoupledDatasetModule:
	def __init__(self, data_dir: str, bs: int, mock: bool = False):
		self.data_dir = data_dir
		self.bs = bs
		self._setup_flag = False
		self._split_flag = False
		self.mock = mock

	def setup(self):
		nrows = 100 if self.mock else None
		self.expr, self.pheno, self.expr_cond, self.pheno_cond, self.pheno_vars = self.load_data(self.data_dir, nrows=nrows)
		self.N_samples = len(self.expr)
		self.n_feat1 = self.expr.shape[1]
		self.n_feat2 = self.pheno.shape[1]
		self.n_feat = self.n_feat1 + self.n_feat2
		self.n_feat_cond1 = self.expr_cond.shape[1]
		self.n_feat_cond2 = self.pheno_cond.shape[1]
		self.n_feat_tot = self.n_feat + self.n_feat_cond1 + self.n_feat_cond2
		self._setup_flag = True

	def split(self, test_size=None, validation_size=None, random_state=None, splits_df=None, scaler='zscore'):
		self._check_valid('setup')
		if splits_df is None:
			splits_df = self.split_indices(self.expr.index.tolist(), test_size=test_size, val_size=validation_size, return_df=True, random_state=random_state)
		self.splits_df = splits_df
		self.scaler=scaler
		self._split_flag = True


	def train_dataloader(self):
		return self._get_dataloader('train', self.bs)

	def val_dataloader(self):
		return self._get_dataloader('val')

	def test_dataloader(self):
		return self._get_dataloader('test')

	def full_train_dataloader(self):
		# All training set (1 batch)
		return self._get_dataloader('train')

	def full_dataloader(self):
		return self._get_dataloader('full')

	def train_dataset(self):
		return self._get_dataset('train')

	def val_dataset(self):
		return self._get_dataset('val')

	def trainval_dataset(self):
		return self._get_dataset('trainval')

	def test_dataset(self):
		return self._get_dataset('test')

	def train_sids(self):
		return self.splits_df[self.splits_df=='train'].index.tolist()

	def val_sids(self):
		return self.splits_df[self.splits_df=='val'].index.tolist()

	def test_sids(self):
		return self.splits_df[self.splits_df=='test'].index.tolist()

	def get_expr(self):
		expr, pheno = self.normalize_data(self.expr, self.pheno, self.pheno_vars, self.splits_df, scaler=self.scaler)
		return expr

	def get_pheno(self):
		expr, pheno = self.normalize_data(self.expr, self.pheno, self.pheno_vars, self.splits_df, scaler=self.scaler)
		return pheno

	def get_full_data(self, cond=True, concat=True):
		self._check_valid('split')
		expr, pheno = self.normalize_data(self.expr, self.pheno, self.pheno_vars, self.splits_df, scaler=self.scaler)
		if cond:
			if concat:
				return pd.concat([expr, pheno, self.expr_cond, self.pheno_cond], axis=1)
			else:
				expr, pheno, self.expr_cond, self.pheno_cond
		else:
			if concat:
				return pd.concat([expr, pheno], axis=1)
			else:
				return expr, pheno

	def get_raw_data(self, cond=True, concat=True):
		self._check_valid('setup')
		if cond:
			if concat:
				return pd.concat([self.expr, self.pheno, self.expr_cond, self.pheno_cond], axis=1)
			else:
				return [self.expr, self.pheno, self.expr_cond, self.pheno_cond]
		else:
			if concat:
				return pd.concat([self.expr, self.pheno], axis=1)
			else:
				return [self.expr, self.pheno]


	def _get_dataloader(self, split: str, bs: int = None):
		self._check_valid('split')
		dataset = self._get_dataset(split)

		if bs is None:
			bs = len(dataset)

		if split == 'train':
			shuffle = True
			drop_last = True
		else:
			shuffle = False
			drop_last = False

		return torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

	def _get_dataset(self, split: str):
		self._check_valid('split')
		expr, pheno = self.normalize_data(self.expr, self.pheno, self.pheno_vars, self.splits_df, scaler=self.scaler)
		if split == 'full':
			dataset = CoupledDataset(data1=expr.values,
									 data2=pheno.values,
									 data_cond1=self.expr_cond.values,
									 data_cond2=self.pheno_cond.values)
		elif split == 'trainval':
			dataset = CoupledDataset(data1=expr.loc[self.splits_df.isin(['train','val'])].values,
									 data2=pheno.loc[self.splits_df.isin(['train','val'])].values,
									 data_cond1=self.expr_cond.loc[self.splits_df.isin(['train','val'])].values,
									 data_cond2=self.pheno_cond.loc[self.splits_df.isin(['train','val'])].values)
		else:
			dataset = CoupledDataset(data1=expr.loc[self.splits_df == split].values,
									 data2=pheno.loc[self.splits_df == split].values,
									 data_cond1=self.expr_cond.loc[self.splits_df == split].values,
									 data_cond2=self.pheno_cond.loc[self.splits_df == split].values)


		return dataset

	def _check_valid(self, step: str):
		if step == 'setup' and not self._setup_flag:
			raise ValueError('CoupledDatasetModule: have to run setup() before running this function')
		if step == 'split' and not self._split_flag:
			raise ValueError('CoupledDatasetModule: have to run split() before running this function')


	@staticmethod
	def load_data(data_dir: str, nrows=None):
		expr = pd.read_table(join(data_dir, 'expr.tsv.gz'), index_col=0, nrows=nrows).astype(np.float32)
		if exists(join(data_dir, 'expr.cond.tsv')):
			expr_cond = pd.read_table(join(data_dir, 'expr.cond.tsv'), index_col=0, nrows=nrows).astype(np.float32)
		else:
			expr_cond = pd.DataFrame(np.zeros([expr.shape[0], 0]), index=expr.index)

		pheno = pd.read_table(join(data_dir, 'pheno.tsv'), index_col=0, nrows=nrows).astype(np.float32)
		pheno_vars = pd.read_table(join(data_dir, 'pheno.vars.tsv'), index_col=0)

		if exists(join(data_dir, 'pheno.cond.tsv')):
			pheno_cond = pd.read_table(join(data_dir, 'pheno.cond.tsv'), index_col=0, nrows=nrows).astype(np.float32)
		else:
			pheno_cond = pd.DataFrame(np.zeros([pheno.shape[0], 0]), index=pheno.index)

		return expr, pheno, expr_cond, pheno_cond, pheno_vars

	@staticmethod
	def normalize_data(expr, pheno, pheno_vars, indices=None, scaler='zscore'):
		expr_norm = expr.copy()
		pheno_norm = pheno.copy()
		if indices is None:
			indices = pd.Series("train", index=expr_norm.index)
		num_col_idx = pheno_vars[pheno_vars.feat_type == 'numerical'].i_start.values

		scaler_func = StandardScaler if scaler == 'zscore' else MinMaxScaler

		expr_ss = scaler_func().fit(expr_norm.loc[indices == 'train'])
		expr_norm[:] = expr_ss.transform(expr_norm)

		pheno_ss = scaler_func().fit(pheno_norm.iloc[(indices == 'train').values, num_col_idx])
		pheno_norm.iloc[:, num_col_idx] = pheno_ss.transform(pheno_norm.iloc[:, num_col_idx])

		return expr_norm, pheno_norm

	@staticmethod
	def split_indices(idxs, test_size, val_size=None, return_df=False, random_state=None):
		idxs_train, idxs_test = train_test_split(idxs,test_size=test_size,random_state=random_state)
		if val_size is not None:
			idxs_train, idxs_val = train_test_split(idxs_train,test_size=val_size / (1-test_size),random_state=random_state)
			if return_df:
				df = pd.Series("",index=idxs)
				df[df.index.isin(idxs_train)] = 'train'
				df[df.index.isin(idxs_val)] = 'val'
				df[df.index.isin(idxs_test)] = 'test'
				return df
			else:
				return idxs_train, idxs_val, idxs_test
		else:
			if return_df:
				df = pd.Series("",index=idxs)
				df[df.index.isin(idxs_train)] = 'train'
				df[df.index.isin(idxs_test)] = 'test'
				return df
			else:
				return idxs_train, idxs_test


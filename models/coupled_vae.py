import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import loss_funcs


class CoupledVAE(nn.Module):
	def __init__(self, x_dim1, x_dim2, cond_dim1, cond_dim2, h_dims1, h_dims2, z_dim, vars_meta, p_dropout=0.5, mode=-1, mmd_weight=100, alpha_12=0.5, alpha_numcat=0.5, alpha_l1=0, alpha_l2=0):
		super(CoupledVAE, self).__init__()
		self.n_feat = x_dim1 + x_dim2
		self.n_feat1 = x_dim1
		self.n_feat2 = x_dim2
		self.n_feat_cond1 = cond_dim1
		self.n_feat_cond2 = cond_dim2
		self.n_feat_tot = self.n_feat + self.n_feat_cond1 + self.n_feat_cond2
		self.h_dims1 = h_dims1[:]
		self.h_dims2 = h_dims2[:]
		self.z_dim = z_dim
		self.vars_meta = vars_meta
		self.cat_col_idx = vars_meta[vars_meta.feat_type == 'categorical']
		self.num_col_idx = vars_meta[vars_meta.feat_type == 'numerical'].i_start.values
		self.n_feat2_num = len(self.num_col_idx)
		self.n_feat2_cat = len(self.cat_col_idx)
		self.mmd_weight = mmd_weight
		self.alpha_12 = alpha_12
		self.alpha_numcat = alpha_numcat
		self.alpha_l1 = alpha_l1
		self.alpha_l2 = alpha_l2
		self.p_dropout = p_dropout
		self.mode = mode

		# encoder part
		if mode != 2:
			self.encoder_layers1 = []
			curr_h_dim = x_dim1 + cond_dim1
			for i in range(len(h_dims1)):
				self.encoder_layers1.append(
					nn.Sequential(
						nn.Linear(curr_h_dim, h_dims1[i], bias=True),
						nn.BatchNorm1d(h_dims1[i]),
						nn.ReLU(),
						nn.Dropout(p=self.p_dropout)
					)
				)
				curr_h_dim = h_dims1[i]
			self.encoder1 = nn.Sequential(*self.encoder_layers1)

		if mode != 1:
			self.encoder_layers2 = []
			curr_h_dim = x_dim2 + cond_dim2
			for i in range(len(h_dims2)):
				self.encoder_layers2.append(
					nn.Sequential(
						nn.Linear(curr_h_dim, h_dims2[i], bias=True),
						nn.BatchNorm1d(h_dims2[i]),
						nn.ReLU(),
						nn.Dropout(p=self.p_dropout)
					)
				)
				curr_h_dim = h_dims2[i]
			self.encoder2 = nn.Sequential(*self.encoder_layers2)

		if mode == -1:
			enc_z_dim = h_dims1[-1] + h_dims2[-1]
		elif mode == 1:
			enc_z_dim = h_dims1[-1]
		elif mode == 2:
			enc_z_dim = h_dims2[-1]

		self.encoder_z_mu = nn.Linear(enc_z_dim, z_dim, bias=True)
		self.encoder_z_var = nn.Linear(enc_z_dim, z_dim, bias=True)



		# decoder part
		if mode != 2:
			h_dims1_rev = h_dims1[::-1]
			self.decoder_layers1 = []
			curr_h_dim = z_dim + cond_dim1
			for i in range(len(h_dims1_rev)):
				self.decoder_layers1.append(
					nn.Sequential(
						nn.Linear(curr_h_dim, h_dims1_rev[i], bias=True),
						nn.BatchNorm1d(h_dims1_rev[i]),
						nn.ReLU(),
						nn.Dropout(p=self.p_dropout)
					)
				)
				curr_h_dim = h_dims1_rev[i]
			self.decoder_layers1.append(nn.Linear(h_dims1_rev[-1], x_dim1, bias=True))
			self.decoder1 = nn.Sequential(*self.decoder_layers1)


		if mode != 1:
			h_dims2_rev = h_dims2[::-1]
			self.decoder_layers2 = []
			curr_h_dim = z_dim + cond_dim2
			for i in range(len(h_dims2_rev)):
				self.decoder_layers2.append(
					nn.Sequential(
						nn.Linear(curr_h_dim, h_dims2_rev[i], bias=True),
						nn.BatchNorm1d(h_dims2_rev[i]),
						nn.ReLU(),
						nn.Dropout(p=self.p_dropout)
					)
				)
				curr_h_dim = h_dims2_rev[i]
			self.decoder_layers2.append(nn.Linear(h_dims2_rev[-1], x_dim2, bias=True))
			self.decoder2 = nn.Sequential(*self.decoder_layers2)




	def encoder(self, x):

		x,cond1,cond2 = self.split_sample(x)

		x1 = x[..., :self.n_feat1]
		x2 = x[..., self.n_feat1:]

		x1 = torch.cat([x1, cond1], dim=-1)
		x2 = torch.cat([x2, cond2], dim=-1)

		if self.mode==1:
			h1 = self.encoder1(x1)
			hall = h1
		elif self.mode==2:
			h2 = self.encoder2(x2)
			hall = h2
		else:
			h1 = self.encoder1(x1)
			h2 = self.encoder2(x2)
			hall = torch.cat([h1, h2], dim=-1)

		zmu = self.encoder_z_mu(hall)
		zvar = self.encoder_z_var(hall)
		return zmu, zvar


	def sampling(self, mu, log_var):
		#return mu
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)  # return z sample

	def decoder(self, z):

		z,cond1,cond2 = self.split_z(z)

		z1 = torch.cat([z, cond1], dim=-1)
		z2 = torch.cat([z, cond2], dim=-1)

		if self.mode != 2:
			x1 = torch.sigmoid(self.decoder1(z1))
		else:
			x1 = torch.zeros(list(z1.shape[:-1])+[self.n_feat1])

		if self.mode != 1:
			h2 = self.decoder2(z2)

			x2 = torch.zeros_like(h2)
			x2[..., self.num_col_idx] = torch.sigmoid(h2[..., self.num_col_idx])
			for i in range(len(self.cat_col_idx)):
				i_start, i_end = self.cat_col_idx.iloc[i][['i_start', 'i_end']].astype(int)
				x2[..., i_start:i_end + 1] = F.softmax(h2[..., i_start:i_end + 1], dim=-1)
		else:
			x2 = torch.zeros(list(z1.shape[:-1]) + [self.n_feat2])

		return torch.cat([x1, x2], dim=-1)

	def forward(self, data):

		x,cond1,cond2 = self.split_sample(data)

		mu, log_var = self.encoder(data.view(-1, self.n_feat_tot))
		z = self.sampling(mu, log_var)
		return self.decoder(torch.cat([z, cond1, cond2], dim=-1)), z, mu, log_var

	def loss_func(self, pred, true):
		pred, z, _, _ = pred
		bs = pred.shape[0]

		n_feat1 = self.n_feat1
		n_feat2_num = self.n_feat2_num
		n_feat2_cat = self.n_feat2_cat

		num_col_idx = self.num_col_idx
		cat_col_idx = self.cat_col_idx

		mmd_weight = self.mmd_weight
		mse1 = F.mse_loss(pred[..., :n_feat1], true[..., :n_feat1].view(-1, n_feat1), reduction='sum') / self.n_feat1
		#mse1 = (1-nn.CosineSimilarity(dim=-1)(pred[..., :n_feat1], true[..., :n_feat1].view(-1, n_feat1))).sum(axis=-1)
		#print(nn.CosineSimilarity(dim=-1)(pred[..., :n_feat1], true[..., :n_feat1].view(-1, n_feat1)))
		mse2_num = F.mse_loss(pred[..., n_feat1 + num_col_idx],
							  true[..., n_feat1 + num_col_idx].view(-1, len(num_col_idx)),
							  reduction='sum')

		#print('mse1:',mse1)
		#print('mse2_num:',mse2_num/self.n_feat2_num)

		bce2_cat = 0
		for i in range(len(cat_col_idx)):
			i_start, i_end = cat_col_idx.iloc[i][['i_start', 'i_end']].astype(int)
			i_start += n_feat1
			i_end += n_feat1 + 1
			log_p = torch.log(pred[:, i_start:i_end] + 1e-8)
			#print('bce2_cat:',(-log_p * true[:, i_start:i_end]).sum()/self.n_feat2_cat)
			bce2_cat += (-log_p * true[:, i_start:i_end]).sum()

		#print('bce2_tot:', bce2_cat / self.n_feat2_cat)
		#print('---')

		mmd = loss_funcs.MMD(z)

		l1_term = 0
		l2_term = 0

		if self.alpha_l1 > 0:
			tot_param = 0
			for param in self.parameters():
				tot_param += torch.numel(param)
				l1_term += torch.norm(param,1)
			l1_term = l1_term / tot_param

		if self.alpha_l2 > 0:
			tot_param = 0
			for param in self.parameters():
				tot_param += torch.numel(param)
				l2_term += torch.norm(param,2)
			l2_term = l2_term / tot_param

		alpha_numcat = self.alpha_numcat
		alpha_12 = self.alpha_12

		if self.mode==1:
			alpha_12 = 1
		elif self.mode==2:
			alpha_12 = 0

		norm_factor1 = bs
		norm_factor2_num = bs * self.n_feat2_num
		norm_factor2_cat = bs * self.n_feat2_cat
		norm_factor_mmd = bs * (bs - 1)

		loss1 = alpha_12 * mse1 / norm_factor1
		loss2_num = (1 - alpha_12) * alpha_numcat * mse2_num / norm_factor2_num
		if alpha_numcat < 1:
			loss2_cat = (1 - alpha_12) * (1 - alpha_numcat) * bce2_cat / norm_factor2_cat
		else:
			loss2_cat = torch.tensor(0.)
		loss_mmd = mmd_weight * mmd / norm_factor_mmd
		loss_l1 = l1_term * self.alpha_l1
		loss_l2 = l2_term * self.alpha_l2
		print(loss1, loss2_num, loss2_cat, loss_mmd, loss_l1, loss_l2)

		loss = loss1 + loss2_num + loss2_cat + loss_mmd + loss_l1 + loss_l2
		loss_dict = dict(loss=loss,
						mse1 = mse1,
						mse2_num = mse2_num,
						bce2_cat = bce2_cat,
						mmd = mmd,
						 l1_term = l1_term,
						 l2_term = l2_term,
						batch_size = bs,
						 rec_loss = loss1 + loss2_num + loss2_cat,
						 loss_fractions={'mse1':loss1/loss, 'mse2_num':loss2_num/loss, 'bce2_cat':loss2_cat/loss, 'mmd':loss_mmd/loss, 'l1':loss_l1/loss, 'l2':loss_l2/loss})

		return loss_dict

	def loss_status(self, loss_dict):

		norm_factor1 = loss_dict['batch_size']# * self.n_feat1
		norm_factor2_num = loss_dict['batch_size'] * self.n_feat2_num
		norm_factor2_cat = loss_dict['batch_size'] * self.n_feat2_cat
		norm_factor_mmd = loss_dict['batch_size']



		return "Loss: {:.6f}, ".format(loss_dict['loss'].item()) + \
			   "MSE1: {:.6f} ({:.3f}%), ".format(loss_dict['mse1'].item() / norm_factor1, loss_dict['loss_fractions']['mse1']*100) + \
			   "MSE2: {:.6f} ({:.3f}%), ".format(loss_dict['mse2_num'].item() / norm_factor2_num, loss_dict['loss_fractions']['mse2_num']*100) + \
			   "BCE2: {:.6f} ({:.3f}%), ".format(loss_dict['bce2_cat'].item() / norm_factor2_cat, loss_dict['loss_fractions']['bce2_cat']*100) + \
			   "MMD: {:.6f} ({:.3f}%), ".format(loss_dict['mmd'].item() / norm_factor_mmd, loss_dict['loss_fractions']['mmd']*100) + \
				"L1: {:.6f} ({:.3f}%), ".format(loss_dict['l1_term'],loss_dict['loss_fractions']['l1']*100) + \
				"L2: {:.6f} ({:.3f}%), ".format(loss_dict['l2_term'],loss_dict['loss_fractions']['l2']*100)


	def embed(self, x):
		prev_training = self.training
		self.eval()
		x = torch.tensor(x)
		z, zvar = self.encoder(x.view(-1, self.n_feat_tot))
		if prev_training:
			self.train()
		return z.detach().numpy()

	def reconstruct(self, x):
		prev_training = self.training
		self.eval()
		x = torch.tensor(x)
		out = self(x)[0]

		if prev_training:
			self.train()
		return out.detach().numpy()

	def generate(self, z):
		prev_training = self.training
		self.eval()
		#z, cond1, cond2 = self.split_z(z)
		z = torch.tensor(z)
		out = self.decoder(z)

		if prev_training:
			self.train()
		return out.detach().numpy()

	def split_sample(self, sample):
		x = sample[...,:self.n_feat]
		cond1 = sample[..., self.n_feat:self.n_feat+self.n_feat_cond1]
		cond2 = sample[..., self.n_feat + self.n_feat_cond1:]

		return x,cond1,cond2

	def split_z(self, z):
		z_vec = z[..., :self.z_dim]
		z_cond1 = z[..., self.z_dim:self.z_dim+self.n_feat_cond1]
		z_cond2 = z[..., self.z_dim + self.n_feat_cond1:]

		return z_vec, z_cond1, z_cond2

	def reset(self):
		for layer in self.children():
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

	def serialize(self):
		return {'state_dict': self.state_dict(),
				'hyperparams': \
					dict(x_dim1=self.n_feat1,
						 x_dim2=self.n_feat2,
						 cond_dim1=self.n_feat_cond1,
						 cond_dim2=self.n_feat_cond2,
						 h_dims1=self.h_dims1,
						 h_dims2=self.h_dims2,
						 z_dim=self.z_dim,
						 vars_meta=self.vars_meta,
						 mmd_weight=self.mmd_weight,
						 alpha_12=self.alpha_12,
						 alpha_numcat=self.alpha_numcat,
						 p_dropout=self.p_dropout,
						 mode=self.mode
						 )
				}



	@staticmethod
	def deserialize(obj):
		model = CoupledVAE(**obj['hyperparams'])
		model.load_state_dict(obj['state_dict'])
		return model

import torch
import torch.nn.functional as F

def compute_rbf(x1, x2):
	z_var = 2
	# Convert the tensors into row and column vectors
	D = x1.size(1)
	N = x1.size(0)

	x1 = x1.unsqueeze(-2)  # Make it into a column tensor
	x2 = x2.unsqueeze(-3)  # Make it into a row tensor

	x1 = x1.expand(N, N, D)
	x2 = x2.expand(N, N, D)
	z_dim = x2.size(-1)
	sigma = 2. * z_dim * z_var

	result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
	return result


def MMD(z):
	prior_z = torch.randn_like(z)
	prior_z__kernel = compute_rbf(prior_z, prior_z)
	z__kernel = compute_rbf(z, z)
	priorz_z__kernel = compute_rbf(prior_z, z)
	mmd = prior_z__kernel.mean(-1) + z__kernel.mean(-1) - 2 * priorz_z__kernel.mean(-1)
	return mmd.sum()

def coupled_loss_mmd(pred, true, z, n_feat1, n_feat2, num_col_idx, cat_col_idx, bs, bias_corr = None):
	if bias_corr is None:
		bias_corr = 1 / (bs * (bs - 1))
	BCE1 = F.mse_loss(pred[..., :n_feat1], true[..., :n_feat1].view(-1, n_feat1), reduction='sum')
	BCE2_num = F.mse_loss(pred[..., n_feat1 + num_col_idx], true[..., n_feat1 + num_col_idx].view(-1, len(num_col_idx)),
						  reduction='sum')

	BCE2_cat = 0
	for i in range(len(cat_col_idx)):
		i_start, i_end = cat_col_idx.iloc[i][['i_start', 'i_end']].astype(int)
		i_start += n_feat1
		i_end += n_feat1 + 1
		log_p = torch.log(pred[:, i_start:i_end] + 1e-8)
		BCE2_cat += (-log_p * true[:, i_start:i_end]).sum()

	mmd = MMD(z)

	alpha_numcat = 0.1
	alpha_12 = 0.2
	return ((alpha_12 * BCE1 / n_feat1 + (1 - alpha_12) * \
			 (alpha_numcat * BCE2_num + (1 - alpha_numcat) * BCE2_cat) / n_feat2)) + bias_corr * mmd, \
		   BCE1 / n_feat1, \
		   BCE2_num / n_feat2, \
		   BCE2_cat / n_feat2, \
		   mmd


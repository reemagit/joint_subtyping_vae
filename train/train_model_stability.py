import click


### Training functions

def train(epoch, vae, optimizer, data_loader, verbose=True):
	vae.train()
	train_loss = 0
	total_batches = 0
	for batch_idx, batch in enumerate(data_loader):

		# data = data.cuda()
		optimizer.zero_grad()

		# data is a tuple (x, cond1, cond2)
		pred = vae(batch)
		loss_dict = vae.loss_func(pred, batch)
		loss = loss_dict['loss']
		loss.backward()
		train_loss += loss.item()
		optimizer.step()

		if batch_idx % 10 == 0:
			param_str = vae.loss_status(loss_dict)
			if verbose:
				print(
					'Train Epoch: {} [{}/{} ({:.0f}%)]\t\t\t\t{}'.format(
						epoch, batch_idx * len(batch), len(data_loader.dataset),
							   100. * batch_idx / len(data_loader), param_str))

		total_batches += 1
	if verbose:
		print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / total_batches))

	return train_loss / total_batches


def validate(epoch, vae, data_loader, verbose=True):
	import torch
	vae.eval()

	with torch.no_grad():
		batch = next(iter(data_loader))
		# data = data.cuda()
		pred = vae(batch)

		# sum up batch loss
		loss_dict = vae.loss_func(pred, batch)
		val_loss = loss_dict['rec_loss'].item()
		if verbose:
			print('Test set {}'.format(vae.loss_status(loss_dict)))

	if verbose:
		print('====> Test set loss: {:.4f}'.format(val_loss))

	return loss_dict


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('-t', '--n_times', 'n_times', type=int, default=100)
@click.option('-m', '--mode', 'mode', type=int, default=-1)
@click.option('-n', '--n_iter', 'n_iter', type=int, default=150)
@click.option('-z', '--zdim', 'zdim', type=int, default=10)
@click.option('-bs', '--batch_size', 'bs', type=int, default=32)
@click.option('-h1', '--h1_dims', 'h1', type=str, default="32,16")
@click.option('-h2', '--h2_dims', 'h2', type=str, default="32,16")
@click.option('-p', '--p_drop', 'p_drop', type=float, default=0)
@click.option('-mmd', '--mmd_weight', 'mmd_weight', type=float, default=100)
@click.option('-a12', '--alpha_12', 'alpha_12', type=float, default=0.5)
@click.option('-anc', '--alpha_numcat', 'alpha_numcat', type=float, default=0.5)
@click.option('-l1', '--alpha_l1', 'alpha_l1', type=float, default=0.)
@click.option('-l2', '--alpha_l2', 'alpha_l2', type=float, default=0.)
@click.option('-lr', '--learning_rate', 'lr', type=float, default=1e-3)
@click.option('-r', '--restore_dir', 'restore_dir', type=click.Path(exists=True))
@click.option('-exp', '--experiment', 'experiment', type=str, default=None)
def main(input_dir, output_dir, n_times, mode, n_iter, zdim, bs, h1, h2, p_drop, mmd_weight, alpha_12, alpha_numcat, alpha_l1,
		 alpha_l2, lr, restore_dir, experiment):
	from copy import deepcopy
	from pathlib import Path

	import numpy as np
	import pandas as pd
	import torch
	import torch.optim as optim

	from utils.coupled_dataset_module import CoupledDatasetModule
	from models.coupled_vae import CoupledVAE
	from utils import loader_funcs, setup_logging

	### Setting up logging and tensorboard logging
	if experiment is None:
		experiment = output_dir.split('/')[-1]

	params = locals()

	output_dir = Path(output_dir)
	(output_dir / "embeds").mkdir(parents=True, exist_ok=True)
	(output_dir / "splits").mkdir(parents=True, exist_ok=True)

	logger = setup_logging(output_dir, __name__)

	#writer = SummaryWriter(Path('logs/tb_experiments') / experiment)

	logger.info('Importing data and setting up model')

	### Setting up model and optimizer


	h_dims1 = list(map(int, h1.split(',')))
	h_dims2 = list(map(int, h2.split(',')))

	cdm = CoupledDatasetModule(input_dir, bs=bs)
	cdm.setup()

	for i in range(n_times):
		cdm.split(test_size=0.1, validation_size=0.1, random_state=i, scaler='minmax')

		pheno_vars = cdm.pheno_vars
		splits_df = cdm.splits_df

		vae = CoupledVAE(x_dim1=cdm.n_feat1,
						 x_dim2=cdm.n_feat2,
						 cond_dim1=cdm.n_feat_cond1,
						 cond_dim2=cdm.n_feat_cond2,
						 h_dims1=h_dims1,
						 h_dims2=h_dims2,
						 z_dim=zdim,
						 vars_meta=pheno_vars,
						 mmd_weight=mmd_weight,
						 alpha_12=alpha_12,
						 alpha_numcat=alpha_numcat,
						 alpha_l1=alpha_l1,
						 alpha_l2=alpha_l2,
						 p_dropout=p_drop,
						 mode=mode)

		for param, val in params.items():
			logger.info(param + ': ' + str(val))

		optimizer = optim.Adam(vae.parameters(), lr=lr)

		# loader_funcs.save_splits(splits_df, output_dir / 'splits_df.tsv')

		# logger.info(f'Train data shape: {train_dataset.shapes}')
		# logger.info(f'Validation data shape: {validation_dataset.shapes}')
		# logger.info(f'Test data shape: {test_dataset.shapes}')



		train_loader = cdm.train_dataloader()
		val_loader = cdm.val_dataloader()
		test_loader = cdm.test_dataloader()

		full_train_loader = cdm.full_train_dataloader()

		if torch.cuda.is_available():
			vae.cuda()

		### Training body

		logger.info('Training')

		try:
			best_val_loss = np.inf
			best_val_loss_epoch = 0

			for epoch in range(1, n_iter):
				train_loss = train(epoch, vae, optimizer, train_loader)

				full_train_loss = validate(epoch, vae, full_train_loader, verbose=False)['rec_loss']
				print('Full training set loss: {:.4f}'.format(full_train_loss))

				val_loss_dict = validate(epoch, vae, val_loader)
				val_loss = val_loss_dict['rec_loss']

				print('Overfitting ratio: {:.4f}%'.format((val_loss - full_train_loss) / full_train_loss * 100))

				if val_loss < best_val_loss:
					best_model = deepcopy(vae.state_dict())
					best_val_loss_dict = val_loss_dict
					best_val_loss = val_loss
					best_val_loss_epoch = epoch
					torch.save(vae.serialize(), output_dir / 'model.pth')
					print('*** NEW BEST ***')

				print('Best Test set loss: {:.4f} (epoch {})'.format(best_val_loss, best_val_loss_epoch))

				#writer.add_scalar("Loss/train", train_loss, epoch)
				#writer.add_scalar("Loss/full_train", full_train_loss, epoch)

				#for key, val in val_loss_dict.items():
				#	if not isinstance(val, dict):
				#		writer.add_scalar('val_' + key, val, epoch)

		except KeyboardInterrupt:
			logger.info('Interrupted')

		logger.info('Best Test set loss: {:.4f} (epoch {})'.format(best_val_loss, best_val_loss_epoch))

		#writer.add_hparams(
		#	{'zdim': zdim, 'h1_dim': h1, 'h2_dim': h2, 'bs': bs, 'lr': lr, 'mmd': mmd_weight, 'alpha_12': alpha_12,
		#	 'alpha_numcat': alpha_numcat},
		#	{f'hparam/val_{key}': best_val_loss_dict[key] for key in
		#	 ['rec_loss', 'loss', 'mse1', 'mse2_num', 'bce2_cat', 'mmd']})

		vae.load_state_dict(best_model)

		logger.info('Loaded best model')

		logger.info('Saving')

		#torch.save(vae.serialize(), output_dir / 'model.pth')
		#torch.save(optimizer.state_dict(), output_dir / 'optimizer.pth')

		full_data = cdm.get_full_data()
		embeds = vae.embed(full_data.values)
		embeds = pd.DataFrame(embeds, index=full_data.index, columns=[f'D{i}' for i in range(embeds.shape[1])])
		embeds.to_csv(output_dir / f'embeds/embeds_{i}.tsv', sep='\t')
		splits_df.to_csv(output_dir / f'splits/split_{i}.tsv', sep='\t')

		#writer.flush()

		logger.info('Complete')


if __name__ == '__main__':
	main()
import click

@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('model_dir', type=click.Path(exists=True))
@click.option('-o','--out_dir', type=str, default=None)
def main(data_dir, model_dir, out_dir):
	from models.coupled_vae import CoupledVAE
	import pandas as pd

	from pathlib import Path

	import torch

	from utils import loader_funcs
	from utils.coupled_dataset_module import CoupledDatasetModule

	print('Loading data')

	model_dir = Path(model_dir)

	model_state = torch.load(model_dir / 'model.pth')

	splits_df = loader_funcs.load_splits(model_dir / "splits_df.tsv")

	vae = CoupledVAE.deserialize(model_state)

	cdm = CoupledDatasetModule(data_dir, bs=-1)
	cdm.setup()
	cdm.split(splits_df=splits_df, scaler='minmax')

	print('Calculating embedding')

	full_data = cdm.get_full_data()
	embeds = vae.embed(full_data.values)

	z = vae.embed(full_data.values)
	embeds = pd.DataFrame(embeds, index=full_data.index, columns=[f'D{i}' for i in range(embeds.shape[1])])

	print('- Saving')

	if out_dir is None:
		out_dir = model_dir
	else:
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

	embeds.to_csv(out_dir / 'embeddings.tsv', sep='\t')


if __name__ == '__main__':
    main()
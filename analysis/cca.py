import click

@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path())
def main(model_dir, out_dir):
	import numpy as np
	import pandas as pd

	from pathlib import Path

	from utils import loader_funcs
	from utils.coupled_dataset_module import CoupledDatasetModule

	from sklearn.cross_decomposition import CCA

	np.random.seed(0)

	print(f'- Import data')

	model_dir = Path(model_dir)
	embeds_path = model_dir / 'embeddings.tsv'
	data_dir = Path('data/coupled')
	out_cca_dir = Path(out_dir)

	splits_df = loader_funcs.load_splits(model_dir / "splits_df.tsv")

	cdm = CoupledDatasetModule(data_dir, bs=-1)
	cdm.setup()
	cdm.split(splits_df=splits_df, scaler='minmax')

	expr, pheno = cdm.get_full_data(cond=False, concat=False)

	embeds = pd.read_table(embeds_path, index_col=0)

	print(f'- Evaluating embeddings')

	pheno_c, expr_c = CCA(n_components=embeds.shape[1]).fit_transform(pheno.values, expr.values)

	print(f'- Saving')

	out_cca_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(pheno_c,index=embeds.index, columns=[f'CC{i}' for i in range(pheno_c.shape[1])]).to_csv(out_cca_dir / 'pheno_cca.tsv',sep='\t')
	pd.DataFrame(expr_c,index=embeds.index, columns=[f'CC{i}' for i in range(expr_c.shape[1])]).to_csv(out_cca_dir / 'expr_cca.tsv',sep='\t')

if __name__ == '__main__':
	main()

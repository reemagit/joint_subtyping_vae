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
	from utils import get_pca

	np.random.seed(0)

	print(f'- Import data')

	model_dir = Path(model_dir)
	embeds_path = model_dir / 'embeddings.tsv'
	data_dir = Path('data/coupled')
	out_pca_dir = Path(out_dir)

	splits_df = loader_funcs.load_splits(model_dir / "splits_df.tsv")

	cdm = CoupledDatasetModule(data_dir, bs=-1)
	cdm.setup()
	cdm.split(splits_df=splits_df, scaler='minmax')

	expr_pheno = cdm.get_full_data(cond=False, concat=True)

	expr, pheno = cdm.get_full_data(cond=False, concat=False)

	embeds = pd.read_table(embeds_path, index_col=0)

	print(f'- Evaluating embeddings')

	expr_pca = get_pca(expr, n_components=embeds.shape[1], return_obj=False)
	pheno_pca = get_pca(pheno, n_components=embeds.shape[1], return_obj=False)
	expr_pheno_pca = get_pca(expr_pheno, n_components=embeds.shape[1], return_obj=False)

	print(f'- Saving')

	out_pca_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(expr_pca.values,index=embeds.index, columns=[f'PC{i}' for i in range(expr_pca.shape[1])]).to_csv(out_pca_dir / 'expr_pca.tsv',sep='\t')
	pd.DataFrame(pheno_pca.values,index=embeds.index, columns=[f'PC{i}' for i in range(pheno_pca.shape[1])]).to_csv(out_pca_dir / 'pheno_pca.tsv',sep='\t')
	pd.DataFrame(expr_pheno_pca.values,index=embeds.index, columns=[f'PC{i}' for i in range(expr_pheno_pca.shape[1])]).to_csv(out_pca_dir / 'expr_pheno_pca.tsv',sep='\t')

if __name__ == '__main__':
	main()

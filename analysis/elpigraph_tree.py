import click


def get_elpi_projs(embeds, elpi_graph, bound_start=False, bound_end=False):
	import pandas as pd
	from clintrajan import clintraj_eltree

	from utils import trajfuncs

	lbl = clintraj_eltree.partition_data_by_tree_branches(embeds.values, elpi_graph)

	branch_nodes = clintraj_eltree.gbrancher.find_branches(clintraj_eltree.convert_elpigraph_to_igraph(elpi_graph))[
		'branches']

	all_projs = pd.DataFrame([], columns=['branch', 'branch_pos', 'branch_dist'], index=embeds.index)
	for i in range(len(branch_nodes)):
		projs = trajfuncs.get_projs(embeds.loc[lbl == i], elpi_graph['NodePositions'][branch_nodes[i], :], bound_start=bound_start,
									bound_end=bound_end)
		all_projs.loc[projs.index, ['branch_pos', 'branch_dist']] = projs[['traj_pos', 'traj_dist']].values
		all_projs.loc[projs.index, 'branch'] = i

	return all_projs

@click.command()
@click.argument('input_embeds', type=str)
@click.argument('out_dir', type=str)
@click.option('-p','--pert', 'pert_embeds_dir', default=None, type=str, help='Directory of perturbed embeddings')
@click.option('-s','--seed', 'seed', default=42, type=int)
def main(input_embeds, out_dir,pert_embeds_dir,seed):
	import os
	import pickle
	from pathlib import Path

	import numpy as np
	import pandas as pd

	from clintrajan import clintraj_eltree
	from scipy.stats import percentileofscore
	from tqdm import tqdm, trange

	from utils import setup_logging, trajfuncs
	from utils.loader_funcs import load_embeddings

	np.random.seed(seed)

	params = locals()

	nReps = 10
	n_nodes = 30
	collapse = True

	logger = setup_logging(out_dir, __name__)

	for param, val in params.items():
		logger.info(param + ': ' + str(val))

	logger.info('- Loading data')
	embeds = load_embeddings(input_embeds)

	logger.info('- Processing')

	graph = trajfuncs.get_tree(embeds, n_nodes=n_nodes, collapse=collapse, nReps=nReps)

	projs = get_elpi_projs(embeds,elpi_graph=graph)

	projs['top_50'] = False
	for i in sorted(projs.branch.unique()):
		perc_val = np.percentile(projs.loc[projs.branch==i].branch_pos.values,50)
		projs.loc[(projs.branch==i) & (projs.branch_pos>perc_val), 'top_50'] = True
		vals = projs.loc[projs.branch==i].branch_pos.values
		projs.loc[projs.branch==i, 'branch_perc'] = [percentileofscore(vals,val,kind='strict') for val in vals] # workaround since scipy is older version

	tree_data = clintraj_eltree.return_eltree_coords(graph, embeds, projs.branch, scatter_parameter=0.05)

	projs['x'] = tree_data['x_data']
	projs['y'] = tree_data['y_data']
	
	logger.info('- Saving')

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)


	with open(out_dir / 'elpi_tree.pickle', 'wb') as fp:
		pickle.dump(graph, fp)

	projs.to_csv(out_dir / 'branches.tsv', sep='\t')

	if pert_embeds_dir is not None:

		pert_embeds_dir = Path(pert_embeds_dir)

		logger.info('- Loading perturbed data')

		pert_embeds = []
		pert_splits = []
		for i in trange(100):
			if os.path.exists(pert_embeds_dir / f'embeds/embeds_{i}.tsv'):
				pert_embeds.append(pd.read_table(pert_embeds_dir / f'embeds/embeds_{i}.tsv', index_col=0))
				pert_splits.append(pd.read_table(pert_embeds_dir / f'splits/split_{i}.tsv', index_col=0)['0'])

		logger.info('- Processing')

		pert_graphs = [trajfuncs.get_tree(elem, n_nodes=n_nodes, collapse=collapse, nReps=nReps) for elem in tqdm(pert_embeds)]

		pert_projs = pd.DataFrame([],index=embeds.index, columns=[f'pert_{i}' for i in range(len(pert_graphs))])
		for i in range(len(pert_graphs)):
			lbl = clintraj_eltree.partition_data_by_tree_branches(pert_embeds[i].values, pert_graphs[i])
			pert_projs.iloc[:,i] = lbl

		logger.info('- Saving')

		with open(out_dir / 'pert_elpi_trees.pickle', 'wb') as fp:
			pickle.dump(pert_graphs, fp)

		pert_projs.to_csv(out_dir / 'pert_branches.tsv',sep='\t')


if __name__ == '__main__':

    main()


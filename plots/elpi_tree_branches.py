import click

def plot(plotdata, feature, plot_branch_labels=True):
	import matplotlib.pyplot as plt
	import matplotlib.patheffects as PathEffects

	#cmap = plt.cm.tab10

	plt.scatter(plotdata['x_data'], plotdata['y_data'], c=feature,s=5,cmap='tab10')
	plt.scatter(plotdata['x_nodes'], plotdata['y_nodes'], c='k', s=30)
	for i in range(len(plotdata['edges'])):
		plt.plot(*plotdata['edges'][i],'-',color='black')

	if plot_branch_labels:
		branch_labels = []
		for i in range(len(plotdata['branches_lbl'])):
			branch_labels.append(plt.text(*plotdata['branches_lbl'][i], color='black',fontsize=15))
			branch_labels[-1].set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
	plt.xticks([])
	plt.yticks([])
	plt.box(False)

@click.command()
@click.argument('embeds_path', type=click.Path(exists=True))
@click.argument('graph_dir', type=click.Path(exists=True))
@click.option('-o','--out_file', type=str, default=None)
@click.option('--branch_labels/--no-branch_labels', default=True)
@click.option('--show/--no-show', default=True)
def main(embeds_path, graph_dir, out_file, branch_labels, show):
	import pandas as pd
	import numpy as np
	from pathlib import Path
	import pickle
	import sys
	from clintrajan import clintraj_eltree

	import matplotlib.pyplot as plt
	from os import system

	print('- Import')

	embeds_path = Path(embeds_path)
	graph_dir = Path(graph_dir)
	#data_dir = Path(data_dir)

	# Embeds
	embeds = pd.read_table(embeds_path, index_col=0)

	# Graph
	graph = pickle.load(open(graph_dir / 'elpi_tree.pickle', "rb"))

	# Labels
	lbl = pd.read_table(graph_dir / 'branches.tsv',index_col=0).branch.values

	print('- Plotting')

	plotdata = clintraj_eltree.return_eltree_coords(graph, embeds, lbl, scatter_parameter=0.05)

	plot(plotdata,lbl, plot_branch_labels=branch_labels)

	print('- Saving')

	if out_file is None:
		out_file = graph_dir / 'elpi_tree_branches.pdf'
	out_file = Path(out_file)
	plt.savefig(out_file)

	if show:
		system(f"open {out_file}")


if __name__ == '__main__':
	main()

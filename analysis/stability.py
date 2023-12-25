import click
import numpy as np
from sklearn.cluster import KMeans

# Functions

def most_freq_val(lbls):
	values, counts = np.unique(lbls, return_counts=True)
	ind = np.argmax(counts)
	return values[ind]

def purity(lbls):
	most_freq = most_freq_val(lbls)
	return (lbls==most_freq).sum()/len(lbls)

def coparticipation(lbl1,lbl2):
	freq1 = most_freq_val(lbl1)
	freq2 = most_freq_val(lbl2)
	return ((lbl1 == freq2).sum()/len(lbl1) + (lbl2 == freq1).sum()/len(lbl2))/2

def point_purity(i_point, lbl, pert_lbls):
	i_group = lbl[i_point]
	return sum([pert_lbls[j][i_point] == most_freq_val(pert_lbls[j][lbl==i_group]) for j in range(len(pert_lbls))]) / len(pert_lbls)

def get_extremal_points(projs, branch,perc=50):
	perc_val = np.percentile(projs[projs.branch==branch].branch_pos.values,perc)
	return ((projs.branch_pos >= perc_val) & (projs.branch == branch)).values

def get_clustering(embeds,n_cluster):
	
	kmeans = KMeans(init="random", n_clusters=n_cluster, n_init=10, max_iter=300, random_state=42 )
	kmeans.fit(embeds.values)
	return kmeans.labels_

def dist_from_centroid(vals):
	return np.sqrt(((vals-vals.mean(axis=0,keepdims=True))**2).sum(axis=1))

def get_clustering_core(embeds,lbls,perc=50,sel_lbls=None):
	if sel_lbls is None:
		sel_lbls = sorted(list(set(lbls)))
	lbls = lbls.copy()
	for lbl in sel_lbls:
		print((lbls==lbl).sum())
		embeds_loc = embeds.loc[lbls==lbl]
		dist_vals = dist_from_centroid(embeds_loc.values)
		perc_val = np.percentile(dist_vals,perc)
		lbls[embeds.index.isin(embeds_loc.loc[dist_vals>perc_val].index)] = -1
		print((lbls==lbl).sum())
	return lbls

def jsonify(obj): # quick and dirty
	if isinstance(obj, np.ndarray):
		return list(obj)
	else:
		return [list(elem) for elem in obj]


@click.command()
def main():
	import pandas as pd
	from tqdm.auto import trange
	from pathlib import Path
	import json
	import os

	branches_dir = Path('gendata_1/elpi')
	embeds_dir = Path('gendata_1/model')
	pert_dir = Path('gendata_1/stability')
	out_dir = Path('gendata_1/stability')

	print('- Load data')

	projs = pd.read_table(branches_dir / 'branches.tsv',index_col=0)
	pert_branches = pd.read_table(branches_dir / 'pert_branches.tsv',index_col=0)
	embeds = pd.read_table(embeds_dir / 'embeddings.tsv',index_col=0)
	pert_embeds = []
	for i in trange(100):
		if os.path.exists(pert_dir / f'embeds/embeds_{i}.tsv'):
			pert_embeds.append(pd.read_table(pert_dir / f'embeds/embeds_{i}.tsv', index_col=0))

		
	print('- Processing')

	lbl = projs.branch.values
	pert_lbls = pert_branches.values.T

	# Branches

	groups = sorted(np.unique(lbl))
	group_purities = []
	group_purities_extrema = []
	for j in range(len(groups)):
		group = groups[j]
		group_purities.append([purity(pert_lbls[i,:][lbl==group]) for i in range(pert_lbls.shape[0])])
		group_purities_extrema.append([purity(pert_lbls[i,:][get_extremal_points(projs, j, perc=50)]) for i in range(pert_lbls.shape[0])])

	# K-means clusters

	lbl_kmeans = get_clustering(embeds, n_cluster=len(groups))
	pert_lbl_kmeans = np.asarray([get_clustering(pert_embeds[i],n_cluster=len(set(pert_lbls[i]))) for i in trange(len(pert_embeds))])

	lbl_kmeans_core = get_clustering_core(embeds, lbl_kmeans)
	
	groups_kmeans = sorted(np.unique(lbl_kmeans))
	group_purities_kmeans = []
	group_purities_kmeans_core = []
	for j in range(len(groups_kmeans)):
		group = groups_kmeans[j]
		group_purities_kmeans.append([purity(pert_lbl_kmeans[i,:][lbl_kmeans==group]) for i in range(pert_lbl_kmeans.shape[0])])
		group_purities_kmeans_core.append([purity(pert_lbl_kmeans[i,:][lbl_kmeans_core==group]) for i in range(pert_lbl_kmeans.shape[0])])

	## Point purities

	point_purities = np.array([point_purity(i, lbl, pert_lbls) for i in trange(len(lbl))])

	results = {
		'purities':jsonify(group_purities),
		'purities_top50':jsonify(group_purities_extrema),
		'purities_kmeans':jsonify(group_purities_kmeans),
		'purities_kmeans_top50':jsonify(group_purities_kmeans_core),
		'point_purities':jsonify(point_purities),
		'readme': f'Generated with python script {os.path.abspath(__file__)}'
	}

	out_dir.mkdir(parents=True, exist_ok=True)

	json.dump(results, open(out_dir / 'purity_results.json','w'))

if __name__ == '__main__':

	main()


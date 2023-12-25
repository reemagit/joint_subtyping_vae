import numpy as np
import pandas as pd
import click
import logging
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
from tqdm import trange, tqdm
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import rankdata

from scipy import stats

from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import fdrcorrection

expr_dir = 'data/expr/processed/'
pheno_dir = 'data/pheno/raw/'


def get_redundant_features(sim_matr, feat_names, threshold=0.99):
    sim_matr = sim_matr.copy()
    np.fill_diagonal(sim_matr,0)
    feat_graph = nx.from_edgelist([(feat_names[i],feat_names[j]) for i,j in zip(*np.where(sim_matr>=threshold))])
    feat_to_keep = [pd.Series(nx.closeness_centrality(nx.subgraph(feat_graph, cc))).sort_values(ascending=False).index.tolist()[0] for cc in nx.connected_components(feat_graph)]
    feat_to_remove = [feat for feat in list(feat_graph.nodes()) if feat not in feat_to_keep]
    return feat_to_remove

def get_pvals_num(j, expr, pheno, sel_cols):
    res = []
    for i in trange(expr.shape[1],leave=False):
        res.append(stats.spearmanr(expr.iloc[:,i].values,pheno[sel_cols[j]].values).pvalue)
    return pd.Series(res,name=sel_cols[j],index=expr.columns)

def get_pvals_cat(j, expr, pheno, sel_cols):
    res = []
    for i in tqdm(expr.columns,leave=False):
        vals = [expr.loc[pheno[sel_cols[j]] == k,i].values for k in pheno[sel_cols[j]].unique()]
        res.append(stats.kruskal(*vals).pvalue)
    return pd.Series(res,name=sel_cols[j],index=expr.columns)



def main():


	expr_adj = pd.read_table(expr_dir + 'deseq_counts_lc_age_sex_race.tsv',index_col=0)

	pheno = pd.read_table(pheno_dir + 'P1P2_Pheno_Flat_All_sids_Mar19.txt', index_col=0)
	pheno_vars = pd.read_table(pheno_dir + 'P1P2_Pheno_Flattened_Mar19_DataDict_categorized2.csv', index_col=0, sep=',')

	masterfile = pd.read_table(expr_dir + 'masterfile.tsv', index_col=0)
      
	pheno_vars.loc[pheno_vars.Category == 'QCT-Slicer', 'Category'] = 'QCT'
	pheno_vars.loc[pheno_vars.Category == 'QCT-Thirona', 'Category'] = 'QCT'

	for feat in ['Asthma', 'Emphysema', 'HayFev', 'Pneumonia', 'SleepApnea']:
		pheno[f'{feat}_P2'] = ((pheno[f'{feat}_P1'] == 1) | (pheno[f'{feat}_slv_P2'] == 1)).astype(int)
		pheno = pheno.drop(columns=[f'{feat}_P1', f'{feat}_slv_P2'])
		pheno_vars = pheno_vars.append(pd.Series(
			{'Label': f'{feat}', 'Category': pheno_vars.loc[f'{feat}_P1', 'Category'],
			 'feat_type': pheno_vars.loc[f'{feat}_P1', 'feat_type']}, name=f'{feat}_P2'))
		pheno_vars = pheno_vars.drop(index=[f'{feat}_P1', f'{feat}_slv_P2'])

	pheno_vars = (
		pheno_vars
			.pipe(lambda df: df[~df.index.str.endswith('P1')])
			.pipe(lambda df: df[df.Category.notna()])
			.pipe(lambda df: df[df.Category.apply(lambda x: 'Change' not in x.split('|'))])
			.drop(index=['ChronBronch_slv_P2'])
	)

	pheno = pheno.rename(columns={'ShrtBrthAttk': 'ShrtBrthAttk_P2'})
	pheno_vars = pheno_vars.rename(index={'ShrtBrthAttk': 'ShrtBrthAttk_P2'})

	pheno['O2_Therapy_P2'] = pheno.O2_Therapy_P2.apply(lambda x: np.nan if x != x or x == '.' else float(x))

	categories_to_keep = ['MedicalHx', 'QCT-Thirona', 'RespDisease', 'MedicationHx', 'MedicationHx|RespDisease', \
						  'LungFunc', 'CBC', 'Demos', 'LungFunc', 'LungFunc|DLCO', 'Symptoms', 'QCT-Slicer', 'SmokeHx',
						  '6minWalk', 'Smoke', 'DrugResponse', 'QCT']

	all_sids = [sid for sid in masterfile.index.tolist() if sid in pheno.index.tolist()]

	pheno = pheno.loc[all_sids]

	sel_cat_cols = pheno_vars[
		pheno_vars.apply(lambda x:
						 x.feat_type == 'categorical' and
						 x.Category in categories_to_keep and
						 pheno[x.name].count() / len(pheno) > 0.9 and
						 pheno[x.name].value_counts().values[0] / len(pheno) < 0.99
						 , axis=1)
	].index.tolist()

	imputer = SimpleImputer(strategy='most_frequent')
	pheno_proc1 = pheno.copy()
	pheno_proc1[sel_cat_cols] = imputer.fit_transform(pheno_proc1[sel_cat_cols].values)

	cat_sims = np.zeros([len(sel_cat_cols), len(sel_cat_cols)])
	for i in trange(len(sel_cat_cols)):
		for j in range(i + 1, len(sel_cat_cols)):
			cat_sims[i, j] = adjusted_rand_score(pheno_proc1[sel_cat_cols[i]].values,
												 pheno_proc1[sel_cat_cols[j]].values)

	redundant_vars = get_redundant_features(cat_sims, sel_cat_cols, 1)

	sel_cat_cols = [col for col in sel_cat_cols if col not in redundant_vars]


	sel_num_cols = pheno_vars[
		pheno_vars.apply(lambda x:
						 x.name not in ['Visit_Date_P1', 'Visit_Date_P2'] and
						 x.Category in categories_to_keep and
						 x.feat_type == 'numerical' and
						 pheno_proc1[x.name].count() / len(pheno_proc1) > 0.9 and
						 pheno_proc1[x.name].std() > 0
						 , axis=1)
	].sort_values('Form').index.tolist()

	imputer_num = KNNImputer(n_neighbors=10, weights="uniform")
	pheno_proc2 = pheno_proc1.copy()
	pheno_proc2[sel_num_cols] = imputer_num.fit_transform(pheno_proc1[sel_num_cols])

	#num_corr = pheno_proc2[sel_num_cols].corr().values
	#redundant_vars = get_redundant_features(num_corr, sel_num_cols, threshold=0.95)

	# These variables have been determined redundant by the get_redundant_features function, but had the same
	# centrality value of their cognate variables, and therefore the choice is arbitrary. To ensure reproducibility
	# they are hardcoded
	redundant_vars = ['hemoglobin_P2',
					  'FEV1_utah_P2',
					  'FEV1_FVC_utah_P2',
					  'FEV6_utah_P2',
					  'FVC_utah_P2',
					  'pre_FEV6_P2',
					  'MeanAtten_Insp_Thirona_P2',
					  'Perc15_Insp_RLL_Thirona_P2',
					  'Perc15_Insp_Thirona_P2',
					  'Perc15_Insp_RUL_Thirona_P2',
					  'Perc15_Insp_LUL_Thirona_P2',
					  'adj_density_plethy_P2',
					  'Insp_LAA950_LLL_Thirona_P2',
					  'Insp_LAA950_RLL_Thirona_P2',
					  'pctEmph_Thirona_P2',
					  'Insp_LAA950_LUL_Thirona_P2',
					  'Insp_LAA950_RUL_Thirona_P2',
					  'HAA700_Insp_Slicer_P2',
					  'HAA500_Insp_Slicer_P2']

	# Lung function variables predicted at population level (do not relate to subject)
	predicted_lung = pheno_vars[
		(pheno_vars.Category == 'LungFunc') & (pheno_vars.index.str.lower().str.contains('pred'))].index.tolist()

	sel_num_cols = [col for col in sel_num_cols if col not in redundant_vars and col not in predicted_lung]

	pheno_adj = pheno_proc2.loc[expr_adj.index].copy()

	pheno_adj = pheno_adj[sel_num_cols + sel_cat_cols]

	pvals_num = [get_pvals_num(j, expr_adj, pheno_adj, sel_num_cols) for j in trange(len(sel_num_cols))]
	pvals_cat = [get_pvals_cat(j, expr_adj, pheno_adj, sel_cat_cols) for j in trange(len(sel_cat_cols))]
	pvals = pvals_num + pvals_cat
	pvals_df = pd.concat(pvals, axis=1)
      
	pvals_categories_to_keep = ['RespDisease', 'LungFunc', 'LungFunc|DLCO', 'Symptoms', 'DrugResponse']

	pvals_matr = pvals_df

	pvals_matr = pvals_matr[[col for col in sel_num_cols + sel_cat_cols if pheno_vars.loc[col, 'Category'] in pvals_categories_to_keep and col in pvals_matr.columns]].copy()
	for gene in tqdm(pvals_matr.index):
		pvals_matr.loc[gene] = fdrcorrection(pvals_matr.loc[gene].values)[1]

	sig_genes = ((pvals_matr < 1e-4).sum(axis=1) > 0)
	sig_genes = sig_genes[sig_genes].index.tolist()

	# write sig_genes list to disk
	with open(expr_dir + 'sig_genes.txt','w') as f:
		f.write('\n'.join(sig_genes))


if __name__ == '__main__':
    main()
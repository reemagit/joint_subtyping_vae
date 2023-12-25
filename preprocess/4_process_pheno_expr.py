import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_rand_score
import networkx as nx
from tqdm import trange, tqdm
from statsmodels.stats.multitest import fdrcorrection

in_expr_dir = 'data/expr/processed/'
in_pheno_dir = 'data/pheno/raw/'

out_dir = 'data/coupled/'

def dummify(pheno, pheno_vars):
	pheno_vars = pheno_vars.copy()
	cols = pheno.columns.tolist()
	pheno = pd.get_dummies(pheno, prefix_sep='___')
	for col in cols:
		row = pheno_vars.loc[col]
		if row.feat_type == 'numerical':
			pheno_vars.at[col, ['i_start', 'i_end']] = pheno.columns.tolist().index(col)
		else:
			idxs = np.where(pheno.columns.str.startswith(col + '___'))[0]
			assert sorted(idxs) == list(range(min(idxs), max(idxs) + 1))
			pheno_vars.at[col, ['i_start', 'i_end']] = (min(idxs), max(idxs))

	pheno_vars = (
		pheno_vars
			.dropna(subset=['i_start', 'i_end'])
			.assign(i_start=lambda x: x.i_start.astype(int), i_end=lambda x: x.i_end.astype(int))
	)
	return pheno, pheno_vars

def get_redundant_features(sim_matr, feat_names, threshold=0.99):
    sim_matr = sim_matr.copy()
    np.fill_diagonal(sim_matr,0)
    feat_graph = nx.from_edgelist([(feat_names[i],feat_names[j]) for i,j in zip(*np.where(sim_matr>=threshold))])
    feat_to_keep = [pd.Series(nx.closeness_centrality(nx.subgraph(feat_graph, cc))).sort_values(ascending=False).index.tolist()[0] for cc in nx.connected_components(feat_graph)]
    feat_to_remove = [feat for feat in list(feat_graph.nodes()) if feat not in feat_to_keep]
    return feat_to_remove



def main():
	expr = pd.read_table(in_expr_dir + 'deseq_counts_lc.tsv', index_col=0).astype(
		np.float32)
	masterfile = pd.read_table(in_expr_dir + 'masterfile.tsv', index_col=0)

	pheno = pd.read_table(in_pheno_dir + 'P1P2_Pheno_Flat_All_sids_Mar19.txt', index_col=0)
	pheno_vars = pd.read_table(in_pheno_dir + 'P1P2_Pheno_Flattened_Mar19_DataDict_categorized2.csv', index_col=0, sep=',')

	# Here some data munging

	pheno_vars.loc[pheno_vars.Category == 'QCT-Slicer', 'Category'] = 'QCT'
	pheno_vars.loc[pheno_vars.Category == 'QCT-Thirona', 'Category'] = 'QCT'

	# These features have only information in Phase 1, and "since last visit" in Phase 2, integrate to create a Phase 2 feature
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

	sids_no_cbc = pheno[pheno_vars[pheno_vars.Category=='CBC'].index].isna().any(axis=1).pipe(lambda x: x[x].index)

	all_sids = [sid for sid in expr.index.tolist() if sid in pheno.index.tolist() and sid not in sids_no_cbc]


	expr = expr.loc[all_sids]
	pheno = pheno.loc[all_sids]
	masterfile = masterfile.loc[all_sids]

	categories_to_keep = ['MedicalHx', 'QCT', 'RespDisease', 'MedicationHx', 'MedicationHx|RespDisease',
						  'LungFunc', 'CBC', 'Demos', 'LungFunc', 'LungFunc|DLCO', 'Symptoms', 'SmokeHx',
						  'Smoke', 'DrugResponse']

	sel_cat_cols = pheno_vars[
		pheno_vars.apply(lambda x:
						 x.feat_type == 'categorical' and
						 x.Category in categories_to_keep and
						 pheno[x.name].count() / len(pheno) > 0.9 and
						 pheno[x.name].value_counts().values[0] / len(pheno) < 0.8
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


	predicted_lung = pheno_vars[(pheno_vars.Category == 'LungFunc') & (pheno_vars.index.str.lower().str.contains('pred'))].index.tolist()

	sel_num_cols = [col for col in sel_num_cols if col not in redundant_vars and col not in predicted_lung]

	# Conditional variables

	pheno_cond_vars = ['gender','race','Age_P2']
	expr_cond_vars = ['gender','race','Age_P2']

	# Prepare tables

	pheno_proc3 = pheno_proc2.copy()
	pheno_proc3[sel_cat_cols] = pheno_proc3[sel_cat_cols].astype('category')
	pheno_proc3 = pheno_proc3[sel_num_cols + sel_cat_cols].copy()
	pheno_cond = pheno_proc3[pheno_cond_vars].copy()

	expr_cond = pheno_proc3[expr_cond_vars].copy()


	pheno_proc3 = pheno_proc3.drop(pheno_cond_vars, axis=1)

	pheno_proc4, pheno_vars4 = dummify(pheno_proc3, pheno_vars)

	pheno_cond2 = pd.get_dummies(pheno_cond, prefix_sep='___')
	pheno_cond2 = (pheno_cond2 - pheno_cond2.min(axis=0)) / (pheno_cond2.max(axis=0) - pheno_cond2.min(axis=0))


	expr_cond2 = pd.get_dummies(expr_cond, prefix_sep='___')
	expr_cond2 = (expr_cond2 - expr_cond2.min(axis=0)) / (expr_cond2.max(axis=0) - expr_cond2.min(axis=0))


	# TODO REMOVE COMMENTS
	# Processing expression

	# Only keep genes that are loosely associated to at least one feature of the categories below
	#pvals_categories_to_keep = ['QCT-Thirona', 'RespDisease', \
	#							'LungFunc', 'LungFunc|DLCO', 'Symptoms', 'QCT-Slicer',
	#							'DrugResponse','QCT']

	
	# Matrix is pre-calculated
	#pvals_matr = pd.read_table(in_expr_dir + 'assoc_pvals.tsv',index_col=0)

	#pvals_matr = pvals_matr[[col for col in sel_num_cols + sel_cat_cols if pheno_vars.loc[col, 'Category'] in pvals_categories_to_keep and col in pvals_matr.columns]].copy()
	#for gene in tqdm(pvals_matr.index):
	#	pvals_matr.loc[gene] = fdrcorrection(pvals_matr.loc[gene].values)[1]


	#sig_genes = ((pvals_matr < 1e-4).sum(axis=1) > 0)
	#sig_genes = sig_genes[sig_genes].index

	# import sig_genes from txt file as a list
	sig_genes = []
	with open(in_expr_dir + 'sig_genes.txt','r') as f:
		for line in f.readlines():
			sig_genes.append(line.strip())

	expr = expr[sig_genes]

	main_dir = Path(out_dir)
	main_dir.mkdir(parents=True, exist_ok=True)

	pheno_proc4.to_csv(main_dir / 'pheno.tsv', sep='\t', index_label='sid')
	pheno_vars4.to_csv(main_dir / 'pheno.vars.tsv', sep='\t')
	pheno_cond2.to_csv(main_dir / 'pheno.cond.tsv', sep='\t', index_label='sid')

	pheno.to_csv(main_dir / 'pheno.feats.tsv', sep='\t', index_label='sid')
	pheno_vars.to_csv(main_dir / 'pheno.feats.vars.tsv', sep='\t')

	# we also use the same conditional variables for expression
	expr_cond2.to_csv(main_dir / 'expr.cond.tsv', sep='\t', index_label='sid')
	expr.to_csv(main_dir / 'expr.tsv.gz', sep='\t', index_label='sid')

if __name__ == '__main__':

    main()
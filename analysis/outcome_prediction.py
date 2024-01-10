import click


def get_performance(all_X, y, clf=None):
	from tqdm.auto import trange, tqdm
	from imblearn.pipeline import Pipeline
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
	from sklearn.dummy import DummyClassifier
	from imblearn.over_sampling import SMOTE

	smt = SMOTE(random_state=1)
	pipe = Pipeline([('scaler', MinMaxScaler()),('smt', smt),('clf', clf(n_estimators=100, random_state=0))])

	#pipe = Pipeline([('scaler', MinMaxScaler()),
	#				 ('clf', clf(n_estimators=100, max_depth=5, random_state=0, max_features=3))])
	res = {}
	for eid in tqdm(all_X):
		X = all_X[eid]
		res[eid] = cross_validate(pipe, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1),
								  return_estimator=True,
								  scoring=['f1', 'average_precision', 'balanced_accuracy'])
	return res

def report(y,name):
	print(f'Embedding: {name}; Num samples: {len(y)}; sparsity: {y.sum()/y.shape[0]}')


@click.command()
@click.argument('model_dir', type=click.Path(exists=True))
@click.argument('gendata_dir', type=click.Path(exists=True))
@click.option('-o', '--out_dir', 'out_dir', type=click.Path(), default=None,
			  help='Output directory of results file (.tsv)')
def main(model_dir, gendata_dir, out_dir):
	import numpy as np
	import pandas as pd

	from sklearn.ensemble import RandomForestClassifier

	from pathlib import Path

	from scipy.stats import ttest_rel

	np.random.seed(0)

	print(f'- Import data')

	model_dir = Path(model_dir)
	gendata_dir = Path(gendata_dir)
	embeds_path = model_dir / 'embeddings.tsv'
	pca_dir = Path(gendata_dir / 'pca')
	cca_dir = Path(gendata_dir / 'cca')
	mofa_dir = Path(gendata_dir / 'mofa')
	outcome_dir = Path(gendata_dir / 'data')

	embeds = pd.read_table(embeds_path, index_col=0)

	pheno_p3 = pd.read_table(outcome_dir / 'pheno.tsv',index_col=0)
	pheno_p3 = pheno_p3.loc[embeds.index]

	mortality = pheno_p3[['P3_vitalstatus','vital_status_3yr']]
	#mortality = mortality.loc[embeds.index]

	print(f'- Evaluating embeddings')

	# PCA precalcualted with pca.py
	expr_pca = pd.read_table(pca_dir / 'expr_pca.tsv', index_col=0)
	pheno_pca = pd.read_table(pca_dir / 'pheno_pca.tsv', index_col=0)
	expr_pheno_pca = pd.read_table(pca_dir / 'expr_pheno_pca.tsv', index_col=0)

	# CCA precalculated with cca.py
	pheno_c = pd.read_table(cca_dir / 'pheno_cca.tsv',index_col=0)
	expr_c = pd.read_table(cca_dir / 'expr_cca.tsv',index_col=0)

	# MOFA factors have been pre-calculated with mofa.py
	mofa = pd.read_table(mofa_dir / 'Z_fact.tsv',index_col=0)

	all_embeds = {'vae': embeds,
				  'expr_pca': expr_pca,
				  'pheno_pca': pheno_pca,
				  'expr_pheno_pca': expr_pheno_pca,
				  'expr_cca': pheno_c,
				  'pheno_cca': expr_c,
				  'mofa':mofa
				  }

	print(f'- Evaluating performance')

	# Mortality P3

	idx = mortality.P3_vitalstatus.notna().values
	sids = embeds.loc[idx].index.tolist()

	y = mortality.P3_vitalstatus.loc[sids]
	y = y.values.astype(int)

	report(y,'mortality p3')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}
	res_mort = get_performance(all_X, y, RandomForestClassifier)

	res_mort['n_samples'] = len(sids)

	# Mortality 3yr

	idx = mortality.vital_status_3yr.notna().values
	sids = embeds.loc[idx].index.tolist()

	y = mortality.vital_status_3yr.loc[sids]
	y = y.values.astype(int)

	report(y,'mortality 3yr')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}
	res_mort_3yr = get_performance(all_X, y, RandomForestClassifier)

	res_mort_3yr['n_samples'] = len(sids)

	# Severe exacerbations

	idx = pheno_p3['Severe_Exacerbations_P3'].notna().values
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'Severe_Exacerbations_P3']
	y = y.values.astype(int)

	report(y,'sev exac')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_exac = get_performance(all_X, y, RandomForestClassifier)

	res_exac['n_samples'] = len(sids)

	# Occurrence of severe exacerbations

	idx = pheno_p3['Severe_Exacerbations_P3'].notna().values & pheno_p3['Severe_Exacerbations_P2'].notna().values & (pheno_p3['Severe_Exacerbations_P2'] == 0)
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'Severe_Exacerbations_P3']
	y = y.values.astype(int)

	report(y,'occ exac')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_occ_exac = get_performance(all_X, y, RandomForestClassifier)

	res_occ_exac['n_samples'] = len(sids)

	# Freq. of exacerbations

	idx = pheno_p3['Exacerbation_Frequency_P3'].notna().values
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'Exacerbation_Frequency_P3'] > 0
	y = y.values.astype(int)

	report(y,'freq exac')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_freq = get_performance(all_X, y, RandomForestClassifier)

	res_freq['n_samples'] = len(sids)

	# Increase in freq. of exacerbations

	idx = pheno_p3['Exacerbation_Frequency_P3'].notna().values & pheno_p3['Exacerbation_Frequency_P2'].notna().values
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'Exacerbation_Frequency_P3'] > pheno_p3.loc[sids, 'Exacerbation_Frequency_P2']
	y = y.values.astype(int)

	report(y,'incr freq exac')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_incr_freq = get_performance(all_X, y, RandomForestClassifier)

	res_incr_freq['n_samples'] = len(sids)

	# Decrease in FEV

	idx = pheno_p3['Change_P2_P3_FEV1pp'].notna().values & (pheno_p3['Change_P2_P3_FEV1pp'].abs() > 10)
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'Change_P2_P3_FEV1pp'] < 0
	y = y.values.astype(int)

	report(y,'decr fev')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_decr_fev = get_performance(all_X, y, RandomForestClassifier)

	res_decr_fev['n_samples'] = len(sids)

	# New chronic bronchitis

	idx = pheno_p3['New_Chronic_Bronchitis_P3'].isin([0, 1]).values & (pheno_p3['Chronic_Bronchitis_P2'] == 0).values
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'New_Chronic_Bronchitis_P3']
	y = y.values.astype(int)

	report(y,'new chron bronch')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_bronchitis = get_performance(all_X, y, RandomForestClassifier)

	res_bronchitis['n_samples'] = len(sids)

	# Increase in MMRC dyspnea score

	idx = pheno_p3['Change_P2_P3_MMRC'].notna().values & (pheno_p3['MMRCDyspneaScor_P3'] < 4)
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'Change_P2_P3_MMRC'] > 0
	y = y.values.astype(int)

	report(y,'decr mmrc')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_mmrc = get_performance(all_X, y, RandomForestClassifier)

	res_mmrc['n_samples'] = len(sids)

	# Decrease in SF-36 Physical compound score

	idx = pheno_p3['SF36_PCS_score_P3'].notna().values & pheno_p3['SF36_PCS_score_P2'].notna().values & pheno_p3['SF36_PCS_score_P2'].notna().values
	sids = embeds.loc[idx].index.tolist()

	y = pheno_p3.loc[sids, 'SF36_PCS_score_P3'] - pheno_p3.loc[sids, 'SF36_PCS_score_P2'] < 0
	y = y.values.astype(int)

	report(y,'decr sf36')

	all_X = {eid: emb.loc[sids].values for eid, emb in all_embeds.items()}

	res_sf36 = get_performance(all_X, y, RandomForestClassifier)

	res_sf36['n_samples'] = len(sids)
	

	# Table
	res_list = [res_mort, res_mort_3yr, res_exac, res_occ_exac, res_freq, res_incr_freq, res_decr_fev, res_bronchitis, res_mmrc, res_sf36, res_encr_emph, res_encr_gas]
	res_ids = ['mortality_P3', 'mortality_3yr', 'sev_exac', 'occ_sev_exac', 'freq_exac', 'freq_exac_increase','res_decr_fev', 'res_bronchitis', 'res_mmrc', 'res_sf36','res_encr_emph','res_encr_gas']

	metric_list = ['f1','average_precision','balanced_accuracy']
	embed_list = ['pheno_pca', 'expr_pca', 'expr_pheno_pca', 'expr_cca', 'pheno_cca', 'mofa','vae']

	df_list = []
	for j, metric in enumerate(metric_list):
		df_list.append(pd.DataFrame(
			[[np.mean(res[embed_list[i]][f'test_{metric}']) for i in range(len(embed_list))] for res in res_list],
			columns=embed_list, index=res_ids))
	df = pd.concat(df_list, axis=0, keys=metric_list)

	df_list = []
	for j, metric in enumerate(metric_list):
		df_list.append(pd.DataFrame(
			[[np.std(res[embed_list[i]][f'test_{metric}']) for i in range(len(embed_list))] for res in res_list],
			columns=embed_list, index=res_ids))
	df_std = pd.concat(df_list, axis=0, keys=metric_list)

	df_pval = pd.DataFrame([],columns=metric_list, index=res_ids)
	for j, metric in enumerate(metric_list):
		for k, res in enumerate(res_list):
			best_embeds_idx = np.argsort([np.mean(res[embed_list[i]][f'test_{metric}']) for i in range(len(embed_list))])[::-1]
			best_embeds = embed_list[best_embeds_idx[0]]
			second_best_embeds = embed_list[best_embeds_idx[1]]
			df_pval.loc[res_ids[k], metric] = ttest_rel(res[best_embeds][f'test_{metric}'],res[second_best_embeds][f'test_{metric}'],alternative='greater').pvalue

	df_num_samples = pd.DataFrame([res['n_samples'] for res in res_list], columns=['n_samples'], index=res_ids)

	if out_dir is None:
		out_dir = model_dir
	else:
		out_dir = Path(out_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

	avg_out_file = out_dir / 'classification_avg.tsv'.format(Path(__file__).stem)
	std_out_file = out_dir / 'classification_std.tsv'.format(Path(__file__).stem)
	pval_out_file = out_dir / 'classification_pval.tsv'.format(Path(__file__).stem)
	num_samples_out_file = out_dir / 'classification_num_samples.tsv'.format(Path(__file__).stem)

	print(f'- Saving to {out_dir}')

	df.to_csv(avg_out_file, sep='\t')
	df_std.to_csv(std_out_file, sep='\t')
	df_pval.to_csv(pval_out_file, sep='\t')
	df_num_samples.to_csv(num_samples_out_file, sep='\t')

if __name__ == '__main__':
	main()

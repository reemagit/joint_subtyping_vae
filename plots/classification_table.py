import click

def get_asterisks(pval):
	if pval < 1e-3:
		return '***'
	elif pval < 1e-2:
		return '**'
	elif pval < 0.05:
		return '*'
	else:
		return '[n.s.]'

def print_table(df, df_std, df_pval, lbl=None, metrics=None):
	if lbl is None:
		lbl = lambda x: x

	col_number = df.shape[1]

	col_string = ''.join(['c']*col_number)

	outcome_list = ['res_decr_fev','res_bronchitis','freq_exac','freq_exac_increase','sev_exac','occ_sev_exac','res_mmrc','res_sf36','mortality_3yr','mortality_P3']

	text = []

	text.append('\\begin{table}[ht]\n')
	text.append('\\centering\n')
	text.append('\\begin{adjustbox}{width=1.\\textwidth,center=\\textwidth}\n')

	text.append('\\begin{tabular}{r' + col_string + '} \\toprule\n')

	if metrics is None:
		metrics = df.index.get_level_values(0).unique()

	for metric in metrics:
		text.append('& \\multicolumn{' + str(df.shape[1]) + '}{c}{\\textbf{' + lbl(metric) + '}}\\\\\n')
		text.append('\\midrule\n')
		text.append(' & {' + '} & {'.join(map(lbl, df.columns.tolist())) + '}\\\\ \\midrule\n')
		for i, outcome in enumerate(outcome_list):
			row = df.loc[(metric,outcome)]
			std_row = df_std.loc[(metric, outcome)]
			pval = df_pval.loc[outcome, metric]

			textrow = lbl(outcome)
			for x, y in zip(row.values, std_row.values):
				ord_vals = sorted(row.values,reverse=True)
				if x == row.max():
					sig_str = get_asterisks(pval)
					textrow += ' & \\textbf{{{:.02f} ({:.02f})}} {}'.format(x, y,sig_str)


				elif x == ord_vals[1]:
					textrow += ' & \\underline{{{:.02f} ({:.02f})}}'.format(x, y)
				else:
					textrow += ' & {:.02f} ({:.02f})'.format(x, y)
			textrow += '\\\\\n'
			text.append(textrow)
		# text.append(i[1] + ' & ' + ' & '.join(map(lambda x: '{:.03f} ({:.03f})'.format(x[0],x[1]),zip(row.values,std_row.values))))
		text.append('\\midrule\n')
	text.append('\\bottomrule\n')
	text.append('\\end{tabular}')

	text.append('\\end{adjustbox}\n')
	text.append('\\end{table}\n')


	return [row.replace('_', '\_') for row in text]

def get_label(lbl):
    conv = {'pheno_pca':'Clin PCA','expr_pca':'Expr PCA','expr_pheno_pca':'Expr+Clin PCA','vae':'VAE',
			'pheno_cca':'Clin CCA','expr_cca':'Expr CCA','mofa':'MOFA',
           'copd_p3':'COPD','copd_increase':'Increased GOLD','mortality_P3':'Mortality (5yr)','mortality_3yr':'Mortality (3yr)',
			'sev_exac':'Sev. Exacerbations (P3)','occ_sev_exac':'$\\Delta$ Sev. Exacerbations (P3>P2)','freq_exac':'Exacerbations (P3)','freq_exac_increase':'$\\Delta$ Exac. Freq. (P3>P2) ',
            'accuracy':'Accuracy','roc_auc':'ROC AUC', 'f1':'F1-score','average_precision':'AUPRC','balanced_accuracy':'Balanced accuracy',
			'res_decr_fev':'$\\Delta$ FEV$_1$\\%pred.',
			'res_bronchitis':'Inc. bronchitis',
			'res_mmrc':'$\\Delta$ MMRC (P3>P2)','res_sf36':'$\\Delta$ SF-36 (P3<P2)',
			'res_encr_emph':'$\\Delta$ Emph.\\% (P3>P2)','res_encr_gas':'$\\Delta$ \\%GasTrap (P3>P2)'
           }
    if lbl in conv:
        return conv[lbl]
    return lbl

@click.command()
@click.argument('table_avg_path', type=click.Path(exists=True))
@click.argument('table_std_path', type=click.Path(exists=True))
@click.argument('table_pval_path', type=click.Path(exists=True))
@click.option('-o', '--out_path', 'out_path', type=click.Path(), default=None, help='Output path of table (.tex)')
def main(table_avg_path, table_std_path, table_pval_path, out_path):
	import pandas as pd

	from pathlib import Path

	table_avg_path = Path(table_avg_path)
	table_std_path = Path(table_std_path)

	df = pd.read_table(table_avg_path, index_col=[0, 1])
	df_std = pd.read_table(table_std_path, index_col=[0, 1])
	df_pval = pd.read_table(table_pval_path, index_col=0)

	if out_path is None:
		out_path = table_avg_path.parents[0] / 'table.latex'

	with open(out_path, 'w') as f:
		f.writelines(print_table(df, df_std, df_pval, get_label, ['f1','average_precision', 'balanced_accuracy']))

if __name__ == '__main__':
	main()

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import viz

branch_labels = {1:'SEV',3:'MOD',4:'SYMPT',5:'NORM1',6:'NORM2'}

def print_table(df, df_std, feats, lbl=None):
    if lbl is None:
        lbl = lambda x: x

    col_number = df.shape[1]

    col_string = ''.join(['c']*col_number)

    text = []
    
    text.append('\\begin{table}[ht]\n')
    text.append('\\centering\n')
    text.append('\\begin{adjustbox}{width=1.\\textwidth,center=\\textwidth}\n')
    
    text.append('\\begin{tabular}{r' + col_string + '} \\toprule\n')
    
    #text.append('& \\multicolumn{' + str(df.shape[1]) + '}{c}{\\textbf{' + lbl(metric) + '}}\\\\\n')
    #text.append('\\midrule\n')
    text.append(' & \\textbf{' + '} & \\textbf{'.join(map(lbl, df.columns.tolist())) + '}\\\\ \\midrule\n')
    for cat in feats:
        text.append('& \\multicolumn{5}{c}{\\textbf{' + lbl(cat) + '}}\\\\ \n')
        text.append(f'\\midrule\n')
        for feat in feats[cat]:
            
            row = df.loc[feat]
            std_row = df_std.loc[feat]

            textrow = lbl(row.name).replace('%','\%').replace('_','\_')
            for x, y in zip(row.values, std_row.values):
                ord_vals = sorted(row.values,reverse=True)
                if np.isnan(y): # binary feature
                    textrow += ' & {:.02f}\%'.format(x * 100)
                else:
                    textrow += ' & {:.01f} ({:.01f})'.format(x, y)
            textrow += '\\\\\n'
            text.append(textrow)
        text.append('\\midrule\n')

    text.append('\\bottomrule\n')
    text.append('\\end{tabular}\n')
    text.append('\\end{adjustbox}\n')
    text.append('\\caption{}\n')
    text.append('\\label{tbl:prediction}\n')
    text.append('\\end{table}\n')

    return [row.replace('_', '\_') for row in text]

print('- Loading data')

input_dir = Path('data/coupled/')
out_plots_folder = Path('gendata_1/clinical_features')
out_plots_folder.mkdir(parents=True, exist_ok=True)

branches_raw = pd.read_table('gendata_1/elpi/branches.tsv',index_col=0)
pheno_raw = pd.read_table(input_dir / 'pheno.tsv',index_col=0)
pheno_all_raw = pd.read_table(input_dir / 'pheno.feats.tsv',index_col=0)

pheno_all_vars = pd.read_table(input_dir / 'pheno.feats.vars.tsv',index_col=0)

pheno_vars = pd.read_table(input_dir / 'pheno.vars.tsv',index_col=0)

sids = branches_raw[branches_raw.branch.isin([1,3,4,5,6]) & branches_raw.top_50].index
branches = branches_raw.loc[sids]
pheno = pheno_raw.loc[sids]
pheno_all = pheno_all_raw.loc[sids, pheno_all_vars.index]
pheno_all['branch'] = branches.branch

pheno_all['gold_01'] = ((pheno_all['finalGold_P2'] == 0) | (pheno_all['finalGold_P2'] == 1)).astype(int)
for i in range(2,5):
    pheno_all['gold_' + str(i)] = (pheno_all['finalGold_P2'] == i).astype(int)
pheno_all['pct_prism'] = (pheno_all['finalGold_P2'] == -1).astype(int)
#out_plots_folder.mkdir(parents=True, exist_ok=True)

feats = {
            'demos':['Age_P2', 'gender', 'race', 'BMI_P2'], # Demos
            'spiro':['FEV1pp_utah_P2', 'FEV1_FVC_utah_P2', 'FVCpp_utah_P2', 'FEF2575_utah_P2', 'BDR_pct_FEV1_P2'], # Lung function
            'gold': ['gold_01','gold_2','gold_3','gold_4', 'pct_prism'], # GOLD
            'ctscan':['pctEmph_Thirona_P2', 'Pi10_Thirona_P2', 'AWT_seg_Thirona_P2', 'pctGasTrap_Thirona_P2'], # CT scans
            'sympt':['MMRCDyspneaScor_P2', 'Exacerbation_Frequency_P2', 'Chronic_Bronchitis_P2', 'distwalked_P2'], # Symptoms
            'sgrq':['SGRQ_scoreSymptom_P2', 'SGRQ_scoreActive_P2', 'SGRQ_scoreImpact_P2', 'SGRQ_scoreTotal_P2'], # SGRQ
            'smoke':['SmokCigNow_P2', 'ATS_PackYears_P2'], # Smoking
            'cbc':['neutrophl_pct_P2', 'lymphcyt_pct_P2', 'eosinphl_pct_P2', 'Platelets_P2'], # CBC
            'other':['HADS_ScorDepr_P2'], # Other
    }
feats_flat = sum(feats.values(),[])

feats_labels_dict = dict(zip([
            'Age_P2', 'gender', 'race', 'BMI_P2', # Demos
            'FEV1pp_utah_P2', 'FEV1_FVC_utah_P2', 'FVCpp_utah_P2', 'FEF2575_utah_P2', # Spirometry
            'BDR_pct_FEV1_P2', # Airway reactivity
            'gold_01','gold_2','gold_3','gold_4','pct_prism', # GOLD
            'pctEmph_Thirona_P2', 'Pi10_Thirona_P2', 'AWT_seg_Thirona_P2', 'pctGasTrap_Thirona_P2', # CT scans
            'MMRCDyspneaScor_P2', 'Exacerbation_Frequency_P2', 'Chronic_Bronchitis_P2', 'distwalked_P2', 'SGRQ_scoreSymptom_P2', 'SGRQ_scoreActive_P2', 'SGRQ_scoreImpact_P2', 'SGRQ_scoreTotal_P2', # Symptoms
            'SmokCigNow_P2', 'ATS_PackYears_P2', # Smoking
            'neutrophl_pct_P2', 'lymphcyt_pct_P2', 'eosinphl_pct_P2', 'Platelets_P2', # CBC
            'HADS_ScorDepr_P2', # Other
            'demos','spiro','gold','ctscan','sympt','sgrq','smoke','cbc','other' # sections
],[
    'Age', '% Female', '% AfroAm.', 'BMI',
    'FEV1%', 'FEV1/FVC', 'FVC%', 'FEF2575',
    'BDR FEV1%',
    'GOLD0/1', 'GOLD2', 'GOLD3', 'GOLD4', '% PRISm',
    '% Emph.', 'Pi10', 'AWT', '% Gas trap.',
    'mMRC', 'Exac.Freq.', 'Chron.Bronch.', 'Dist Walk.', 
    'SGRQ Symptom', 'SGRQ Active', 'SGRQ Impact', 'SGRQ Total',
    'Smoker', 'ATS PackYears',
    'Neutr.%', 'Lymph.%', 'Eosin.%', 'Platelets',
    'HADS Depress.',
    'Demographics','Spirometry','GOLD stage','CT imaging','Symptoms','SGRQ','Smoking','Complete blood counts','Other'
]))

# data_dicts = {
#     'gender': {1: 'M', 2:'F'},
#     'race': {1: 'NHW', 2:'AA'},
#     'Chronic_Bronchitis_P2': {0:'N',1:'Y'},
#     'Exacerbation_Frequency_P2': {0:'0',1:'1',2:'2',3:'3+'},
#     'MMRCDyspneaScor_P2': {i:str(i) for i in range(5)},
#     'SmokCigNow_P2':{0:'N',1:'Y'}
# }


#n_cols = max([len(val) for key,val in feats.items()])
#n_rows = len(feats)

#ordinal_feats = ['MMRCDyspneaScor_P2', 'Exacerbation_Frequency_P2']
#cats = ['demos','spiro','ctscan','sympt','sgrq','smoke','cbc','other']
#data_dict = {'gender':{1:'M',2:'F'},'race':{1:'CW',2:'NHB'},'Chronic_Bronchitis_P2':{0:'N',1:'Y'},'SmokCigNow_P2':{0:'N',1:'Y'}}

print('- Evaluating tables')

avg = pheno_all.groupby('branch').mean().loc[:,feats_flat].T
std = pheno_all.groupby('branch').std().loc[:,feats_flat].T

binary_feats = ['gender','race','Chronic_Bronchitis_P2','SmokCigNow_P2', 'gold_01','gold_2','gold_3','gold_4','pct_prism']
for feat in binary_feats:
    avg.loc[feat,branch_labels.keys()] = pheno_all.groupby('branch')[feat].apply(lambda x: np.nanmean(x-x.min())).loc[branch_labels.keys()]
    std.loc[feat] = np.nan

avg = avg.rename(columns=branch_labels)
avg_std = avg.rename(columns=branch_labels)

with open(out_plots_folder / 'clinical_features.tsv','w') as f:
    f.writelines(print_table(avg,std,feats=feats,lbl=lambda x: feats_labels_dict[x] if x in feats_labels_dict else str(x)))

print('-- Written table to {}'.format(out_plots_folder/'clinical_features.tex'))

print('- Generating FEV1pp_utah_P2 plot')

plt.figure(figsize=[4,4],dpi=200)
viz.plot_box('FEV1pp_utah_P2', pheno_all, branches.branch, clust_colors = ['firebrick','darkorange','steelblue','darkgreen','darkgreen'],fliers=False)

plt.savefig(out_plots_folder/'FEV1pp_utah_P2.pdf')

print('- Generating pctEmph_Thirona_P2 plot')

plt.figure(figsize=[4,4],dpi=200)
viz.plot_box('pctEmph_Thirona_P2', pheno_all, branches.branch, clust_colors = ['firebrick','darkorange','steelblue','darkgreen','darkgreen'],fliers=False)

plt.savefig(out_plots_folder/'pctEmph_Thirona_P2.pdf')

print('- Generating Chronic_Bronchitis_P2 plot')

plt.figure(figsize=[4,4],dpi=200)
cat_colors = matplotlib.cm.get_cmap('hot')([0.2,0.8])
cat_colors[0] = matplotlib.colors.to_rgba('lightgray')
viz.plot_feat_stack('Chronic_Bronchitis_P2', pheno_all, branches.branch, normalize=True, cat_colors=cat_colors, show_labels=False)

plt.savefig(out_plots_folder/'Chronic_Bronchitis_P2.pdf')

print('- Generating Exacerbation_Frequency_P2 plot')

plt.figure(figsize=[4,4],dpi=200)
cat_colors = matplotlib.cm.get_cmap('hot')([0.3,0.4,0.5,0.6,0.7,0.8,0.9])
cat_colors[0] = matplotlib.colors.to_rgba('lightgray')
viz.plot_feat_stack('Exacerbation_Frequency_P2', pheno_all, branches.branch, normalize=True, cat_colors=cat_colors, show_labels=False)

plt.savefig(out_plots_folder/'Exacerbation_Frequency_P2.pdf')

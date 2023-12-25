# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data

input_dir = Path('data/copdgene_freeze4/coupled/coupled_P1P2_nosplits_rev6k_covariates')

branches_raw = pd.read_table('gendata/trajectories/2022_04_15_rev6k_anc_09_z_30_n_20/branches.tsv',index_col=0)
pheno_raw = pd.read_table(input_dir / 'pheno.tsv',index_col=0)
pheno_all_raw = pd.read_table(input_dir / 'pheno.feats.tsv',index_col=0)

pheno_all_vars = pd.read_table(input_dir / 'pheno.feats.vars.tsv',index_col=0)

pheno_vars = pd.read_table(input_dir / 'pheno.vars.tsv',index_col=0)

sids = branches_raw[branches_raw.branch.isin([1,3,4,5,6]) & branches_raw.top_50].index
branches = branches_raw.loc[sids]
pheno = pheno_raw.loc[sids]
pheno_all = pheno_all_raw.loc[sids, pheno_all_vars.index]


pheno_p3 = pd.read_table('data/copdgene_freeze4/pheno/raw/COPDGene_P1P2P3_Flat_SM_NS_Mar20.txt',index_col=0)
pheno_p3 = pheno_p3.loc[sids]

# Functions


def plot_feat_stack(feat, feat_data, labels, data_dict=None, normalize=False, clust_to_show=None, cat_colors=None, clust_labels=None, vals_order=None, show_percentage=False):
    if cat_colors is None:
        cat_colors = ['olivedrab', 'steelblue', 'lightcoral','magenta','orange','blue','red','green']
    cids = clust_to_show if clust_to_show is not None else sorted(labels.unique())
    if vals_order is None:
        allvals = sorted(feat_data[feat].dropna().unique().tolist())
    else:
        allvals = sorted(feat_data[feat].dropna().unique().tolist(), key=lambda x: vals_order[x])
    if data_dict is None:
        allvals_lbl = allvals
    elif isinstance(data_dict, dict):
        allvals_lbl = [data_dict[val] if val in data_dict else val for val in allvals]
    else:
        allvals_lbl = [data_dict(val) for val in allvals]
    major_ticks = np.arange(len(cids))
    #minor_ticks = (np.arange(len(allvals))-len(allvals)/2+0.5)/len(allvals)*0.6
    for c in cids:
        barvals = np.array([(feat_data.loc[feat_data.index.isin(labels[labels==c].index.tolist()),feat]==val).sum() for val in allvals])
        if normalize:
            barvals = barvals / barvals.sum()
        for j in range(len(barvals)):
            plt.bar(cids.index(c), barvals[j],width=0.8,bottom=barvals[:j].sum(), color=cat_colors[j],linewidth=1,edgecolor='k')
            text = f'{allvals_lbl[j]}'
            perc = barvals[j]
            if show_percentage:
                text+='\n({:.0%})'.format(perc)
            if perc < 0.12:
                fontsize=6
            else:
                fontsize=None
            if perc > 0.05:
                plt.text(cids.index(c),perc/2+barvals[:j].sum(),text,ha='center',va='center',color='white',fontsize=fontsize)
                
        #for j in range(len(allvals)):
    if clust_labels is None:
        plt.xticks(major_ticks, [f'Clust {c}' for c in cids]);
    else:
        plt.xticks(major_ticks, clust_labels);
    plt.ylim([0,1])
    plt.box('off')


def get_epd_type(row, include_gold=True):
    emph = row.pctEmph_Thirona_P2
    gold = row.finalGold_P2
    if gold == 0:
        return 'GOLD 0' if include_gold else np.nan
    elif gold == 1:
        return 'GOLD 1' if include_gold else np.nan
    elif gold == -1:
        return 'PRISM' if include_gold else np.nan
    elif emph != emph:
        return np.nan
    elif emph <= 5:
        return 'NEPD'
    elif emph <= 10:
        return 'IE'
    else:
        return 'EPD'


# EPD/APD/NEPD subtypes

pheno_p3['EPD_subtype'] = pheno_p3.apply(lambda x: get_epd_type(x, False),axis=1)
pheno_p3['branch'] = branches.branch

pheno_p3_curr = pheno_p3.copy()
pheno_p3_curr = pheno_p3_curr.dropna(subset=['pctEmph_Thirona_P2'])
branches_curr = branches.loc[pheno_p3_curr.index]
plt.figure(dpi=200)
plot_feat_stack('EPD_subtype', pheno_p3_curr, branches_curr.branch, normalize=True, vals_order={'PRISM': 0, 'GOLD 0':1, 'GOLD 1':2,  'NEPD':3,'IE':4,'EPD':5}, clust_labels=['SEV','MOD','SYMPT','NORM1','NORM2'], show_percentage=True)
plt.ylabel('Percentage',fontsize=12)
plt.savefig('gendata/trajectories/2022_04_15_rev6k_anc_09_z_30_n_20/epd_nepd_mod_to_severe.pdf')


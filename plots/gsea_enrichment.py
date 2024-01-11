import click

def plot_GSEAPy_paths_FDR(fig_name,enrich1_hall,enrich3_hall,enrich4_hall,pval_thresh, top_n=0, hallmark=False, order_by='NES'):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    if top_n==0:
        enrich1_sig = enrich1_hall.loc[enrich1_hall['FDR q-val']<pval_thresh].index.tolist()
        enrich3_sig = enrich3_hall.loc[enrich3_hall['FDR q-val']<pval_thresh].index.tolist()
        enrich4_sig = enrich4_hall.loc[enrich4_hall['FDR q-val']<pval_thresh].index.tolist()
    elif top_n!=0:
        enrich1_sig = enrich1_hall.loc[enrich1_hall['FDR q-val']<pval_thresh].sort_values(by=['FDR q-val']).index.tolist()[:top_n]
        enrich3_sig = enrich3_hall.loc[enrich3_hall['FDR q-val']<pval_thresh].sort_values(by=['FDR q-val']).index.tolist()[:top_n]
        enrich4_sig = enrich4_hall.loc[enrich4_hall['FDR q-val']<pval_thresh].sort_values(by=['FDR q-val']).index.tolist()[:top_n]
    #   
    enrichall_sig_notord = list(set(enrich1_sig + enrich3_sig + enrich4_sig))
    enrichall_sig=enrich1_hall.loc[enrichall_sig_notord].sort_values(by=[order_by]).index.tolist()
    x1 = enrich1_hall.loc[enrichall_sig].NES.values
    x3 = enrich3_hall.loc[enrichall_sig].NES.values
    x4 = enrich4_hall.loc[enrichall_sig].NES.values
    y = np.arange(len(enrichall_sig))
    x1_notsig = (enrich1_hall.loc[enrichall_sig]['FDR q-val']>pval_thresh).values
    x3_notsig = (enrich3_hall.loc[enrichall_sig]['FDR q-val']>pval_thresh).values
    x4_notsig = (enrich4_hall.loc[enrichall_sig]['FDR q-val']>pval_thresh).values
    #
    fig, ax1 = plt.subplots(figsize=[20,20])
    ax1.scatter(x1[~x1_notsig],y[~x1_notsig],s=250,c='firebrick',marker="o",label='SEV')
    ax1.scatter(x3[~x3_notsig],y[~x3_notsig],s=250,c='darkorange',marker="s",label='MOD')
    ax1.scatter(x4[~x4_notsig],y[~x4_notsig],s=350,c='steelblue',marker="X",label='SYMPT')
    ax1.scatter(x1[x1_notsig],y[x1_notsig],c='firebrick',s=250,marker="o",alpha=0.3)
    ax1.scatter(x3[x3_notsig],y[x3_notsig],c='darkorange',s=250,marker="s",alpha=0.3)
    ax1.scatter(x4[x4_notsig],y[x4_notsig],c='steelblue',s=350,marker="X",alpha=0.3)
    ax1.set_yticklabels(['' for i in enrichall_sig])
    ax1.set_yticks(y)
    ax1.set_xticks(np.arange(-3,3,0.5))
    ax1.tick_params(axis='both',labelsize=20)  
    plt.grid()
    #
    for i_path,single_path in enumerate(enrichall_sig):
        if hallmark==False:
            t = plt.text(0,y[i_path],'%s'%single_path,va='center',ha='center',fontsize=15)
            t.set_bbox(dict(facecolor='white',edgecolor='white'))
        elif hallmark==True:
            t = plt.text(0,y[i_path],'%s'%single_path.replace('HALLMARK_', ''),va='center',ha='center',fontsize=15)
            t.set_bbox(dict(facecolor='white',edgecolor='white'))
    ax1.legend(fontsize=20,loc='lower right')
    ax1.set_xlabel('NES',fontsize=20)

    if not os.path.exists(os.path.dirname(fig_name)):
        os.makedirs(os.path.dirname(fig_name))

    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    return enrich1_sig,enrich3_sig,enrich4_sig,enrichall_sig   


@click.command()
@click.argument('de_dir', type=str)
@click.argument('out_path', type=str)
@click.option('--fdr_thresh', default=0.05, help='FDR threshold')
def main(de_dir, out_path, fdr_thresh):
    import pandas as pd
    from pathlib import Path

    de_dir = Path(de_dir)
    pre_res1 = pd.read_csv(de_dir / 'enrich_1high_vs_healthy_hallmark_cellcov_nobasophl_GSEAPy.tsv', sep='\t')
    pre_res3 = pd.read_csv(de_dir / 'enrich_3high_vs_healthy_hallmark_cellcov_nobasophl_GSEAPy.tsv', sep='\t')
    pre_res4 = pd.read_csv(de_dir / 'enrich_4high_vs_healthy_hallmark_cellcov_nobasophl_GSEAPy.tsv', sep='\t')
    pre_res_db1 = pre_res1.set_index('Term')
    pre_res_db3 = pre_res3.set_index('Term')
    pre_res_db4 = pre_res4.set_index('Term')

    enrich1_sig, enrich3_sig, enrich4_sig, enrichall_sig = plot_GSEAPy_paths_FDR(out_path, pre_res_db1, pre_res_db3, pre_res_db4, pval_thresh=fdr_thresh, hallmark=True, order_by='NES')

if __name__ == '__main__':
    main()

import click

@click.command()
@click.argument('gendata_dir', type=str)
@click.argument('out_dir', type=str)
def main(gendata_dir, out_dir):

    import pandas as pd
    from tqdm.auto import tqdm, trange
    import matplotlib.pyplot as plt
    from pathlib import Path
    from lifelines import KaplanMeierFitter
    from lifelines import CoxPHFitter
    import numpy as np
    import os

    import matplotlib.ticker as ticker
    import matplotlib
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    import proplot as pplt

    branch_labels = ['SEV','MOD','SYMPT','NORM1','NORM2']
    branch_ids = [1,3,4,5,6]
    branch_colors = ['firebrick','darkorange','steelblue','darkgreen','darkgreen']

    print('- Loading')

    #gendata_dir = Path('gendata_2/data')
    gendata_dir = Path(gendata_dir)
    out_dir = Path(out_dir)

    branches_dir = gendata_dir / 'elpi'
    data_dir = gendata_dir / 'data'

    branches_raw = pd.read_table(branches_dir / 'branches.tsv',index_col=0)

    sids = branches_raw[branches_raw.branch.isin([1,3,4,5,6]) & branches_raw.top_50].index
    branches = branches_raw.loc[sids]

    pheno_p3 = pd.read_table(data_dir / 'pheno.tsv',index_col=0)
    pheno_p3 = pheno_p3.loc[sids]

    mortality = pheno_p3[['P3_vitalstatus','vital_status_3yr','vital_status','days_followed']]

    #sm_sidlevel_raw = pd.read_table(data_dir / 'raw/LFU_SidLevel_Comorbid_SM_31AUG22.txt',index_col=0)

    lfu = pd.read_table(data_dir / 'lfu.tsv')


    out_dir.mkdir(parents=True, exist_ok=True)
    out_plots_folder = out_dir / 'plots'
    out_plots_folder.mkdir(parents=True, exist_ok=True)

    print('- Kaplan Meier curves')

    curves = []
    fig = pplt.figure()
    axs = fig.subplots()

    for i,b_i in enumerate([1,3,4,5,6]):
        idx = (branches.branch==b_i) & (branches.top_50)
        plotdata = mortality[idx]
        kmf = KaplanMeierFitter(label=f'{branch_labels[i]}').fit(plotdata.days_followed,plotdata.vital_status)
        curves.append(kmf.plot(color=branch_colors[i]))


    axs.format(
            title='Mortality',
            title_kw={'fontsize': 'x-large'},
            xlabel='Days followed',
            ylabel='Percent survival',
            xlim=[1300,5200],
        )
    axs.legend(ncols=2, label='Branches')
    plt.savefig(out_plots_folder / 'kaplan_meier.pdf')

    print('- Cox proportional hazard model')

    cph_matrix = mortality[['vital_status','days_followed']].copy()

    cph_matrix.loc[:,branch_labels[:-1]] = pd.get_dummies(branches.branch)[[1,3,4,5]].values # Branch 6 is used as reference

    cph_matrix['Age_P2'] = pheno_p3.Age_P2
    cph_matrix['gender'] = pheno_p3.gender-1
    cph_matrix['race'] = pheno_p3.race-1

    cph_matrix = cph_matrix[~cph_matrix.isna().any(axis=1)]

    cph = CoxPHFitter()
    cph.fit(cph_matrix, duration_col='days_followed', event_col='vital_status')
    cph.print_summary(style='ascii')

    fig = pplt.figure()
    axs = fig.subplots()
    cph.plot(hazard_ratios=True, columns=branch_labels[:-1])
    axs.format(
            title='Mortality HR',
            title_kw={'fontsize': 'x-large'},
            xlabel='HR',
    )
    plt.savefig(out_plots_folder / 'hazard_ratios.pdf')

    cph.summary.to_csv(out_dir / 'summary.tsv',sep='\t')

    print('- FEV1 change')

    idx = pheno_p3['FEV1pp_utah_P2'].notna() & pheno_p3['FEV1pp_utah_P3'].notna()
    plotdata = pheno_p3.loc[idx].copy()
    plotdata['branch'] = branches.loc[plotdata.index].branch



    fig = pplt.figure(sharey=False)
    axs = fig.subplots(nrows=1, ncols=2)
    axs.format(
        
    )

    pplt.rc['grid.below'] = True
    for i in range(len(branch_ids)):
        b = branch_ids[i]
        y_p2 = plotdata[plotdata.branch==b].FEV1pp_utah_P2.values
        y_p3 = plotdata[plotdata.branch==b].FEV1pp_utah_P3.values
        axs[0].boxplot(i-0.2,y_p2, showfliers=False, fc=branch_colors[i], widths=0.3, alpha=0.4);
        axs[0].boxplot(i+0.2,y_p3, showfliers=False, fc=branch_colors[i], widths=0.3);
        axs[1].boxplot(i,(y_p3-y_p2)/y_p2, showfliers=False, fc=branch_colors[i], widths=0.5);
    axs[0].format(
            xticks=range(5),
            xtickminor=False,
            xticklabels=branch_labels,
            ylabel='FEV$_1$%pred.',
            title='FEV$_1$%pred. P2->P3'
        )
    axs[1].format(
            xticks=range(5),
            xtickminor=False,
            xticklabels=branch_labels,
            ylabel='FEV$_1$%pred. %change',
            title='FEV$_1$%pred. %change P2->P3',
            yformatter=lambda x, y: "{}%".format(round(x*100)),
            ylocator=0.1
        )

    plt.savefig(out_plots_folder / 'fev_change.pdf')

    print('- Poisson Bayes Mixed GLM')

    time_data = lfu[(lfu['days_since_P2']>0) & (lfu.sid.isin(sids))].copy()
    time_data[['age','sex','race']] = pheno_p3.loc[time_data.sid,['Age_P2','gender','race']].values
    time_data['sex'] = time_data.sex-1
    time_data['race'] = time_data.race-1
    time_data['branch'] = branches.reindex(time_data.sid).branch.astype(str).values
    time_data["branch"] = pd.Categorical(time_data["branch"])
    time_data["sid"] = pd.Categorical(time_data["sid"])

    temp_data = time_data[['sid','Net_Exacerbations','branch','age','sex','race']].copy()
    temp_data = temp_data.dropna()
    temp_data['age'] = (temp_data['age'] - temp_data['age'].mean()) / temp_data['age'].std()
    temp_data.to_csv(out_dir / 'lfu_data.tsv',sep='\t')

    #os.system('module load R/4.3.0')
    #os.system('Rscript src/analysis/prospective_exac_glm.r')

    # exac_glm_coef = pd.read_table(out_dir / 'exacerbations_glm_coef.tsv')
    # exac_glm_coef['ExpEstimate'] = np.exp(exac_glm_coef['Estimate'])
    # exac_glm_coef['ExpEstimateStdPlus'] = np.exp(exac_glm_coef['Estimate'] + exac_glm_coef['Std. Error'])
    # exac_glm_coef['ExpEstimateStdMinus'] = np.exp(exac_glm_coef['Estimate'] - exac_glm_coef['Std. Error'])
    # exac_glm_coef.to_csv(out_dir / 'exacerbations_glm_coef.tsv',sep='\t')

    # plotdata = exac_glm_coef.loc[[f'branch{i}' for i in [1,3,4,5]]]

    # x = plotdata.ExpEstimate.values
    # xerrplus = plotdata.ExpEstimateStdPlus.values - x
    # xerrminus = -(plotdata.ExpEstimateStdMinus.values - x)

    # fig = pplt.figure()
    # axs = fig.subplots(nrows=1,ncols=1)
    # axs.format(
    # )

    # axs.errorbar(x, np.arange(plotdata.shape[0]), 
    #              fmt='sk',
    #              xerr=[xerrminus, xerrplus], fillstyle='none')
    # axs.axvline(0,linestyle='--',color='black')
    # axs.invert_yaxis()
    # axs.format(
    #         title='Exacerbations IRR',
    #         title_kw={'fontsize': 'large'},
    #         xlabel='IRR',
    #         #ylabel='Subjects',
    #         xlabel_kw={'fontsize': 'large'},
    #         ylabel_kw={'fontsize': 'large'},
    #         yticks = np.arange(4),
    #         yticklabels= branch_labels[:-1]
    #     )

    # plt.savefig(out_plots_folder / 'exacerbations_glm_coef.pdf')

    print('- Plotting exacerbations')

    #time_data = sm_surveylevel_raw[(sm_surveylevel_raw['days_since_P2']>0) & (sm_surveylevel_raw.sid.isin(sids))].copy()
    time_data_plot = time_data[time_data.days_since_P2.notna()].copy()
    branch_sids = [list(set(time_data_plot.sid.unique()) & set(branches[branches.branch==b].index.tolist())) for b in branch_ids]

    max_N = max([len(elem) for elem in branch_sids])
    max_T = int(time_data_plot.days_since_P2.max()+1)

    heatmaps = []
    for i,branch in enumerate(branch_ids):
        heatmaps.append(np.zeros([max_N,max_T])*np.nan)
        time_data_branch = time_data.loc[time_data_plot.sid.isin(branch_sids[i])]
        for j,sid in enumerate(branch_sids[i]):
            time_data_sid = time_data_branch[time_data_branch.sid==sid].copy()
            time_data_sid = time_data_sid.sort_values('days_since_P2')
            
            days = np.concatenate([[0],time_data_sid.days_since_P2.values]).astype(int)
            values = time_data_sid.Net_Exacerbations.values
            for k in range(len(days)-1):
                heatmaps[i][j, max(days[k],days[k+1]-30*6):days[k+1]] = values[k]

    cmap = matplotlib.cm.YlOrRd.copy()
    cmap.set_bad('white',1.)
    fig = pplt.figure(refaspect=0.4, spanx=False)
    axs = fig.subplots(nrows=1,ncols=5)
    axs.format(
        suptitle='Exacerbations',
        suptitle_kw={'fontsize':'xx-large'}
    )
    for i in range(len(branch_ids)):
        im = axs[i].imshow(heatmaps[i], aspect='auto', interpolation='nearest', cmap=cmap, vmin=-1)
        axs[i].format(
            title=branch_labels[i],
            title_kw={'fontsize': 'x-large'},
            xlabel='Years since Phase 2',
            ylabel='Subjects',
            xlabel_kw={'fontsize': 'large'},
            ylabel_kw={'fontsize': 'large'},
            xminorlocator=None,
            xlocator=365,
            xformatter=lambda x, y: int(x/365)
        )
        if i == len(branch_ids)-1:
            axs[i].colorbar(im)
    plt.savefig(out_plots_folder / 'exacerbations.pdf')

if __name__ == '__main__':
	main()

import click

@click.command()
@click.argument('data_dir', type=str)
@click.argument('out_plots_dir', type=str)
def main(data_dir, out_plots_dir):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import proplot as pplt
    data_dir = Path(data_dir)
    out_plots_dir = Path(out_plots_dir)

    branch_labels = ['SEV','MOD','SYMPT','NORM1','NORM2']

    exac_glm_coef = pd.read_table(data_dir / 'exacerbations_glm_coef.tsv',index_col=0)
    exac_glm_coef['ExpEstimate'] = np.exp(exac_glm_coef['Estimate'])
    exac_glm_coef['ExpEstimateStdPlus'] = np.exp(exac_glm_coef['Estimate'] + exac_glm_coef['Std. Error'])
    exac_glm_coef['ExpEstimateStdMinus'] = np.exp(exac_glm_coef['Estimate'] - exac_glm_coef['Std. Error'])
    exac_glm_coef.to_csv(data_dir / 'exacerbations_glm_coef.tsv',sep='\t')

    plotdata = exac_glm_coef.loc[[f'branch{i}' for i in [1,3,4,5]]]

    x = plotdata.ExpEstimate.values
    xerrplus = plotdata.ExpEstimateStdPlus.values - x
    xerrminus = -(plotdata.ExpEstimateStdMinus.values - x)

    fig = pplt.figure()
    axs = fig.subplots(nrows=1,ncols=1)
    axs.format()

    axs.errorbar(x, np.arange(plotdata.shape[0]), 
                 fmt='sk',
                 xerr=[xerrminus, xerrplus], fillstyle='none')
    axs.axvline(0,linestyle='--',color='black')
    axs.invert_yaxis()
    axs.format(
            title='Exacerbations IRR',
            title_kw={'fontsize': 'large'},
            xlabel='IRR',
            #ylabel='Subjects',
            xlabel_kw={'fontsize': 'large'},
            ylabel_kw={'fontsize': 'large'},
            yticks = np.arange(4),
            yticklabels= branch_labels[:-1]
        )

    plt.savefig(out_plots_dir / 'exacerbations_glm_coef.pdf')

if __name__ == '__main__':
    main()

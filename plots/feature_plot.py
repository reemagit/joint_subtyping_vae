import click

@click.command()
@click.argument('gendata_dir', type=str)
@click.argument('feat_name', type=str)
@click.argument('out_path', type=str)
@click.option('--plot-type', type=click.Choice(['box', 'bar']), default='box', help='Type of plot')
def main(gendata_dir, feat_name, out_path, plot_type):
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from utils import viz

    gendata_dir = Path(gendata_dir)
    out_path = Path(out_path)

    data_dir = gendata_dir / 'data'
    branches_dir = gendata_dir / 'elpi'

    pheno = pd.read_table(data_dir / 'pheno.tsv',index_col=0)
    branches = pd.read_table(branches_dir / 'branches.tsv',index_col=0)

    sids = branches[branches.branch.isin([1,3,4,5,6]) & branches.top_50].index

    branches = branches.loc[sids]
    pheno = pheno.loc[sids]

    plt.figure(figsize=[4,4],dpi=200)
    if plot_type == 'box':
        viz.plot_box(feat_name, pheno, branches.branch, clust_colors = ['firebrick','darkorange','steelblue','darkgreen','darkgreen'],fliers=False)
    elif plot_type == 'bar':
        viz.plot_feat_stack(feat_name, pheno, branches.branch, normalize=True, cat_colors=None, show_labels=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path)

    print(f'Saved to {out_path}')

if __name__ == '__main__':
    main()



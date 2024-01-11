import click


@click.command()
@click.argument('gendata_dir', type=str)
def main(gendata_dir):
    
    import json
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm, trange
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import mannwhitneyu, ttest_ind

    import pickle
    from clintrajan import clintraj_eltree

    branch_labels = {1:'SEV',3:'MOD',4:'SYMPT',5:'NORM1',6:'NORM2'}

    gendata_dir = Path(gendata_dir)


    results = json.load(open(gendata_dir / 'stability/purity_results.json','r'))
    projs = pd.read_table(gendata_dir / 'elpi/branches.tsv',index_col=0)
    embeds_path = gendata_dir / 'model/embeddings.tsv'
    graph_dir = gendata_dir / 'elpi'

    out_plots_folder = gendata_dir / 'stability/plots'
    out_plots_folder.mkdir(parents=True, exist_ok=True)

    # Embeds
    embeds = pd.read_table(embeds_path, index_col=0)

    # Graph
    graph = pickle.load(open(graph_dir / 'elpi_tree.pickle', "rb"))

    # Labels

    embeds = embeds.loc[projs.index]
    elpi_coords = clintraj_eltree.return_eltree_coords(graph, embeds, projs.branch.values, scatter_parameter=0.05)

    def plot_significances(data_list, pos_list = None, delta_ratio = 30, bounds=None, no_fliers=False, test_func=mannwhitneyu, alternative='greater', text_kw={}):
        def non_nan(arr):
            arr = np.asarray(arr)
            return arr[~np.isnan(arr)]
        def mwpval(obs,rdm,alternative):
            return test_func(non_nan(obs),non_nan(rdm),alternative=alternative).pvalue
        if no_fliers:
            def whiskmax(data):
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                return data[data <= np.percentile(data, 75) + 1.5 * iqr].max()
            def whiskmin(data):
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                return data[data >= np.percentile(data, 25) - 1.5 * iqr].min()
            min_ = min([whiskmin(non_nan(data)) for pair in data_list for data in pair])
            max_ = max([whiskmax(non_nan(data)) for pair in data_list for data in pair])
        else:
            min_ = min([min(non_nan(data)) for pair in data_list for data in pair])
            max_ = max([max(non_nan(data)) for pair in data_list for data in pair])
        delta = (max_ - min_) / delta_ratio
        offset = 0
        if bounds is None:
            bounds = [1e-4, 1e-3, 0.05]
        bounds = [-np.inf] + bounds + [np.inf]
        sigtexts = ['***','**','*','n.s.']
        #for pair,pos in zip(data_list, pos_list):
        for pair,pos in zip(data_list,pos_list):
            #pair = data_list[int(pos[0])],data_list[int(pos[1])]
            pval = mwpval(non_nan(pair[0]), non_nan(pair[1]),alternative)
            print(pval)
            sigtext = sigtexts[np.digitize(pval, bounds)-1]
            x1,x2 = pos
            y = max_+offset+delta
            plt.plot([x1, x1, x2, x2], [y, y+delta, y+delta, y], lw=1., c='black')
            plt.text((x1 + x2) * .5, y+delta, sigtext, ha='center', va='bottom', color='black', **text_kw)
            offset += delta * 4
        if plt.gca().get_ylim()[1] < max_ + offset + delta:
            plt.gca().set_ylim([None, max_ + offset + 3*delta])

    def plot_binned(x, y, n_bins=10):
        bins = np.linspace(x.min()-(x.max()-x.min())*0.05, x.max()+(x.max()-x.min())*0.05, n_bins)
        bins_i = np.digitize(x, bins)
        for bin_i in range(len(bins)):
            plt.scatter(np.random.randn((bins_i==bin_i).sum())*0.1 + np.ones((bins_i==bin_i).sum()) * bin_i + 1, y[bins_i==bin_i],color='darkgray',s=0.5)
        plt.boxplot([y[bins_i==bin_i] for bin_i in range(len(bins))], showfliers=False)
        plt.xticks(range(1,11),[f'{i}%' for i in range(10,110,10)], fontsize=8)
        
    print('- Branches vs clusters')

    group_purities_plot = sorted(results['purities_top50'],key=np.median,reverse=True)[:5]
    group_purities_kmeans_plot = sorted(results['purities_kmeans'],key=np.median,reverse=True)[:5]
    group_purities_kmeans_top50_plot = sorted(results['purities_kmeans_top50'],key=np.median,reverse=True)[:5]
    delta = 0.25
    xx = np.arange(len(group_purities_plot))
    plt.figure(figsize=[24,8])
    bp1 = plt.boxplot(group_purities_plot,positions=xx+delta,widths=0.2,patch_artist=True,boxprops={'facecolor':'lightcoral'}, medianprops={'color':'white'})
    bp2 = plt.boxplot(group_purities_kmeans_plot,positions=xx-delta,widths=0.2,patch_artist=True,boxprops={'facecolor':'lightsteelblue'}, medianprops={'color':'white'})
    bp3 = plt.boxplot(group_purities_kmeans_top50_plot,positions=xx,widths=0.2,patch_artist=True,boxprops={'facecolor':'steelblue'}, medianprops={'color':'white'})
    plt.legend([bp2["boxes"][0], bp3["boxes"][0], bp1["boxes"][0]], ['K-means', 'K-means (core)', 'Branches'], loc='lower center',ncol=3,fontsize=30)
    plt.xticks(xx, xx, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid('on')
    #plt.ylim([0,1.1])
    plt.title('Cluster purity',fontsize=30)
    plt.xlabel('Rank',fontsize=30)
    plt.ylabel('Purity',fontsize=30)
    for i in range(len(group_purities_plot)):
        plot_significances([[group_purities_kmeans_top50_plot[i],group_purities_plot[i]],[group_purities_kmeans_plot[i],group_purities_plot[i]]],[(i,i+delta),(i-delta,i+delta)],test_func=mannwhitneyu,alternative='less',text_kw={'fontsize':30})
    plt.tight_layout()
    plt.savefig(out_plots_folder/'branches_vs_clusters.pdf')
    print('Printed to ', out_plots_folder/'branches_vs_clusters.pdf')

    print('- Point purities')

    plt.figure(figsize=[12,8],dpi=200)


    for i,branch_i in enumerate([1,3,4,5,6]):
        idx = np.where(projs.branch.values == branch_i)[0]
        plt.subplot(2,3,i+1)
        plot_binned(projs.branch_pos.values[idx], np.asarray(results['point_purities'])[idx])
        plt.ylim([0,1.1])
        plt.title(f'{branch_labels[branch_i]}',fontsize=20)
        plt.xlabel('Position on branch',fontsize=20)
        plt.ylabel('Purity',fontsize=20);

    plt.subplot(2,3,6)
    branches = [1,3,4,5,6]
    branch_colors = {1:'firebrick',3:'darkorange',4:'steelblue',5:'darkgreen',6:'darkgreen',0:'lightgray',2:'lightgray'}
    plt.scatter(elpi_coords['x_data'], elpi_coords['y_data'], c='lightgray',alpha=0.4,s=0.5)
    for perc in range(10,110,10):
        for i in range(len(branches)):
            perc_val = np.percentile(projs[projs.branch==branches[i]].branch_pos,perc)
            idx = ((projs.branch==branches[i]) & (projs.branch_pos>=perc_val)).values
            plt.scatter(elpi_coords['x_data'][idx], elpi_coords['y_data'][idx], c=branch_colors[branches[i]],alpha=0.1,s=1)

    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    plt.tight_layout()
    plt.savefig(out_plots_folder/ 'point_purities.pdf')
    print('Printed to ', out_plots_folder/'point_purities.pdf')


if __name__ == '__main__':
    main()
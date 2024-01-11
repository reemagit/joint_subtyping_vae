import click

@click.command()
@click.argument('de_dir', type=str)
@click.argument('out_path', type=str)
def main(de_dir, out_path):
    import pandas as pd
    import numpy as np
    import gseapy as gp
    import biorosetta as br
    from pathlib import Path

    idmap = br.IDMapper('all')

    def gendata_GSEAPy_new(DE_branches,contrast=""):
        DE_contrast = DE_branches[['genesID','P.Value_'+contrast,'logFC_'+contrast]].copy()
        DE_contrast['genesID'] = DE_contrast.genesID.str.split('.').str[0]
        DE_contrast['genes_symb'] = idmap.convert(DE_contrast.genesID.tolist(),'ensg','symb')
        DE_contrast['rank'] = -np.log10(DE_contrast['P.Value_'+contrast].values)*np.sign(DE_contrast['logFC_'+contrast].values)
        DE_contrast.sort_values(by=['rank'],ascending=False,inplace=True)
        return DE_contrast

    de_dir = Path(de_dir)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)


    #Load data
    joined_DE_cellcov = pd.read_csv(de_dir / 'joined_DE_cellcov_nobasophl.tsv',sep='\t')
    

    #Perform enrichment
    DE1_vs_healthy = gendata_GSEAPy_new(joined_DE_cellcov,'col1high-colhealthy')
    DE3_vs_healthy = gendata_GSEAPy_new(joined_DE_cellcov,'col3high-colhealthy')
    DE4_vs_healthy = gendata_GSEAPy_new(joined_DE_cellcov,'col4high-colhealthy')
    rank1 = DE1_vs_healthy[['genes_symb','rank']].set_index('genes_symb').copy()
    pre_res1 = gp.prerank(rnk=rank1,gene_sets='MSigDB_Hallmark_2020',threads=4,min_size=5,max_size=1000,permutation_num=1000,outdir=None,seed=6,verbose=True,)
    rank3 = DE3_vs_healthy[['genes_symb','rank']].set_index('genes_symb').copy()
    pre_res3 = gp.prerank(rnk=rank3,gene_sets='MSigDB_Hallmark_2020',threads=4,min_size=5,max_size=1000,permutation_num=1000,outdir=None,seed=6,verbose=True,)
    rank4 = DE4_vs_healthy[['genes_symb','rank']].set_index('genes_symb').copy()
    pre_res4 = gp.prerank(rnk=rank4,gene_sets='MSigDB_Hallmark_2020',threads=4,min_size=5,max_size=1000,permutation_num=1000,outdir=None,seed=6,verbose=True,)
    pre_res_db1 = pre_res1.res2d.set_index('Term')
    pre_res_db3 = pre_res3.res2d.set_index('Term')
    pre_res_db4 = pre_res4.res2d.set_index('Term')
    pre_res_db1.to_csv(out_dir / 'enrich_1high_vs_healthy_hallmark_cellcov_nobasophl_GSEAPy.tsv',sep='\t')
    pre_res_db3.to_csv(out_dir / 'enrich_3high_vs_healthy_hallmark_cellcov_nobasophl_GSEAPy.tsv',sep='\t')
    pre_res_db4.to_csv(out_dir / 'enrich_4high_vs_healthy_hallmark_cellcov_nobasophl_GSEAPy.tsv',sep='\t')










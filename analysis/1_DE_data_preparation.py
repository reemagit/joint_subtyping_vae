import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path('data/coupled')
branches_dir = Path('gendata_1/elpi')
out_dir = Path('gendata_1/DE')
out_dir.mkdir(parents=True, exist_ok=True)

pheno = pd.read_csv(data_dir / 'pheno.feats.tsv',sep='\t',index_col=0)
branches = pd.read_csv(branches_dir / 'branches.tsv',sep='\t',index_col=0)
pheno.loc[branches.index.tolist()]\
        [['Age_P2','race','gender','neutrophl_pct_P2','lymphcyt_pct_P2','monocyt_pct_P2','eosinphl_pct_P2','basophl_pct_P2']]\
            .to_csv(out_dir / 'pheno_branches_covariates_cellcov.tsv',sep='\t')

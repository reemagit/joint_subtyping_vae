import pandas as pd
import numpy as np
import os

in_pheno_dir = 'data/pheno/raw/'
out_pheno_dir = 'data/pheno/processed/'

pheno = pd.read_table(in_pheno_dir + 'P1P2_Pheno_Flat_All_sids_Mar19.txt',index_col='sid')

pheno_dict = pd.read_table(in_pheno_dir + 'P1P2_Pheno_Flattened_Mar19_DataDict_categorized2.csv',index_col='VariableName')

# Some data munging to make the columns match up
indexes_to_rename = ['years_from_baseline'] + pheno_dict[pheno_dict.Form == 'CT QA'].index.tolist() 
pheno_dict.rename(index={idx: idx + '_P2' for idx in indexes_to_rename}, inplace=True)

pheno = pheno[pheno_dict.index.tolist()]

pheno['O2_Therapy_P2'] = pheno.O2_Therapy_P2.apply(lambda x: np.nan if x!=x or x=='.' else int(x)) # this column is messed up

pheno.to_csv(out_pheno_dir + 'pheno.tsv',sep='\t',index_label='COPDgeneID')



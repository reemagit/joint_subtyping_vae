import numpy as np
import pandas as pd
from pathlib import Path

input_dir = Path('data/coupled')
model_dir = Path('gendata')
out_dir = Path('data/pheno/processed')

pheno_p3 = pd.read_table('data/pheno/raw/COPDGene_P1P2P3_Flat_SM_NS_Mar20.txt',index_col=0)

vit = pd.read_table('data/pheno/raw/COPDGene_VitalStatus_SM_NS_Oct22.csv',sep=',',index_col=0)
vit = vit.loc[pheno_p3.index]


vit['P3_days'] = pheno_p3.Days_Phase1_Phase3
vit['P2P3_days'] = pheno_p3.Days_Phase2_Phase3
vit['days_since_P2'] = vit.days_followed - pheno_p3.Days_Phase1_Phase2
vit['P3_death'] = (vit.vital_status==1) & (pheno_p3.visit_pattern=='P1P2--') & (vit.days_followed<365*11)
vit['P3_vitalstatus'] = vit['P3_death'].astype(int)

vit['vital_status_3yr'] = ((vit.vital_status==1) & (pheno_p3.visit_pattern=='P1P2--') & (vit.days_since_P2<365*3)).astype(int)

# these are people that are registered as alive but did not go to P3 and have not been followed more than 9 years after baseline, so probably dropouts
vit.loc[(vit.days_followed<365*9) & (vit.P3_vitalstatus==0) & (pheno_p3.visit_pattern!='P1P2P3'),'P3_vitalstatus'] = np.nan
# these are people that are registered as alive but did not go to P3 and have not been followed more than 3 years after P2, so probably dropouts
vit.loc[(vit.days_since_P2<365*3) & (vit.vital_status_3yr==0) & (pheno_p3.visit_pattern!='P1P2P3'),'vital_status_3yr'] = np.nan

out_dir.mkdir(exist_ok=True,parents=True)
vit.to_csv(out_dir / 'vital_status.tsv',sep='\t')


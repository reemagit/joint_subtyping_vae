module load R/3.6.3

# Preprocess
python preprocess/1_process_pheno.py
Rscript preprocess/2_process_expr.R
python preprocess/3_gene_selection.py
python preprocess/4_process_pheno_expr.py
python preprocess/5_mortality.py

# Training
python train/train_model.py data/coupled/ gendata_1/model_retrain -m -1 -n 20 -z 30 -bs 128 -h1 1024,512 -h2 64 -p 0 -mmd 100 -a12 0.5 -anc 0.9 -l1 0 -l2 500 -lr 0.00136119 -s 42

# Outcome prediction
python analysis/cca.py gendata_1/model --out_dir gendata_1/cca
python analysis/mofa.py gendata_1/model --out_dir gendata_1/mofa
python analysis/outcome_prediction.py -o gendata_1/outcome_prediction gendata_1/model gendata_1
python plots/classification_table.py gendata_1/outcome_prediction/classification_avg.tsv gendata_1/outcome_prediction/classification_std.tsv gendata_1/outcome_prediction/classification_pval.tsv -o gendata_1/outcome_prediction/performances.tex

# Perturbed embeddings
python train/train_model_stability.py data/coupled/ gendata_1/stability -t 100 -m -1 -n 20 -z 30 -bs 128 -h1 1024,512 -h2 64 -p 0 -mmd 100 -a12 0.5 -anc 0.9 -l1 0 -l2 500 -lr 0.00136119

# ElPiGraph
python analysis/elpigraph_tree.py gendata_1/model/embeddings.tsv gendata_1/elpi -p gendata_1/stability

# Stability and plots
python analysis/stability.py
python plots/stability.py


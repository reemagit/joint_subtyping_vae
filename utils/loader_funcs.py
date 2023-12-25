import pandas as pd

def load_splits(path):
	return pd.read_table(path, index_col=0,sep=',')['0']

def save_splits(splits, path):
	splits.to_csv(path,sep=',')

def load_embeddings(path):
	return pd.read_table(path, index_col=0)

def save_embeddings(embeds, path):
	embeds.to_csv(path, sep='\t')
import logging
import sys
from pathlib import Path

import numpy as np
#from umap import UMAP
from sklearn.decomposition import PCA
from tqdm.auto import tqdm, trange

import pandas as pd
import seaborn as sns




def setup_logging(output_dir, name):
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
								  '%m-%d-%Y %H:%M:%S')

	file_handler = logging.FileHandler(out_dir / 'logs.log')
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)

	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setLevel(logging.DEBUG)
	stdout_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stdout_handler)
	return logger


def get_pca(data, n_components=50, print_expl_var=False, return_obj=False):
    pca = PCA(n_components=n_components,random_state=0)
    pca.fit(data.values)
    pca_vals = pca.transform(data.values)
    pca_vals = pd.DataFrame(pca_vals, index=data.index, columns=[f'pca_{i}' for i in range(n_components)])
    if print_expl_var:
        print(np.cumsum(pca.explained_variance_ratio_))
    if return_obj:
        return pca_vals, pca
    else:
        return pca_vals

# def get_umap(data, pca_comp=None, n_neighbors=15, return_obj=False, random_state=None):
#     if pca_comp is not None:
#         data = get_pca(data, pca_comp)
#     umap_obj = UMAP(n_neighbors=n_neighbors,random_state=random_state).fit(data.values)
#     umap_vals = umap_obj.transform(data.values)
#     umap_vals = pd.DataFrame(umap_vals, index=data.index, columns=[f'umap_{i}' for i in range(2)])
#     if return_obj:
#         return umap_vals, umap_obj
#     else:
#         return umap_vals

def plot_feature(data, color, qup=100, qdown=0,highlight=None,order=None,*args, **kwargs):
    if isinstance(data,pd.DataFrame):
        data = data.values
    if isinstance(color, pd.Series):
        color = color.values

    if highlight is not None:
        order = np.argsort((color==highlight).astype(int))
        data = data[order,:]
        color = color[order]
    if order is not None:
        if order=='ascending':
            order = np.argsort(color)
        else:
            order = np.argsort(color)[::-1]
        data = data[order,:]
        color = color[order]

    if np.isreal(color[0]):
        up = np.percentile(color[~np.isnan(color)],qup)
        down = np.percentile(color[~np.isnan(color)],qdown)
        if highlight is not None:
            sns.scatterplot(x=data[color!=highlight,0],y=data[color!=highlight,1],hue=color[color!=highlight],hue_norm=(down,up),alpha=0.4,*args, **kwargs)
            sns.scatterplot(x=data[color==highlight,0],y=data[color==highlight,1],c=['red'] * (color==highlight).sum(),hue_norm=(down,up),*args, **kwargs)
        else:
            sns.scatterplot(x=data[:,0],y=data[:,1],hue=color,hue_norm=(down,up),*args, **kwargs)
    else:
        if highlight is not None:
            sns.scatterplot(x=data[color!=highlight,0],y=data[color!=highlight,1],hue=color[color!=highlight],alpha=0.4,*args, **kwargs)
            sns.scatterplot(x=data[color==highlight,0],y=data[color==highlight,1],c=['red'] * (color==highlight).sum(),*args, **kwargs)
        else:
            sns.scatterplot(x=data[:,0],y=data[:,1],hue=color,*args, **kwargs)



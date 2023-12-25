import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def convert_to_proplot(list_of_lists):
    maxval = max([len(elem) for elem in list_of_lists])
    out = np.empty((maxval, len(list_of_lists)))
    out[:] = np.nan
    for i,elem in enumerate(list_of_lists):
        out[:len(elem),i] = elem
    return out

def plot_feature(data, color, qup=100, qdown=0,highlight=None,order='ascending',*args, **kwargs):
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

def plot_bar(feat, feat_data, labels, clust_colors, data_dict=None, normalize=False):
    cids = sorted(labels.unique())
    allvals = sorted(feat_data[feat].dropna().unique().tolist())
    if data_dict is None:
        allvals_lbl = allvals
    elif isinstance(data_dict, dict):
        allvals_lbl = [data_dict[val] if val in data_dict else val for val in allvals]
    else:
        allvals_lbl = [data_dict(val) for val in allvals]
    major_ticks = np.arange(len(cids))
    minor_ticks = (np.arange(len(allvals))-len(allvals)/2+0.5)/len(allvals)*0.6
    for c in cids:
        barvals = [(feat_data.loc[feat_data.index.isin(labels[labels==c].index.tolist()),feat]==val).sum() for val in allvals]
        if normalize:
            barvals = [val/sum(barvals) for val in barvals]
        plt.bar(minor_ticks+cids.index(c), barvals,width=0.1,color=clust_colors[c])
        for j in range(len(allvals)):
            plt.text(minor_ticks[j]+cids.index(c),barvals[j]+max(barvals)*0.05,f'{allvals_lbl[j]}',ha='center',va='bottom')
    plt.xticks(major_ticks, [f'Clust {c+1}' for c in cids]);
    plt.ylim([None,plt.ylim()[1]+plt.ylim()[1]*0.1])
    if normalize:
        plt.ylim([0,1.1])

def plot_box(feat, feat_data, labels, clust_colors, fliers=False):
    cids = sorted(labels.unique())
    vals = {c:feat_data.loc[feat_data.index.isin(labels[labels==c].index.tolist()),feat].dropna().values for c in cids}
    bp=plt.boxplot([vals[c] for c in cids],positions=range(len(cids)),patch_artist=True,boxprops={'facecolor':'lightgray'},medianprops={'color':'darkgray'}, showfliers=fliers);
    for i in range(len(cids)):
        bp['boxes'][i].set(facecolor=clust_colors[i])
    plt.xticks(range(len(cids)), [f'Clust {c}' for c in cids])

def plot_feat_scatter(x_data, y_data, color, qup=100, qdown=0,highlight=None,order='ascending',*args, **kwargs):
    xy_data = np.column_stack([x_data,y_data])
    #if isinstance(xy_data,pd.xy_dataFrame):
    #    xy_data = xy_data.values
    if isinstance(color, pd.Series):
        color = color.values    

    if highlight is not None:
        order = np.argsort((color==highlight).astype(int))
        xy_data = xy_data[order,:]
        color = color[order]
    if order is not None:
        if order=='ascending':
            order = np.argsort(color)
        else:
            order = np.argsort(color)[::-1]
        xy_data = xy_data[order,:]
        color = color[order]

    if np.isreal(color[0]):
        up = np.percentile(color[~np.isnan(color)],qup)
        down = np.percentile(color[~np.isnan(color)],qdown)
        if highlight is not None:
            sns.scatterplot(x=xy_data[color!=highlight,0],y=xy_data[color!=highlight,1],hue=color[color!=highlight],hue_norm=(down,up),alpha=0.4,*args, **kwargs)
            sns.scatterplot(x=xy_data[color==highlight,0],y=xy_data[color==highlight,1],c=['red'] * (color==highlight).sum(),hue_norm=(down,up),*args, **kwargs)
        else:
            sns.scatterplot(x=xy_data[:,0],y=xy_data[:,1],hue=color,hue_norm=(down,up),*args, **kwargs)
    else:
        if highlight is not None:
            sns.scatterplot(x=xy_data[color!=highlight,0],y=xy_data[color!=highlight,1],hue=color[color!=highlight],alpha=0.4,*args, **kwargs)
            sns.scatterplot(x=xy_data[color==highlight,0],y=xy_data[color==highlight,1],c=['red'] * (color==highlight).sum(),*args, **kwargs)
        else:
            sns.scatterplot(x=xy_data[:,0],y=xy_data[:,1],hue=color,*args, **kwargs)

def plot_feat_stack(feat, feat_data, labels, data_dict=None, normalize=False, clust_to_show=None, cat_colors=None, show_labels=True, data_labels_kw={}):
    if cat_colors is None:
        cat_colors = ['lightcoral', 'steelblue', 'olivedrab','magenta','orange','blue','red','green']
    cids = clust_to_show if clust_to_show is not None else sorted(labels.unique())
    allvals = sorted(feat_data[feat].dropna().unique().tolist())
    if data_dict is None:
        allvals_lbl = allvals
    elif isinstance(data_dict, dict):
        allvals_lbl = [data_dict[val] if val in data_dict else val for val in allvals]
    else:
        allvals_lbl = [data_dict(val) for val in allvals]
    major_ticks = np.arange(len(cids))
    #minor_ticks = (np.arange(len(allvals))-len(allvals)/2+0.5)/len(allvals)*0.6
    for c in cids:
        barvals = np.array([(feat_data.loc[feat_data.index.isin(labels[labels==c].index.tolist()),feat]==val).sum() for val in allvals])
        if normalize:
            barvals = barvals / barvals.sum()
        for j in range(len(barvals)):
            plt.bar(cids.index(c), barvals[j],width=0.8,bottom=barvals[:j].sum(), color=cat_colors[j])
            if show_labels:
                plt.text(cids.index(c),barvals[j]/2+barvals[:j].sum(),f'{allvals_lbl[j]}',ha='center',va='center',color='white', **data_labels_kw)
        #for j in range(len(allvals)):
            
    plt.xticks(major_ticks, [f'Clust {c}' for c in cids]);
    plt.ylim([None,plt.ylim()[1]+plt.ylim()[1]*0.1])
    plt.box('off')
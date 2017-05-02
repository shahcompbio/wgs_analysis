import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import pylab
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dst


def binary_heatmap(fig, df):

    seaborn.set_style('white')

    dat = df.values
    rownames = np.array(df.index.values)
    colnames = np.array(df.columns.values)

    # Compute and plot column dendrogram
    ax = fig.add_axes([0.0,0.8,1.0,0.2])
    D = dst.squareform(dst.pdist(dat.T, 'cityblock'))
    Y = sch.linkage(D, method='complete')
    Z = sch.dendrogram(Y, color_threshold=-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot distance matrix.
    ax = fig.add_axes([0.0,0.0,1.0,0.75])
    idx1 = np.lexsort(dat.T)
    idx2 = Z['leaves']
    dat = dat[idx1,:]
    dat = dat[:,idx2]
    rownames = rownames[idx1]
    colnames = colnames[idx2]
    im = ax.matshow(dat, aspect='auto', origin='lower', cmap=pylab.cm.Blues)
    ax.set_yticks([])
    ax.set_xticks(range(len(colnames)))
    ax.set_xticklabels(colnames)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



def intensity_heatmap(fig, df, sort_rows=True):

    seaborn.set_style('white')

    dat = np.log(df.values + 1)
    rownames = np.array(df.index.values)
    colnames = np.array(df.columns.values)

    # Compute and plot column dendrogram
    ax = fig.add_axes([0.0,0.8,0.9,0.15])
    D = dst.squareform(dst.pdist(dat.T, 'cityblock'))
    Y = sch.linkage(D, method='complete')
    Z = sch.dendrogram(Y, color_threshold=-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot distance matrix.
    ax = fig.add_axes([0.0,0.0,0.9,0.75])
    D2 = dst.squareform(dst.pdist(dat, 'cityblock'))
    Y2 = sch.linkage(D2, method='complete')
    Z2 = sch.dendrogram(Y2, color_threshold=-1, no_plot=True)
    idx1 = Z2['leaves']
    idx2 = Z['leaves']
    if sort_rows:
        dat = dat[idx1,:]
        rownames = rownames[idx1]
    dat = dat[:,idx2]
    colnames = colnames[idx2]
    im = ax.matshow(dat, aspect='auto', origin='lower', cmap=pylab.cm.Blues)
    ax.set_yticks([])
    ax.set_xticks(range(len(colnames)))
    ax.set_xticklabels(colnames)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='minor', color='black')
    
    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    cb = pylab.colorbar(im, cax=axcolor, drawedges=True)
    cb.outline.set_linewidth(0)
    cb.dividers.set_linewidth(0)
    axcolor.set_ylabel('log(read count + 1)')

    return Z

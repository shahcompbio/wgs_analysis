import argparse
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.backends.backend_pdf

import remixt.cn_plot

import ith_project.study
import wgs_analysis.plots.utils


def plot_cnv_genome_density(cn_table):
    wgs_analysis.plots.utils.setup_plot()
    fig = plt.figure(figsize=(12, 3))

    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 

    ax0 = wgs_analysis.plots.utils.setup_axes(plt.subplot(gs[0]))
    remixt.cn_plot.plot_cnv_genome(ax0, cn_table, maxcopies=6, major_col='major_raw', minor_col='minor_raw')
    ax0.set_ylabel('Raw copy number')

    ax1 = wgs_analysis.plots.utils.setup_axes(plt.subplot(gs[1]))
    cov = 0.001
    data = cn_table[['minor_raw', 'major_raw', 'length']].dropna()
    remixt.utils.filled_density_weighted(ax1, data['minor_raw'].values, data['length'].values, 'blue', 0.5, 0.0, 6.0, cov, rotate=True)
    remixt.utils.filled_density_weighted(ax1, data['major_raw'].values, data['length'].values, 'red', 0.5, 0.0, 6.0, cov, rotate=True)
    ax1.set_ylim(ax0.get_ylim())
    ax1.set_xlabel('Density')

    return fig


def create_cnv_plots(plot_filename, results_filenames):
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_filename)

    for results_filename in results_filenames:
        with pd.HDFStore(results_filename, 'r') as store:
            for key in store.keys():
                if not key.endswith('/cn'):
                    continue

                cn_table = store[key]

                incompatible = False
                for col in ['length', 'major_raw', 'minor_raw']:
                    if col not in cn_table.columns:
                        incompatible = True
                if incompatible:
                    continue

                data = cn_table.replace([np.inf, -np.inf], np.nan).dropna(subset=['length', 'major_raw', 'minor_raw'])
                length = data['length'].values
                major = data['major_raw'].values
                minor = data['minor_raw'].values
                raw_ploidy = (length * (major + minor)).sum() / length.sum()
                if np.isnan(raw_ploidy):
                    raise Exception('unable to calculate ploidy for {}, {}'.format(results_filename, key))

                fig = plot_cnv_genome_density(cn_table)
                fig.suptitle('{}, {}, ploidy={}'.format(results_filename, key, raw_ploidy))

                plt.tight_layout()

                pdf.savefig(bbox_inches='tight')
                plt.close()

    pdf.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--plot_filename',
                           required=True,
                           help='Plot filename')

    argparser.add_argument('--results_filenames',
                           required=True, nargs='+',
                           help='Plot filename')

    args = vars(argparser.parse_args())

    create_cnv_plots(args['plot_filename'], args['results_filenames'])


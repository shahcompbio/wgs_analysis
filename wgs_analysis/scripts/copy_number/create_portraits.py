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


def create_patient_portrait(plot_filename):
    pdf = matplotlib.backends.backend_pdf.PdfPages(plot_filename)

    study_info = ith_project.study.StudyInfo()

    feature_table = []

    for patient_id, sample_ids in study_info.get_patient_tumour_sample_map().iteritems():
        copy_number_results_filename = study_info.get_filename_template('copy_number_results').format(patient_id=patient_id)

        with pd.HDFStore(copy_number_results_filename, 'r') as copy_number_results:
            for sample_id in sample_ids:
                try:
                    cn_table = copy_number_results['/copy_number/remixt/sample_{sample_id}/cn'.format(sample_id=sample_id)]
                except KeyError:
                    print 'no copy number results for patient {patient_id}, sample {sample_id}'.format(patient_id=patient_id, sample_id=sample_id)
                    continue

                wgs_analysis.plots.utils.setup_plot()
                fig = plt.figure(figsize=(12, 3))

                gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 

                ax0 = wgs_analysis.plots.utils.setup_axes(plt.subplot(gs[0]))
                remixt.cn_plot.plot_cnv_genome(ax0, cn_table, maxcopies=6, major_col='major_raw', minor_col='minor_raw')
                ax0.set_ylabel('Raw copy number')
                ax0.set_title('Raw copy number for patient {patient_id}, sample {sample_id}'.format(patient_id=patient_id, sample_id=sample_id))

                ax1 = wgs_analysis.plots.utils.setup_axes(plt.subplot(gs[1]))
                cov = 0.000001
                data = cn_table[['minor_raw', 'major_raw', 'length']].dropna()
                remixt.utils.filled_density_weighted(ax1, data['minor_raw'].values, data['length'].values, 'blue', 0.5, 0.0, 6.0, cov, rotate=True)
                remixt.utils.filled_density_weighted(ax1, data['major_raw'].values, data['length'].values, 'red', 0.5, 0.0, 6.0, cov, rotate=True)
                ax1.set_ylim(ax0.get_ylim())
                ax1.set_xlabel('Density')

                plt.tight_layout()

                pdf.savefig(bbox_inches='tight')
                plt.close()

    #         features = copy_number_results['/copy_numbers/destruct/copy_number']
    #         features['patient_id'] = patient_id
    #         feature_table.append(features)

    # feature_table = pd.concat(feature_table, ignore_index=True)

    pdf.close()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--plot_filename',
                           required=True,
                           help='Plot filename')
    args = vars(argparser.parse_args())

    create_patient_portrait(args['plot_filename'])


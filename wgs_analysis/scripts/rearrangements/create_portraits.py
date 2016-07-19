import argparse
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import ith_project.study


def create_patient_portrait(plot_filename):
    study_info = ith_project.study.StudyInfo()

    feature_table = []

    for patient_id in study_info.sample_info['patient_id'].unique():
        breakpoint_results_filename = study_info.get_filename_template('breakpoint_results').format(patient_id=patient_id)

        with pd.HDFStore(breakpoint_results_filename, 'r') as breakpoint_results:
            features = breakpoint_results['/breakpoints/destruct/breakpoint']
            features['patient_id'] = patient_id
            feature_table.append(features)

    feature_table = pd.concat(feature_table, ignore_index=True)

    # Calculate features
    feature_table['log_num_reads'] = np.log10(feature_table['num_reads'])
    feature_table['log_num_split'] = np.log10(feature_table['num_split'])

    def classify_rearrangement_type(entry):
        break1 = entry['position_1']
        break2 = entry['position_2']
        size = abs(int(break1) - int(break2))
        orientation_type = entry['type']
        if size <= 1000000 and orientation_type == 'deletion':
            rearrangement_type = 'deletion'
        if size <= 1000 and orientation_type == 'inversion':
            rearrangement_type = 'foldback'
        elif size <= 1000000 and orientation_type == 'inversion':
            rearrangement_type = 'inversion'
        elif size <= 1000000 and orientation_type == 'duplication':
            rearrangement_type = 'duplication'
        else:
            rearrangement_type = 'unbalanced'
        return rearrangement_type

    def annotate_rearrangement_type(df):
        rearrangement_types = list()
        for idx, row in df.iterrows():
            rearrangement_types.append(classify_rearrangement_type(row))
        df['rearrangement_type'] = rearrangement_types

    annotate_rearrangement_type(feature_table)

    pdf = PdfPages(plot_filename)

    #
    # Number of unique breakpoints per patient
    #
    data = feature_table.groupby('patient_id').size().reset_index().rename(columns={0: 'num_breakpoints'})

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.barplot(ax=ax, x='patient_id', y='num_breakpoints', data=data)
    ax.set_title('Number of unique breakpoints per patient')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    #
    # Distribution of total read counts per breakpoint across patients
    #
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.boxplot(ax=ax, data=feature_table, x='patient_id', y='log_num_reads')
    ax.set_title('Distribution of total read counts per breakpoint across patients')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    #
    # Distribution of split read counts per breakpoint across patients
    #
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.boxplot(ax=ax, data=feature_table, x='patient_id', y='log_num_split')
    ax.set_title('Distribution of split read counts per breakpoint across patients')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    #
    # Distribution of breakpoint prediction sequence lengths across patients
    #
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.boxplot(ax=ax, data=feature_table, x='patient_id', y='template_length_min')
    ax.set_title('Distribution of breakpoint prediction sequence lengths across patients')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    #
    # Distribution of breakpoint homology lengths across patients
    #
    data = feature_table[feature_table['homology'] < 20]
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.boxplot(ax=ax, data=data, x='patient_id', y='homology')
    ax.set_title('Distribution of breakpoint homology lengths across patients')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    #
    # Proportion of each rearrangement type across patients
    #
    data = feature_table.groupby(['patient_id', 'rearrangement_type']).size().reset_index().rename(columns={0: 'count'})
    data = data.merge(feature_table.groupby('patient_id').size().reset_index().rename(columns={0: 'total'}))
    data['proportion'] = data['count'] / data['total']

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.barplot(ax=ax, x='patient_id', y='proportion', hue='rearrangement_type', data=data)
    ax.set_title('Proportion of each rearrangement type across patients')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    #
    # Proportion of each rearrangement type across patients, thresholded
    #
    filtered_data = feature_table[feature_table['num_reads'] >= 8]
    data = filtered_data.groupby(['patient_id', 'rearrangement_type']).size().reset_index().rename(columns={0: 'count'})
    data = data.merge(filtered_data.groupby('patient_id').size().reset_index().rename(columns={0: 'total'}))
    data['proportion'] = data['count'] / data['total']

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    seaborn.barplot(ax=ax, x='patient_id', y='proportion', hue='rearrangement_type', data=data)
    ax.set_title('Proportion of each rearrangement type across patients, thresholded at num_reads >= 8')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    pdf.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--plot_filename',
                           required=True,
                           help='Plot filename')
    args = vars(argparser.parse_args())

    create_patient_portrait(args['plot_filename'])


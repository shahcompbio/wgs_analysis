'''
# General plot functions

'''

import matplotlib.pyplot as plt

import wgs_analysis.helpers as helpers


def plot_patient_legend(legend_filename):
	fig = plt.figure(figsize=(0.75, 2))
	ax = fig.add_subplot(111)
	ax.axis('off')

	artists = [plt.Circle((0, 0), color=c) for c in helpers.patient_cmap.values()]
	ax.legend(artists, helpers.patient_cmap.keys(), loc='upper right', title='Patient')

	fig.savefig(legend_filename, bbox_inches='tight')


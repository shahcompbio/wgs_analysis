'''
# Clonal composition plots

'''

import matplotlib.pyplot as plt

import wgs_analysis.helpers as helpers


def plot_mixture_class_legend(legend_filename):
	def create_marker_artist(m):
	    return plt.Line2D(range(1), range(1), color='w', marker=m, markerfacecolor='k')

	fig = plt.figure(figsize=(0.75, 2))
	ax = fig.add_subplot(111)
	ax.axis('off')

	artists = [create_marker_artist(m) for m in helpers.phylogenetic_class_markers.values()]
	labels = [helpers.phylogenetic_class_labels[m] for m in helpers.phylogenetic_class_markers.keys()]
	ax.legend(artists, labels, loc='lower right', title='Mixture\nclassification')

	fig.savefig(legend_filename, bbox_inches='tight')


'''
# Circos plotting

Python interface for circos plots of breakpoints, copy number histograms, and SNVs as stack boxes.  

'''

import collections
import os
import glob
import subprocess
import pandas as pd
import numpy as np


def create_circos_config(chromosomes, breakpoints, copynumbers, snvs, config_filename, svg_filename):

    template_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 'circos.conf'))

    def circos_chromosome(chromosome):
        return 'hs' + chromosome

    links_by_color = collections.defaultdict(list)
    breakpoints = breakpoints.drop_duplicates(['cluster_id', 'cluster_end'])
    for cluster_id, rows in breakpoints.groupby('cluster_id'):
        row1 = rows.iloc[0]
        row2 = rows.iloc[1]
        color = 'black'
        try:
            color = row1['link_color']
        except:
            pass
        link = ((str(cluster_id), circos_chromosome(row1['chrom']), str(row1['coord']), str(row1['coord']+1)),
                (str(cluster_id), circos_chromosome(row2['chrom']), str(row2['coord']), str(row2['coord']+1)))
        links_by_color[color].append(link)

    link_text = ''
    z = 20
    index = 0
    for color, links in sorted(links_by_color.items()):
        link_filename = config_filename + '.{0}.circos.link'.format(color)
        with open(link_filename, 'w') as link_file:
            for link in links:
                link_file.write('\t'.join(link[0]) + '\n')
                link_file.write('\t'.join(link[1]) + '\n')
        link_text += '\t<link {0}>\n'.format(color)
        link_text += '\t\tz={0}\n'.format(z)
        link_text += '\t\tthickness=2\n'
        link_text += '\t\tcolor={0}\n'.format(color)
        link_text += '\t\tfile={0}\n'.format(link_filename)
        link_text += '\t</link>\n'
        z += 1
        index += 1

    output_text = ''
    output_text += '\tdir={0}\n'.format(os.path.abspath(os.path.dirname(svg_filename)))
    output_text += '\tfile={0}\n'.format(os.path.basename(svg_filename))

    copies_text = ''
    track = 0.0
    copy_track_width = 0.04
    for id, copynumber in copynumbers.iteritems():
        copynumber['chrom'] = copynumber.apply(lambda row: circos_chromosome(row['chrom']), axis=1)
        colors = ['red', 'blue']
        hist_datas = [copynumber[copynumber['median'] > 0.0], copynumber[copynumber['median'] < 0.0]]
        for color, hist_data in zip(colors, hist_datas):
            hist_filename = config_filename + '.{0}.circos.cnv.{1}'.format(id, color)
            hist_data.to_csv(hist_filename, sep='\t', header=False, index=False, cols=['chrom', 'start', 'end', 'median'])
            copies_text += '<plot>\n'
            copies_text += 'type = histogram\n'
            copies_text += 'file = {0}\n'.format(hist_filename)
            copies_text += 'r0 = {0}r\n'.format(1.0 - copy_track_width * (track + 1.0) - 0.01)
            copies_text += 'r1 = {0}r\n'.format(1.0 - copy_track_width * track - 0.01)
            copies_text += 'min=-0.5\n'
            copies_text += 'max=0.5\n'
            copies_text += 'color = {0}\n'.format(color)
            copies_text += 'fill_under = yes\n'
            copies_text += 'fill_color = {0}\n'.format(color)
            copies_text += 'thickness = 0\n'
            copies_text += 'extend_bin = no\n'
            copies_text += 'background = no\n'
            copies_text += '</plot>\n'
        track += 1.0

    snv_tiles_filename = config_filename + '.snv.tiles'
    with open(snv_tiles_filename, 'w') as snv_tiles_file:
        for idx, row in snvs.iterrows():
            snv_tiles_file.write('\t'.join([circos_chromosome(row['chrom']), str(row['pos']), str(row['pos'])]) + '\n')

    snvs_track_width = 0.2
    snvs_text = ''
    snvs_text += 'type = tile\n'
    snvs_text += 'layers_overflow = grow\n'
    snvs_text += '<plot>\n'
    snvs_text += 'file = {0}\n'.format(snv_tiles_filename)
    snvs_text += 'r1 = {0}r\n'.format(1.0 - copy_track_width * track - 0.01)
    snvs_text += 'r0 = {0}r\n'.format(1.0 - copy_track_width * track - 0.01 - snvs_track_width)
    snvs_text += 'orientation = in\n'
    snvs_text += 'layers = 15\n'
    snvs_text += 'margin = 0.02u\n'
    snvs_text += 'thickness = 4\n'
    snvs_text += 'padding = 2\n'
    snvs_text += 'stroke_thickness = 4\n'
    snvs_text += 'stroke_color = orange\n'
    snvs_text += 'color = blue\n'
    snvs_text += '</plot>\n'

    with open(template_filename, 'r') as template:
        config_text = template.read()

    config_text = config_text.replace('!insert_svg_here!', output_text)
    config_text = config_text.replace('!insert_links_here!', link_text)
    config_text = config_text.replace('!insert_links_radius_here!', str(1.0 - copy_track_width * track - 0.01))
    config_text = config_text.replace('!insert_chromosomes_default_here!', 'no')
    config_text = config_text.replace('!insert_chromosomes_here!', ' '.join([circos_chromosome(a) for a in chromosomes]))
    config_text = config_text.replace('!insert_copies_here!', copies_text)
    config_text = config_text.replace('!insert_snvs_here!', snvs_text)

    with open(config_filename, 'w') as config_file:
        config_file.write(config_text)



def create_circos_plot(chromosomes, breakpoints, copynumbers, snvs, plot_filename):

    config_filename = plot_filename + '.tmp.conf'
    create_circos_config(chromosomes, breakpoints, copynumbers, snvs, config_filename, plot_filename)
    circos_bin = os.path.expanduser('~/Build/circos-0.52/bin/circos')
    subprocess.check_call('perl {0} -conf {1}'.format(circos_bin, config_filename), shell=True)
    for filename in glob.glob(os.path.abspath(plot_filename) + '.tmp*'):
        os.remove(filename)


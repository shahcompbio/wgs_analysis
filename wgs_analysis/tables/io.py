import itertools
import pandas as pd
import numpy as np

def values_to_fixed_width(values, widths, fill=' ', sep='|'):
    text = []
    text.append(sep)
    for v, w in zip(values, widths):
        text.append(fill)
        text.append(v)
        text.append(fill * (w - len(v)))
        text.append(fill + sep)
    return ''.join(text)


def df_to_markdown(df):
    ''' Convert a DataFrame to a markdown table
    '''

    markdown_text = []

    df = df.applymap(lambda v: '{0}'.format(v))
    cols = ['`{0}`'.format(a) for a in df.columns]

    data_width = df.applymap(lambda v: len(v)).max(axis=0).values
    header_width = np.array([len(str(a)) for a in cols])

    cell_width = np.maximum(data_width, header_width)

    markdown_text.append(values_to_fixed_width(cols, cell_width))

    markdown_text.append(values_to_fixed_width(itertools.repeat(''), cell_width, fill='-'))

    for idx, row in df.iterrows():
        markdown_text.append(values_to_fixed_width(row.values, cell_width))

    markdown_text = '\n'.join(markdown_text)

    return markdown_text

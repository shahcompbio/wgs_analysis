import re
from collections import defaultdict
import pandas as pd
import numpy as np
from .. import refgenome

# classes
class Target:
    def __init__(self, chromosome, start, end):
        self.chromosome = chromosome
        self.start = start
        self.end = end
        self.chrom_short = chromosome.replace('chr', '')
    def __repr__(self):
        return f'{self.chromosome}({self.chrom_short}):{self.start}-{self.end}'


# modules
def get_repr_transcript_id(gtf, gene_name, lenient=False):
    """ Return a transcript_id with largest interval size
    - gtf: pygtf object
    - gene_name: gene symbol str
    """
    # gene_symbol = 'BCL2'
    if not lenient:
        transcripts = (
            gtf
            .query(f"gene_name == '{gene_name}'")
            .query("Feature == 'transcript'")
            .query("transcript_biotype == 'protein_coding'")
            .query("transcript_support_level == '1'")
        ).copy()
    else:
        transcripts = (
            gtf
            .query(f"gene_name == '{gene_name}'")
            .query("Feature == 'transcript'")
        ).copy()
    transcript_id = None
    if transcripts.shape[0] > 0:
        transcripts['length'] = transcripts['End'] - transcripts['Start']
        transcript = transcripts.sort_values(by=['length'], ascending=False).iloc[0]
        transcript_id = transcript['transcript_id']
    return transcript_id

def get_transcript_exons(gtf, transcript_id):
    exons = gtf[
        (gtf['transcript_id']==transcript_id) &
        (gtf['Feature'].isin(['exon', 'CDS']))
    ]
    return exons

# parse savana
class Breakpoint:
    def __init__(self, brk):
        self.chrom = brk['chrom']
        self.pos = brk['pos']
        self.self_id = brk['breakend'] # ID
        brk_ix = int(self.self_id[-1])
        assert brk_ix in {1, 2}, f'ERROR: brk_ix = {brk_ix}'
        self.ref = brk['ref']
        self.alt = brk['alt']
        self.info = brk['INFO']
        bp_notation = re.search('BP_NOTATION=([^;]+);', self.info).groups()[0]
        self.adjacency_id = brk['adjacency'] # ID
        self.mate_id = f'{self.adjacency_id}_{3-brk_ix}'
        
        _strand_combination = {'++', '--', '+-', '-+'}
        if bp_notation == '<INS>':
            self.strand = None
        elif bp_notation in _strand_combination:
            self.strand = bp_notation[brk_ix-1]
        else:
            raise ValueError(f'ERROR: bp_notation = {bp_notation}')

class Adjacency:
    def __init__(self, brks): # brks <- paired dataframe
        assert brks.shape[0] == 2, brks
        brk1, brk2 = brks.iloc[0], brks.iloc[1]
        self.brk1 = Breakpoint(brk1)
        self.brk2 = Breakpoint(brk2)
        self.type = 'n/a'
        
        self.type = self.get_svtype()
        self.length = abs(self.brk2.pos - self.brk1.pos)
        if self.type == 'translocation': 
            self.length = np.inf
    
    def get_svtype(self): # N-> <-N // <-N N-> // N<- N<- // ->N ->N
        if self.brk1.chrom != self.brk2.chrom: # - <TRA>
            return 'translocation'
        if (self.brk1.strand, self.brk2.strand) == ('+', '+'):
            return 'inversion'
        elif (self.brk1.strand, self.brk2.strand) == ('-', '-'):
            return 'inversion'
        elif (self.brk1.strand, self.brk2.strand) == ('+', '-'):
            return 'deletion'
        elif (self.brk1.strand, self.brk2.strand) == ('-', '+'):
            return 'duplication'
        else:
            raise ValueError(f'ERROR: (strand1, strand2) = ({self.brk1.strand}, {self.brk1.strand})')

def parse_vcf_breakpoints(vcf_path, refgenome=refgenome):
    vcf_cols = ['chrom', 'pos', 'breakend', 'ref', 'alt', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'sample']
    svs_cols = ['chromosome_1', 'position_1', 'strand_1', 
                'chromosome_2', 'position_2', 'strand_2', 'type', 'length']
    df = pd.read_table(vcf_path, comment='#', names=vcf_cols)
    chroms = refgenome.info.chromosomes

    svs = pd.DataFrame(columns=svs_cols) # savana sv

    df['adjacency'] = df['breakend'].str.rsplit('_', 1, expand=True)[0]
    
    svtype_cnt = defaultdict(int)
    for i, (_, brks) in enumerate(df.groupby('adjacency')):
        brks_in_adj = brks.shape[0]
        if brks_in_adj == 2:
            brk1, brk2 = brks.iloc[0], brks.iloc[1]
            brk1.chrom = brk1.chrom.replace('chr', '')
            brk2.chrom = brk2.chrom.replace('chr', '')
            if brk1.chrom not in chroms: continue
            if brk2.chrom not in chroms: continue
            if brk1.chrom == brk2.chrom:
                assert brk1.pos < brk2.pos, (brk1.pos, brk2.pos)
            adj = Adjacency(brks)
            svtype_cnt[adj.type] += 1
            line = [adj.brk1.chrom, adj.brk1.pos, adj.brk1.strand, adj.brk2.chrom, adj.brk2.pos, adj.brk2.strand, adj.type, adj.length]
        elif brks_in_adj == 1:
            brk = brks.squeeze()
            brk1 = Breakpoint(brk)
            assert brk['alt'] == '<INS>', brk
            svtype = 'insertion'
            match = re.search('INSSEQ=([A-Z]+);', brk['INFO'])
            insseq = match.groups()[0]
            svlength = len(insseq)
            svtype_cnt[svtype] += 1
            line = [brk1.chrom, brk1.pos, brk1.strand, brk1.chrom, brk1.pos, brk1.strand, svtype, svlength]
        svs.loc[i] = line

    return svs 


# parse WGS-BREAKPOINTCALLING
def parse_csv_breakpoints(csv_path):
    svs_cols = ['chromosome_1', 'position_1', 'strand_1', 'chromosome_2', 'position_2', 'strand_2',
                'type', 'length']
    df = pd.read_csv(csv_path, dtype={'chromosome_1':str, 'chromosome_2':str})
    for chrom_col in ['chromosome_1', 'chromosome_2']:
        df[chrom_col] = df[chrom_col].str.replace('chr', '')
    if 'length' not in df.columns:
        if 'break_distance' in df.columns:
            df.rename(columns={'break_distance': 'length'}, inplace=True)
    return df[svs_cols]


# parse WGS-REMIXT-POSTPROCESS
def _update_blocks_and_reset_prev(blocks, prev, row, 
        features=['chromosome', 'start', 'end', 'major_1', 'minor_1']):
    if prev['chromosome']: # not null
        blocks = pd.concat([blocks, prev.to_frame().T])
    prev = pd.Series({f: row[f] for f in features}, index=features)
    return blocks, prev


def merge_segments(remixt):
    """ Merge remixt segments
    - remixt: remixt-like DataFrame
    """
    #in_pp = '/juno/work/shah/isabl_data_lake/analyses/37/74/23774/SPECTRUM-OV-081_S1_LEFT_OVARY_cn.csv'
    # remixt = pd.read_table(in_pp, dtype=str, low_memory=False)

    features = ['chromosome', 'start', 'end', 'major_1', 'minor_1'] # only use major_1 and minor_1
    for feature in features:
        assert feature in remixt.columns, f'{feature} not in remixt.columns \n[remixt.columns: {remixt.columns}]'
    remixt = remixt[features] # reduce features for the sake of memory

    # swap cn if minor > major
    major_smaller_than_minor = (remixt['major_1'] < remixt['minor_1'])
    remixt.loc[ major_smaller_than_minor, 'major_1' ] = remixt.loc[ major_smaller_than_minor, 'minor_1' ]

    # merge segments with same major_1 and minor_1
    prev = pd.Series({f:None for f in features}, index=features) # init prev cn
    blocks = pd.DataFrame(columns=features) # init empty blocks

    for _, row in remixt.iterrows():
        if prev['chromosome'] != row['chromosome']:
            blocks, prev = _update_blocks_and_reset_prev(blocks, prev, row)
            continue
        if (prev['major_1'] != row['major_1']) or (prev['minor_1'] != row['minor_1']):
            blocks, prev = _update_blocks_and_reset_prev(blocks, prev, row)
            continue
        # if prev['end'] != row['start']:
        #     blocks, prev = _update_blocks_and_reset_prev(blocks, prev, row)
        #     continue
        # update prev block
        prev['end'] = row['end']

    blocks = pd.concat([blocks, prev.to_frame().T]) # add prev
    blocks.loc[:, ['start', 'end', 'major_1', 'minor_1']] = blocks.loc[:, ['start', 'end', 'major_1', 'minor_1']].astype(int)
    blocks['start'] = blocks['start'] + 1 # so segments don't overlap; w/o this !mixFlag[j] error occurs in MutationTimeR step
    return blocks
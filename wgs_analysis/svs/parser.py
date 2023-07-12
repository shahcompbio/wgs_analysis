

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


# Copy number analysis

## Plot cohort-level aggregated CN 
```python
gene_list_path = '/juno/work/shah/users/chois7/tickets/cohort-cn-qc/resources/gene_list.txt'
for cohorts in (['SPECTRUM'], ['Metacohort'], ['SPECTRUM', 'Metacohort']):
    cn = CopyNumberChangeData(gene_list=gene_list_path, cohorts=cohorts)
    cohort_symbol = '_'.join(cn.cohorts)
    for signature in cn.signature_counts:
        if cn.signature_counts[signature] > 5:
            cn.plot_pan_chrom_cn(group=signature, out_path=f'{cohort_symbol}.{signature}.pdf')
            cn.plot_per_chrom_cn(group=signature, out_path=f'{cohort_symbol}.{signature}.per-chrom.pdf')
```

## Fisher's exact test for CN enrichment in signatures
```python
gene_cn = cn.get_gene_cn_counts()
results = evaluate_enrichment(cn.signatures, cn.signature_counts, cn.gene_list, 
        cn.sample_counts, padj_cutoff=0.1)
results.to_csv('fet.tsv', sep='\t', index=False)
```

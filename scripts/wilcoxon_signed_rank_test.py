"""Paired Wilcoxon signed-rank tests for model-error columns, with Holm adjustment."""
import argparse
from itertools import combinations
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from common import columns, read_table

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--input',required=True); ap.add_argument('--error-columns',required=True); ap.add_argument('--output',default='wilcoxon_results.csv'); a=ap.parse_args()
    df=read_table(a.input); rows=[]
    for left,right in combinations(columns(a.error_columns),2):
        pair=df[[left,right]].dropna(); stat,p=wilcoxon(pair[left],pair[right],alternative='two-sided')
        rows.append({'model_a':left,'model_b':right,'statistic':stat,'p_value':p})
    result=pd.DataFrame(rows); result['p_adjusted_holm']=multipletests(result.p_value,method='holm')[1]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); result.to_csv(a.output,index=False)
if __name__=='__main__': main()

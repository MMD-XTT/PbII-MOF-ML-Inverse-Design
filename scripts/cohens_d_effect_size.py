"""Paired Cohen's d effect sizes for model-error columns."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from common import columns, read_table

def paired_d(a,b):
    difference=a.to_numpy()-b.to_numpy(); sd=difference.std(ddof=1)
    return np.nan if sd==0 else difference.mean()/sd

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--input',required=True); ap.add_argument('--error-columns',required=True); ap.add_argument('--output',default='cohens_d_results.csv'); a=ap.parse_args()
    df=read_table(a.input); rows=[]
    for left,right in combinations(columns(a.error_columns),2):
        pair=df[[left,right]].dropna(); rows.append({'model_a':left,'model_b':right,'cohens_d':paired_d(pair[left],pair[right])})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=='__main__': main()

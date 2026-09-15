"""Create a SHAP beeswarm plot for a fitted tree model."""
import argparse
from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd, shap
from common import columns, read_table

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--input',required=True); ap.add_argument('--model',required=True); ap.add_argument('--features',required=True); ap.add_argument('--output',default='shap_beeswarm_plot.png'); a=ap.parse_args()
    df=read_table(a.input); x=pd.get_dummies(df[columns(a.features)],dummy_na=True).fillna(0); values=shap.TreeExplainer(joblib.load(a.model)).shap_values(x)
    shap.summary_plot(values,x,show=False); plt.tight_layout(); Path(a.output).parent.mkdir(parents=True,exist_ok=True); plt.savefig(a.output,dpi=600,bbox_inches='tight')
if __name__=='__main__': main()

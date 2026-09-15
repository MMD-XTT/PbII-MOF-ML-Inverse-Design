"""Fit and save the six-locus random-forest TAGA fitness model."""
import argparse
from pathlib import Path
import joblib, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from common import columns, read_table

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--input',required=True); ap.add_argument('--target',required=True); ap.add_argument('--features',required=True,help='MN,LB1,FG1,LB2,FG2,topo'); ap.add_argument('--model-out',default='RF_TAGA_Fitness_Model.joblib'); a=ap.parse_args()
    df=read_table(a.input); x=pd.get_dummies(df[columns(a.features)],dummy_na=True).fillna(0); y=df[a.target]; model=RandomForestRegressor(n_estimators=1000,random_state=42,n_jobs=-1)
    pred=cross_val_predict(model,x,y,cv=KFold(5,shuffle=True,random_state=42),n_jobs=-1)
    print({'R2':r2_score(y,pred),'RMSE':mean_squared_error(y,pred,squared=False),'MAE':mean_absolute_error(y,pred),'SRCC':spearmanr(y,pred).statistic})
    model.fit(x,y); Path(a.model_out).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,a.model_out)
if __name__=='__main__': main()

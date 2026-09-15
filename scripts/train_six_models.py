"""Five-fold comparison of LR, SVM, RF, XGBoost, Extra Trees, and CatBoost."""
import argparse
from pathlib import Path
import joblib, pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from common import columns, read_table

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--input',required=True); ap.add_argument('--target',required=True); ap.add_argument('--features',required=True); ap.add_argument('--output-dir',default='outputs'); a=ap.parse_args()
    df=read_table(a.input); x=pd.get_dummies(df[columns(a.features)],dummy_na=True).fillna(0); y=df[a.target]; cv=KFold(5,shuffle=True,random_state=42)
    models={'LR':LinearRegression(),'SVM':make_pipeline(StandardScaler(),SVR(C=10,epsilon=.1)),'RF':RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1),'XGBoost':XGBRegressor(n_estimators=500,random_state=42,n_jobs=-1),'Extra_Trees':ExtraTreesRegressor(n_estimators=500,random_state=42,n_jobs=-1),'CatBoost':CatBoostRegressor(verbose=False,random_seed=42)}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    for name,model in models.items():
        pred=cross_val_predict(model,x,y,cv=cv)
        rows.append({'model':name,'R2':r2_score(y,pred),'RMSE':mean_squared_error(y,pred,squared=False),'MAE':mean_absolute_error(y,pred),'SRCC':spearmanr(y,pred).statistic})
        model.fit(x,y); joblib.dump(model,out/f'{name}_model.joblib')
    pd.DataFrame(rows).sort_values('R2',ascending=False).to_csv(out/'six_model_cv_summary.csv',index=False)
if __name__=='__main__': main()

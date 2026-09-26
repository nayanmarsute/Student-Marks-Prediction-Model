from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_collection import generate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT/'data', ROOT/'models', ROOT/'reports'
for p in [DATA_DIR, MODEL_DIR, REPORT_DIR]: p.mkdir(exist_ok=True)

FEATURES = ['study_hours','attendance_percentage','previous_exam_marks','assignment_marks','internal_marks','practice_test_score','sleep_hours','study_environment','internet_quality']
TARGET = 'final_exam_marks'
NUMERIC = FEATURES[:7]
CATEGORICAL = FEATURES[7:]


def clean_data(df: pd.DataFrame):
    work = df.copy()
    before = len(work)
    duplicate_count = int(work.duplicated().sum())
    work = work.drop_duplicates().copy()
    invalid_count = 0
    bounds = {
        'study_hours': (0, 16), 'attendance_percentage': (0,100),
        'previous_exam_marks': (0,100), 'assignment_marks': (0,100),
        'internal_marks': (0,100), 'practice_test_score': (0,100),
        'sleep_hours': (0,24), 'final_exam_marks': (0,100)
    }
    for col,(lo,hi) in bounds.items():
        mask = work[col].notna() & ~work[col].between(lo,hi)
        invalid_count += int(mask.sum())
        work.loc[mask,col] = np.nan
    # IQR clipping for predictor outliers; target is only domain-validated.
    outlier_counts={}
    for col in NUMERIC:
        q1,q3=work[col].quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        mask=work[col].notna() & ((work[col]<lo)|(work[col]>hi))
        outlier_counts[col]=int(mask.sum())
        work[col]=work[col].clip(lo,hi)
    missing_before=int(work[FEATURES+[TARGET]].isna().sum().sum())
    work=work.dropna(subset=[TARGET])
    return work, {'raw_rows':before,'duplicates_removed':duplicate_count,'invalid_values_found':invalid_count,'missing_cells_before_imputation':missing_before,'iqr_outliers_by_feature':outlier_counts,'clean_rows':len(work)}


def make_preprocessor():
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
    return ColumnTransformer([('numeric',num,NUMERIC),('categorical',cat,CATEGORICAL)])


def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train,y_train)
    pred=np.clip(model.predict(X_test),0,100)
    mse=mean_squared_error(y_test,pred)
    return {'MAE':float(mean_absolute_error(y_test,pred)),'MSE':float(mse),'RMSE':float(np.sqrt(mse)),'R2':float(r2_score(y_test,pred))},pred


def main():
    raw_path=DATA_DIR/'student_marks_raw.csv'
    generate_dataset()
    raw=pd.read_csv(raw_path)
    clean,quality=clean_data(raw)
    clean.to_csv(DATA_DIR/'student_marks_clean.csv',index=False)

    X,y=clean[FEATURES],clean[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42)

    linear=Pipeline([('preprocessor',make_preprocessor()),('feature_selection',SelectKBest(f_regression,k='all')),('model',LinearRegression())])
    ridge=Pipeline([('preprocessor',make_preprocessor()),('model',Ridge(alpha=1.0))])
    rf=Pipeline([('preprocessor',make_preprocessor()),('model',RandomForestRegressor(n_estimators=250,max_depth=8,random_state=42,n_jobs=-1))])
    gbr=Pipeline([('preprocessor',make_preprocessor()),('model',GradientBoostingRegressor(random_state=42,n_estimators=150,max_depth=2,learning_rate=.05))])

    models={'Linear Regression':linear,'Ridge Regression':ridge,'Random Forest':rf,'Gradient Boosting':gbr}
    results={}; predictions={}
    for name,model in models.items():
        metrics,pred=evaluate(model,X_train,y_train,X_test,y_test)
        cv=-cross_val_score(model,X_train,y_train,cv=5,scoring='neg_root_mean_squared_error',n_jobs=None).mean()
        metrics['CV_RMSE_5Fold']=float(cv)
        results[name]=metrics; predictions[name]=pred

    # Required model: Linear Regression. Persist it for the app.
    joblib.dump(linear,MODEL_DIR/'student_marks_pipeline.joblib')

    feature_names=linear.named_steps['preprocessor'].get_feature_names_out()
    selector=linear.named_steps['feature_selection']
    scores=selector.scores_
    feat_df=pd.DataFrame({'feature':feature_names,'f_score':scores,'selected':selector.get_support()}).sort_values('f_score',ascending=False)
    feat_df.to_csv(REPORT_DIR/'feature_selection.csv',index=False)

    pd.DataFrame(results).T.reset_index(names='Model').to_csv(REPORT_DIR/'model_comparison.csv',index=False)
    pred_df=pd.DataFrame({'actual':y_test.values,'predicted':predictions['Linear Regression']})
    pred_df.to_csv(REPORT_DIR/'test_predictions.csv',index=False)
    (REPORT_DIR/'metrics.json').write_text(json.dumps({'primary_model':'Linear Regression','linear_regression':results['Linear Regression'],'models':results,'data_quality':quality,'train_rows':len(X_train),'test_rows':len(X_test)},indent=2),encoding='utf-8')

    # Coefficients from the fitted LR model after preprocessing.
    coefs=linear.named_steps['model'].coef_
    selected_names=feature_names[selector.get_support()]
    coef_df=pd.DataFrame({'feature':selected_names,'coefficient':coefs}).sort_values('coefficient',key=lambda s:s.abs(),ascending=False)
    coef_df.to_csv(REPORT_DIR/'linear_coefficients.csv',index=False)

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8,5)); plt.scatter(y_test,predictions['Linear Regression'],alpha=.65); lo,hi=0,100; plt.plot([lo,hi],[lo,hi],'--'); plt.xlim(0,100); plt.ylim(0,100); plt.xlabel('Actual Final Marks'); plt.ylabel('Predicted Final Marks'); plt.title('Linear Regression — Actual vs Predicted'); plt.tight_layout(); plt.savefig(REPORT_DIR/'actual_vs_predicted.png',dpi=170); plt.close()

    comparison=pd.DataFrame(results).T
    plt.figure(figsize=(9,5)); plt.bar(comparison.index, comparison['RMSE']); plt.xticks(rotation=15); plt.ylabel('RMSE (lower is better)'); plt.xlabel(''); plt.title('Model Comparison — Test RMSE'); plt.tight_layout(); plt.savefig(REPORT_DIR/'model_comparison.png',dpi=170); plt.close()

    plt.figure(figsize=(9,5)); top=coef_df.head(8).sort_values('coefficient'); plt.barh(top['feature'], top['coefficient']); plt.axvline(0,ls='--'); plt.title('Linear Regression Feature Coefficients'); plt.tight_layout(); plt.savefig(REPORT_DIR/'feature_coefficients.png',dpi=170); plt.close()

    residuals=y_test.values-predictions['Linear Regression']
    plt.figure(figsize=(8,5)); plt.scatter(predictions['Linear Regression'],residuals,alpha=.65); plt.axhline(0,ls='--'); plt.xlabel('Predicted Marks'); plt.ylabel('Residual (Actual − Predicted)'); plt.title('Residual Analysis'); plt.tight_layout(); plt.savefig(REPORT_DIR/'residuals.png',dpi=170); plt.close()

    print(json.dumps({'primary_model':results['Linear Regression'],'models':results,'data_quality':quality},indent=2))

if __name__=='__main__': main()

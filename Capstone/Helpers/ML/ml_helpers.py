import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import warnings
import seaborn as sns
import statsmodels.api as sm

from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import OneHotEncoder, scale, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import plotly.graph_objects as go


def one_hot(df, cat_list):
  encoder = OneHotEncoder()
  df_encoded = encoder.fit_transform(df[cat_list]).toarray()
  
  cols = encoder.get_feature_names()
  encoded_df = pd.DataFrame(df_encoded, columns = cols, dtype = "int64")
  df.drop(cat_list, axis = 1, inplace = True)
  
  df_final = df.join(encoded_df, on = df.index)
  return df_final

def remove_outliers(df, threshold):
  z_score = np.abs(stats.zscore(df))
  df_rem_out = df[(z_score < threshold).all(axis=1)]
  df_rem_out.reset_index(inplace = True, drop = True)
  return df_rem_out

def remove_corr(df, threshold = 0.6):
  corr_matrix = df.corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
  to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
  to_drop = list(filter(lambda x: x != 'mv_log', to_drop))
  return df.drop(df[to_drop], axis = 1)

def prepare_data(df,drop_cols, cat_cols,output = "fee_log"):
  df_dropped = df.drop(drop_cols, axis = 1)
  df_dropped = one_hot(df_dropped, cat_cols)
  df_dropped = remove_outliers(df_dropped, 3)
  df_dropped = df_dropped.drop_duplicates(keep = False)
  y = df_dropped[output]
  X = df_dropped.drop(output,axis = 1)
  X = remove_corr(X)
  return (X,y)

def fsel_rforest(X_train, y_train, n):
  assert n < X_train.shape[1]
  rfr = RandomForestRegressor(max_depth = 18)
  rfr.fit(X_train, y_train)
  df = pd.DataFrame({'feature':X_train.columns,'importance':rfr.feature_importances_})
  df.sort_values(by = 'importance', inplace = True, ascending = False)
  return (df.iloc[0:n,0:1]["feature"].unique())

def significant_inputs(X,y):
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p = []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    return (model,cols)

def rmse(true, predictions, scaled_back = True):
    if(scaled_back):
      true = np.expm1(true)
      predictions = np.expm1(predictions)

    sq_dev = np.power(true - predictions, 2)
    mse = np.mean(sq_dev)
    
    return np.sqrt(mse)

def rsquared(y_test, y_pred, y_train, scaled_back = True):
    if(scaled_back):
      y_test = np.expm1(y_test)
      y_pred = np.expm1(y_pred)
      y_train = np.expm1(y_train)
    ssr = np.sum((y_test - y_pred)**2)
    sst = np.sum((y_test - np.mean(y_train))**2) 
    r2 = 1 - ssr/sst
    return r2


def model_summary(models, x_train, y_train, x_test, y_test):
    model_list = []
    r2_scorer = make_scorer(rsquared, y_train = y_train, scaled_back = True)
    rmse_scorer = make_scorer(rmse,scaled_back = True)
    for model in models:
        model.fit(x_train, y_train)
        score = model.score(x_train, y_train)
    
        cv_scores_num_rmse = cross_val_score(model, x_train,
                                        (y_train),
                                        cv = 5,
                                        scoring = rmse_scorer)
                
        cv_scores_r2 = cross_val_score(model, x_train,
                                       y_train, cv = 5,
                                       scoring = r2_scorer)
        
        cv_scores_r2_mean = np.mean(cv_scores_r2)
        
        results_dict = {
            "model": [type(model).__name__],
            "train_r2": [score],
            "test_r2": [rsquared((y_test), (model.predict(x_test)), (y_train))],
            "train_rmse": [int(rmse((model.predict(x_train)), (y_train)))],
            "test_rmse": [int(rmse((model.predict(x_test)), (y_test)))],
            "rmse_cv_mean": int(np.mean(cv_scores_num_rmse)),
            "r2_cv_mean": [cv_scores_r2_mean]
        }

        model_list.append(pd.DataFrame(results_dict))
    
    df = pd.concat(model_list, keys = range(len(models)), ignore_index = True)
    return df

def chart_regression(pred,y,title = "Player's actual and predicted prices",
                     scaled_back = False, y_axis = "Player's transfer price"):
  if(scaled_back):
    pred = np.expm1(pred)
    y = np.expm1(y)
  random_x = np.linspace(0, 1, len(pred))
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=random_x, y=pred,
                    mode='lines',
                    name='predicted'))
  fig.add_trace(go.Scatter(x=random_x, y=y,
                  mode='lines',
                  name='actual'))
  fig.update_layout(title= title,
                    yaxis_title=y_axis)
  fig.show()    


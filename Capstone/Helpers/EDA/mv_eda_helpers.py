import pandas as pd
import numpy as np
import math
from dfply import *
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statistics 
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import pearsonr
import plotly.graph_objects as go
import warnings
from scipy.stats import shapiro
from datetime import date, timedelta
from IPython.display import Image
from dateutil.relativedelta import *
import datetime as dt

def get_age(age_delta):
  result = math.nan
  try:
    result = int(age_delta / timedelta(days=365.2425))
  except:
    pass
  return result

def season_to_date(season):
    end = int(season.split("/")[1])
    if 59 <= end <= 99:
      date = '19' + str(end)
    else:
      date = '20' + str(end)
    return date

def add_months(date):
    result = date
    try:
       result = date + relativedelta(months=+5) + dt.timedelta(days=+29)
    except Exception as e:
        print(str(e))
        pass
    return result

def ready_for_eda(markval, players):
  players_tm = players.reset_index()
  markval = markval[markval.mv.notnull()]
  markval = markval[markval.tm_id.isin(players.tm_id.unique())]
  # markval['date'] = markval.index
  markval = markval.drop_duplicates()
  markval_merged = pd.merge(markval,
                  players_tm[['tm_id','name', 'dob', 'nationality', 'continent','field_position','main_field_position']],
                  on='tm_id')
  markval_merged["age"] = (markval_merged.date - markval_merged.dob)
  markval_merged.age = markval_merged.age.apply(get_age) 
  # markval_merged.set_index("date", inplace = True, drop = True)
  markval_merged["year"] = markval_merged.season.apply(season_to_date)
  return markval_merged

def markval_by_time(df, time_var, xaxis, title, group_var, summarizer = "mean", legend = True, facet = False):
  if(summarizer == "mean"):
    plot_df = ( df >> 
               group_by(X[time_var], X[group_var]) >>
               summarise(mv_summary = X.mv.mean())
               )
  elif(summarizer == "median"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(mv_summary = X.mv.median())
            )
  elif(summarizer == "max"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(mv_summary = X.mv.max())
            )
  elif(summarizer == "min"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(mv_summary = X.mv.min())
            )    
  elif(summarizer == "range"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(mv_summary = X.mv.max() - X.mv.min())
            )    

  if(facet):
    fig = px.line(plot_df, x = time_var, y = "mv_summary", 
                  color = group_var, title = title,
                  facet_col = group_var, facet_col_wrap=3,
                  labels = { time_var : xaxis, "mv_summary" : ""},
                  )
    fig.update_layout(
        showlegend = legend
    )
  else:
    fig = px.line(plot_df, x = time_var, y = "mv_summary", 
                  color = group_var, title = title,
                  labels = { time_var : xaxis, "mv_summary" : "Summarized market value"}
                  )
    fig.update_layout(
        showlegend = legend
    )
  fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
  fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))
  fig.show()


def previous_year_mv(markval):
  markval_1 = markval
  markval_1["date"] = markval.index
  markval_1 = markval_1.groupby('tm_id').apply(lambda x: x.sort_values('year', ascending = False))
  markval_1.reset_index(inplace=True, drop = True)
  markval_1["last_year_mv"] = markval_1.groupby('tm_id').pipe(lambda x: x.mv.shift(-1))
  markval_1 = markval_1[markval_1["last_year_mv"].notnull()]
  markval_1.set_index("date", inplace = True, drop = True)
  return markval_1


def scatter_mv_change(df,xvar,yvar,facet_var,xlabel,ylabel,title):
  fig = px.scatter(df, xvar, yvar, facet_col = facet_var, 
                 facet_col_wrap=5, hover_data = ["name","age","nationality","club"],
                 labels = {yvar: ylabel, xvar : xlabel},
                 title = title, trendline="ols")
  fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
  fig.show()
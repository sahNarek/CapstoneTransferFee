import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from dfply import *
import plotly.express as px
import statistics 
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import pearsonr
import plotly.graph_objects as go
from scipy.stats import shapiro
from datetime import date, timedelta

def scatter(df, xvar, yvar, labels, title, facet_var = "main_field_position",
            facet = False, facet_row = False, facet_row_var = "continent"):
  transfers_plot = df[(df[xvar].notnull()) & (df[yvar].notnull())]
  if(facet):
    fig = px.scatter(transfers_plot, xvar, yvar,hover_data = ["name","age","nationality","from",
                                                              "to","year", "fee","mv"],
                    labels = {xvar:"", yvar:""},
                    title = title, color = facet_var ,trendline = "ols",
                    facet_col = facet_var, facet_col_wrap = 2)
  elif(facet_row):
    fig = px.scatter(transfers_plot, xvar, yvar,hover_data = ["name","age","nationality","from",
                                                              "to","year", "fee","mv"],
                labels = {xvar:"", yvar:""},
                title = title, color = facet_var,trendline = "ols",
                facet_col = facet_var, facet_col_wrap = 2, facet_row = facet_row_var)
  else: 
    fig = px.scatter(transfers_plot, xvar, yvar,hover_data = ["name","age","nationality","from",
                                                              "to","year", "fee","mv"], 
                  labels = labels, trendline = "ols",
                  title = title, color = "main_field_position" )
    
  fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
  fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))
  corr, _ = pearsonr(transfers_plot[xvar],transfers_plot[yvar])
  print("Correlation between the variables",corr)
  fig.show()
  return fig

def cor_over_time(df, xvar, yvar, label, title):
  transfers_mv = df[(df[xvar].notnull()) & (df[yvar].notnull())]
  transfers_corr = transfers_mv.groupby('year')[[xvar,yvar]].corr()
  transfers_corr.reset_index(inplace = True, drop = False)
  transfers_corr = transfers_corr[transfers_corr.level_1 == xvar]
  transfers_corr.drop(["level_1",xvar], axis = 1, inplace = True)
  transfers_corr.columns = ["year", "corr_x_y"]
  transfers_corr.reset_index(inplace = True, drop = True)
  transfers_corr.dropna(inplace = True)
  fig = px.line(transfers_corr, x = "year", y = "corr_x_y" ,
                labels = {"corr_x_y": label},
                title = title)
  fig.update_xaxes(nticks=17)
  fig.show()
  return fig

def transfers_by_time(df, time_var, xaxis, title, group_var, summarizer = "mean",
                      legend = True, facet = False):
  if(summarizer == "mean"):
    plot_df = ( df >> 
               group_by(X[time_var], X[group_var]) >>
               summarise(fee_summary = X.fee.mean())
               )
  elif(summarizer == "median"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(fee_summary = X.fee.median())
            )
  elif(summarizer == "max"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(fee_summary = X.fee.max())
            )
  elif(summarizer == "min"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(fee_summary = X.fee.min())
            )    
  elif(summarizer == "range"):
    plot_df = ( df >> 
            group_by(X[time_var], X[group_var]) >>
            summarise(fee_summary = X.fee.max() - X.fee.min())
            )    
  if(facet):
    fig = px.line(plot_df, x = time_var, y = "fee_summary", 
                  color = group_var, title = title,
                  facet_col = group_var, facet_col_wrap=4,
                  labels = { time_var : xaxis, "fee_summary" : ""},
                  )
    fig.update_layout(
        showlegend = legend
    )
  else:
    fig = px.line(plot_df, x = time_var, y = "fee_summary", 
                  color = group_var, title = title,
                  labels = { time_var : xaxis, "fee_summary" : "Summarized transfer fee"}
                  )
    fig.update_layout(
        showlegend = legend
    )
  fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
  fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))
  fig.show()
  return fig  

def heat_map(df,title, diagonal = "both", w = 30,h = 20):
  df = df.select_dtypes(include=['float64'])
  corr = df.corr()
  plt.figure(figsize = (w, h))
  if(diagonal == "lower"):
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    ax = sns.heatmap(corr, annot = True, linewidths=.5, mask = mask)
  elif(diagonal == "upper"):
    mask = np.tril(np.ones_like(corr, dtype=np.bool))
    ax = sns.heatmap(corr, annot = True, linewidths=.5, mask = mask)
  else:
    ax = sns.heatmap(corr, annot = True, linewidths=.5)
  ax.tick_params(right=True, top=True, labelright=True, labeltop=True)
  plt.title(title, fontsize = 20)
  plt.show()
  return plt

def get_columns(df,cumulative):
  if(cumulative):
    cols = (df >> select(starts_with('cum'))).columns.tolist()
  else:
    cols = (df >> select(~starts_with('cum'))).columns.tolist()
  return cols


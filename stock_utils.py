import os, sys
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import talib as ta

class Indicators:
    def __init__(self):
        self.data = {'upper': [], 'overlay': [], 'lower': []}

    class Component: 
        def __init__(self, function, color, type, width, legend=None):
            self.function = function
            self.color = color
            self.type = type
            self.width = width
            self.legend = legend
            self.go = None
   
    def Add(self, name, placement, components):
        tmp = {}
        tmp['name'] = name
        tmp['enabled'] = True
        tmp['components'] = components
        self.data[placement].append(tmp)

class Stocks: 
    def __init__(self, periods=3650, path='./stocks.pkl'):
        self.dfs   = {}
        self.end   = datetime.today()
        self.start = self.end + timedelta(days=-periods) 
        self.path  = path
        self.dfs = {}
        self.ind = Indicators()
        self.ind.Add('RSI(14)', 'upper', 
                     [Indicators.Component('[70] * len(close)', color='grey',  type='line', width=0.5),
                      Indicators.Component('ta.RSI(close, timeperiod=14)', color='black', type='line', width=0.9, legend='RSI(14)'),
                      Indicators.Component('[30] * len(close)', color='grey',  type='line', width=0.5)])
        self.ind.Add('EMA(34)', 'overlay', 
                     [Indicators.Component('ta.EMA(close, 34)', color='red',  type='line', width=0.7, legend='EMA(34)')])
        self.ind.Add('EMA(89)', 'overlay', 
                     [Indicators.Component('ta.EMA(close, 89)', color='blue',  type='line', width=0.7, legend='EMA(89)')])
        self.ind.Add('EMA(233)', 'overlay', 
                     [Indicators.Component('ta.EMA(close, 233)', color='cyan',  type='line', width=0.7, legend='EMA(233)')])
        self.ind.Add('BB(20, 2, 0)', 'overlay', 
                     [Indicators.Component('ta.BBANDS(close, 20, 2)[0]', color='purple', type='line', width=0.5),
                      Indicators.Component('ta.BBANDS(close, 20, 2)[1]', color='purple', type='dot', width=0.5, legend='BB(20, 2, 0)'), 
                      Indicators.Component('ta.BBANDS(close, 20, 2)[2]', color='purple', type='line', width=0.5)])
        self.ind.Add('SAR(0.02, 0.2)', 'overlay', 
                     [Indicators.Component('ta.SAR(high, low, acceleration=0.02, maximum=0.2)', color='grey', type='mark', width=4, legend='SAR(0.02, 0.2)')])
        self.ind.Add('ADX(14)', 'lower',
                     [Indicators.Component('ta.ADX(high, low, close, timeperiod=14)', color='black', type='line', width=0.9, legend='ADX(14)'),
                      Indicators.Component('ta.MINUS_DI(high, low, close, timeperiod=14)', color='red', type='line', width=0.5, legend='DI-'),
                      Indicators.Component('ta.PLUS_DI(high, low, close, timeperiod=14)', color='green', type='line', width=0.5, legend='DI+')])
        self.ind.Add('MACD(12, 26, 9)', 'lower',
                     [Indicators.Component('ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0]', color='black', type='line', width=0.9, legend='MACD(12, 26, 9)'),
                      Indicators.Component('ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1]', color='red', type='line', width=0.9, legend='Signal'),
                      Indicators.Component('ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2]', color='grey', type='bar', width=0.4, legend='Histogram')])
        self.ind.Add('OBV', 'lower', 
                     [Indicators.Component('ta.OBV(close, volume)', color='black', type='line', width=0.9, legend='OBV')])
        self.ind.Add('WMR(14)', 'lower', 
                     [Indicators.Component('[-20] * len(close)', color='grey',  type='line', width=0.5),
                      Indicators.Component('ta.WILLR(high, low, close, timeperiod=14)', color='black', type='line', width=0.9, legend='WM%R(14)'),
                      Indicators.Component('[-80] * len(close)', color='grey',  type='line', width=0.5)])
                
    def GetData(self, ticker):
        df = pdr.get_data_yahoo(ticker, self.start, self.end).reset_index()
        df['Date'] = pd.to_datetime(df.Date, infer_datetime_format=True)
        df['is_up'] = df.Close > df.Open
        df['Color'] = df.is_up.apply(lambda x: 'rgb(158, 204, 183)' if x else 'rgb(255, 160, 154)')
        df = df.drop(columns=['is_up'])
        self.dfs[ticker]={}
        self.dfs[ticker]['daily'] = df
        self.dfs[ticker]['weekly']=self.Resample(ticker, 'w')
        self.dfs[ticker]['monthly']=self.Resample(ticker, 'm')
        self.ComputeIndicators(ticker)

    def Resample(self, ticker, period):
        df = self.dfs[ticker]['daily']
        df['Yr'] = df.Date.dt.year
        df['NN'] = df.Date.dt.isocalendar().week if period=='w' else df.Date.dt.month
        df = df.groupby(['Yr', 'NN']).agg(Date=('Date', 'first'), High=('High', 'max'), 
                                      Low=('Low', 'min'), Open=('Open', 'first'), 
                                      Close=('Close', 'last'), Volume=('Volume', 'sum'))\
                                      .reset_index()
        df['is_up'] = df.Close > df.Open
        df['Color'] = df.is_up.apply(lambda x: 'rgb(158, 204, 183)' if x else 'rgb(255, 160, 154)')
        df = df.drop(columns=['is_up'])
        self.dfs[ticker]['daily'] = self.dfs[ticker]['daily'].drop(columns=['Yr', 'NN'])
        return df
        
    def ComputeIndicators(self, ticker):
        for mode in ['daily', 'weekly', 'monthly']:
            open = self.dfs[ticker][mode]['Open']
            low = self.dfs[ticker][mode]['Low']
            high = self.dfs[ticker][mode]['High']
            close = self.dfs[ticker][mode]['Close']
            volume = self.dfs[ticker][mode]['Volume']
            for placement in self.ind.data:
                for idx, ind in enumerate(self.ind.data[placement]):
                    for idy, comp in enumerate(ind['components']):
                        self.dfs[ticker][mode][f'{ind["name"]}.{idy}'] = eval(comp.function)

                        
    def ChartIndicators(self, df):
        for placement in self.ind.data:
            for idx, ind in enumerate(self.ind.data[placement]):
                for idy, comp in enumerate(ind['components']):
                    ctitle = f'{ind["name"]}.{idy}'
                    cname = comp.legend if comp.legend is not None else ctitle
                    ctitle = f'{ind["name"]}.{idy}'
                    if comp.type =='line':
                        comp.go = go.Scatter(x=df.Date, y=df[ctitle], mode='lines', name=cname,
                                             line=dict(color=comp.color, width=comp.width))
                    elif comp.type=='dot':
                        comp.go = go.Scatter(x=df.Date, y=df[ctitle], mode='lines', name=cname,
                                             line=dict(color=comp.color, width=comp.width, dash='dot'))
                    elif comp.type=='mark':
                        comp.go = go.Scatter(x=df.Date, y=df[ctitle], mode='markers', name=cname,
                                             marker=dict(size=comp.width, color=comp.color))
                    elif comp.type=='bar':
                        comp.go = go.Bar(x=df.Date, y=df[ctitle], marker=dict(color=comp.color), name=cname)
                  
    def GetTitle(self, ticker, mode):
        df = self.dfs[ticker][mode].tail(1);    df1 = self.dfs[ticker][mode].shift(1).tail(1)
        open = df.Open.values[0];      high = df.High.values[0];       low = df.Low.values[0];
        close = df.Close.values[0];    close1 = df1.Close.values[0];   change = close-close1;
        volume = df.Volume.values[0];  last = df.Date.values[0];       changep = (close/close1-1)*100
        cl = 'green' if (change>=0) else 'red'
        ttxt  = f'{ticker} ({mode})<br><span style="font-size: 11px;">'
        ttxt += f'Open: &#36;{open:.2f}   High: &#36;{high:.2f}   '
        ttxt += f'Low : &#36;{low:.2f}   Prev. Close: &#36;{close1:.2f}       '
        ttxt += f'Volume: {volume:.2e}   [{pd.to_datetime(str(last)).strftime("%b %d")}]: '
        ttxt += f'<b><span style="color: {cl}">&#36;{close:.2f}</span></b>    Change: '
        ttxt += f'<b><span style="color: {cl}">&#36;{change:.2f}   ({changep:.1f}%)</span></b>'
        return ttxt
        
    def Plot(self, ticker, mode='daily', duration='1yr'):
        if duration ==   '5yr': start = self.end + timedelta(days=-365*5)
        elif duration == '3yr': start = self.end + timedelta(days=-365*3)
        elif duration == '1yr': start = self.end + timedelta(days=-365)
        elif duration == '6mo': start = self.end + timedelta(days=-180)
        elif duration == '3mo': start = self.end + timedelta(days=-90)
        elif duration == 'ytd': start = datetime(self.end.year, 1, 1)
        else                  : start = self.start
        
        df = self.dfs[ticker][mode]
        df = df[df.Date >= start]
        
        price  = go.Candlestick(x=df.Date, open=df.Open, high=df.High, low=df.Low, close=df.Close, 
                                line=dict(width=.5), name=ticker)
        volume = go.Bar(x=df.Date, y=df.Volume, marker=dict(color=df.Color), name='Volume')
        
        rows = [0.15]*len(self.ind.data['upper']) + [0.6] + [0.15]*(1+len(self.ind.data['lower']))
        fig = make_subplots(rows=len(rows), row_heights=rows, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.02)
        fig.add_trace(price, row=2, col=1)
        fig.add_trace(volume, row=3, col=1)

        self.ChartIndicators(df)
        for placement in self.ind.data:
            if placement == 'upper':    
                ridx_offset=1
            elif placement == 'lower':
                ridx_offset = len(self.ind.data['upper'])+3    # make room for chart + volume
            else:
                ridx_offset = len(self.ind.data['upper'])+1    
                subtitle = f'<span style="font-size: 10px;">'
                
            for idx, ind in enumerate(self.ind.data[placement]):
                if placement!='overlay':
                    subtitle = f'<span style="font-size: 10px;">'
                ridx = ridx_offset if placement=='overlay' else ridx_offset+idx
                for idy, comp in enumerate(ind['components']):
                    fig.add_trace(comp.go, row=ridx, col=1)
                    ctitle = f'{ind["name"]}.{idy}'
                    if comp.legend is not None:  
                        last = df[ctitle].tail(1).values[0]
                        last = f'{last:.2e}' if last>1e6 else f'{last:.2f}'
                        subtitle += f'<span style="color: {comp.color}">{comp.legend}: {last}</span>&nbsp;&nbsp;&nbsp;'
                if placement!='overlay':
                    fig.add_annotation(xref='x domain', yref='y domain', x=0, y=1, text=subtitle, 
                                       showarrow=False, row=ridx, col=1)
            if placement=='overlay':
                fig.add_annotation(xref='x domain', yref='y domain', x=0, y=1, text=subtitle, 
                                   showarrow=False, row=ridx_offset, col=1)
            
        fig.update_layout()
        fig.update_layout(title=dict(text=self.GetTitle(ticker, mode), xanchor='left', x=0.085), hovermode="x unified",
                         width=950, height=1100, template='plotly_white', showlegend=False, bargap=0.001,
                         xaxis_range = [df.Date.min()+timedelta(days=-1), df.Date.max()+timedelta(days=5)])
        fig.update_xaxes(rangebreaks=[dict(values=pd.date_range(start=df.Date.min(), end=df.Date.max()).difference(df.Date))], 
                         rangeslider_visible=False)
        fig.update_yaxes(type='log', row=len(self.ind.data['upper'])+1, col=1, side='right')
        return fig
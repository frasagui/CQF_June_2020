# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 09:41:55 2020

@author: xiaon
"""

#import relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import statsmodels.api as sm
import scipy
from scipy.optimize import minimize


# Financial Data

def load_data(filename='faang_stocks.csv'):
    '''
    Load csv file.
    Option to modify file name.'''
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df


def data_summary(df=load_data()):
    '''Print a summary of data including count, mean, stdev, percentiles, and max.'''   
    summary = df.describe().T
    return summary


def download_endofday_data_yf(stock_list=['AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX'], start_date= '2013-06-01', end_date= '2020-06-30', save_csv=False, file_name='faang_stocks.csv'):
    '''
    Retrieving end-of-day data from yahoo finance.
    Options to modify the stocks list, start and end dates, and save the data in csv format. '''
    df = yf.download(stock_list, start=start_date, end=end_date, progress=False)['Adj Close']
    if save_csv==True:
        df.to_csv(file_name) 
    return df


def download_ohlc_data_yf(stock_list=['AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX'], start_date= '2013-06-01', end_date= '2020-06-30',save_csv=False):
    '''
    Retrieving end-of-day data from yahoo finance with ohlc.
    Options to modify the stocks list, start and end dates, and save the data of individual stock in different csv files.
    Return the results in a dictionary.'''
    ohlc_data = {symbol: yf.download(symbol, start=start_date, end=end_date, progress=False) for symbol in stock_list}
    if save_csv==True:
        [pd.DataFrame(ohlc_data[symbol]).to_csv(symbol+'.csv') for symbol in stock_list]  
    return ohlc_data


def download_intraday_data_yf(tickers="SPY", period="5d", interval="1m",save_csv=False, file_name='SPY.csv'):
    '''
    Retrieving intraday data from yahoo finance. 
    Options to modify the stocks, period, interval, and save the data in csv format.'''
    dfi = yf.download(tickers=tickers, period=period, interval=interval, progress=False)
    if save_csv==True:
        dfi.to_csv(file_name)
    return dfi


def download_intraday_data_av(key_path = "Key.txt",stock='AMZN',interval='1min'):
    '''
    Retrieving intraday data from alpha vantage. 
    Options to modify the stocks, period, interval, and save the data in csv format.
    Download the key in advance and save as 'Key.csv'. '''
    ts = TimeSeries(key=open(key_path, 'r').read(),output_format='pandas')
    data, metadata  = ts.get_intraday(symbol=stock,interval=interval, outputsize='full')
    return data


#______________________________________________________________________________________________#
# Plot QQ plot
    

def hist_plot(ret=(load_data('M1L1.csv'))['Adj Close'].pct_change().dropna(0),bins = 200):
    '''hist_plot is applied to visulise the sample distribution'''
    ret.hist(bins = bins, figsize = (12 ,6), color='pink')
    plt.title('SPX Daily Returns')
    

def qq_plot(ret=(load_data('M1L1.csv'))['Adj Close'].pct_change().dropna(0)):
    '''qq_plot is applied to check whether the distribution is normal or not. '''
    sm.qqplot(ret.dropna(), line='s')
    plt.grid(True)
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Sample quantiles')
    plt.title('Normal-QQ plot, SPX returns')
    

def is_normal(ret=(load_data('M1L1.csv'))['Adj Close'].pct_change().dropna(0), level=0.01):
    """
    Extention from edhec_risk_kit
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(ret, pd.DataFrame):
        return ret.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(ret)
        return p_value > level
    

#______________________________________________________________________________________________#
# Binomial option pricing

        
def binomial_option(spot, strike, rate, sigma, time, steps, call_option=True, US_option=False, output=0):
    '''Function to price European/ American option based on binomial model. 
    Parameters
    --------------
    spot              int or float       - spot price
    strike            int or float       - strike price
    rate              float              - discount interest rate
    sigma             float              - volatility 
    time              int or float       - expiration time with unit of year
    steps             int                - number of time steps
    call_option       bool               - True: call option, False: call option
    US_option         bool               - True: American option, False: European option
    output            int                - 0: px, price path, 1: payoff, 2: V, option value, 3: d, delta
    '''
    # define option_flag
    if call_option==True:
        option_flag=1
    else: 
        option_flag=-1
    # define US_flag
    if US_option==True:
        US_flag=1
    else:
        US_flag=0    
    # define other intermediate parameters
    time_step=time/steps
    u=1+sigma*np.sqrt(time_step)
    v=1-sigma*np.sqrt(time_step)
    p=1/2+rate*np.sqrt(time_step)/(2*sigma)
    df=1/(1+rate*time_step)
    # initial arrays
    px=np.zeros((steps+1,steps+1))         #price path
    payoff=np.zeros((steps+1,steps+1))     #payoff
    d=np.zeros((steps+1,steps+1))          #hedging delta
    V=np.zeros((steps+1,steps+1))          #option value
    # calculate the price path:px and the payoff of option:payoff 
    for j in range(steps+1):
        for i in range(j+1):
            px[i,j]=spot*np.power(u,j-i)*np.power(v,i)
            payoff[i,j]=np.maximum(option_flag*(px[i,j]-strike),0)       
    # calculate the hedging delta:d and the option value:V
    for j in range(steps+1,0,-1):
        for i in range(j):
            if (j==steps+1):
                d[i,j-1]=0
                V[i,j-1]=payoff[i,j-1]
            else:
                d[i,j-1]=(V[i,j]-V[i+1,j])/(px[i,j]-px[i+1,j])
                V[i,j-1]=np.maximum(df*(p*V[i,j]+(1-p)*V[i+1,j]),US_flag*payoff[i,j-1])
    # combine results
    results = np.around(px,2), np.around(payoff,2), np.around(V,2), np.around(d,4)
    return results[output]


#______________________________________________________________________________________________#
# Portfolio optimisation
    
def d_ret(df=load_data('faang_stocks.csv')):
    '''
    Load the stock Adj Close price and calculate the daily return.'''
    return df.pct_change().dropna(0)


'''defaultly set return equals to faang_stocks' daily arithmetic return
   defaultly set n_asset as the number of assets in faang_stocks
   defaultly set r_free (risk free rate) as 0
   defaultly set n_port (number of monte carlo pathways) as 5000'''
   
   
ret = d_ret()
n_asset = ret.shape[1]
r_free = 0
n_port = 5000

def d_vol(ret):
    '''
    Stdev of daily returns.'''
    return ret.std()


def d_cov(ret):
    '''
    Covariance of daily returns.'''
    return ret.cov()


def ann_ret(ret, periods_per_year=252): 
    """
    Annualizes a set of returns"""
    return ret.mean() * 252


def ann_vol(ret, periods_per_year=252):
    """
    Annualizes the volatility of a set of returns"""
    return ret.std()*(periods_per_year**0.5)


def ann_cov(ret, periods_per_year=252):
    """
    Annualizes the covariance of a set of returns"""
    return ret.cov()*(periods_per_year)


def equal_wts(ret):
    '''
    Generate an array with equal weights'''
    n_asset=ret.shape[1]
    value = 1./n_asset
    wts = np.full((n_asset,1),value)
    return wts


def random_wts(ret):
    '''
    Generate an array with normalised random weights'''
    n_asset = ret.shape[1]
    wts = np.random.random_sample((n_asset,1))
    wts /= sum(wts)
    return wts


def port_ret(wts):
    '''
    Calculate portfolio returns based on individual asset's weights and daily returns.'''
    return wts.T @ np.array(ret.mean() * 252)[:,np.newaxis] 


def port_vol(wts):
    '''
    Calculate portfolio volatility based on individual asset's weights and daily returns.'''
    return np.array((wts.T@ret.cov()*252@wts)**0.5)


def port_sr(wts):
    return (port_ret(wts)-r_free)/port_vol(wts)


def port_stats(wts):
    '''
    Generate portfolio statistics based on individual asset's weights and daily returns.
    Return the result: 0: portfolio return, 1: portfolio volatility, 2: portfolio sharpe ratio'''
    port_rets = wts.T @ np.array(ret.mean() * 252)[:,np.newaxis]    
    port_vols = np.sqrt(np.linalg.multi_dot([wts.T, ret.cov() * 252, wts])) 
    return np.array([port_rets, np.array([port_vols]), (port_rets-r_free)/port_vols]).flatten()


def port_sim(n_port,r_free=0):
    '''
    Generate portfolio simulation based on individual asset's return and number of portfolio.
    Return the result: 0: portfolio return, 1: portfolio volatility, 2: the random portfolio weights generated
    '''
    n_asset = ret.shape[1]
    # Initialize the lists
    rets = []
    vols = []
    wts = []

    # Simulate n_port portfolios
    for i in range (n_port):
    
        # Generate random weights
        weights = np.random.random(n_asset)[:, np.newaxis]
        weights /= sum(weights)
        
        port_ret = weights.T @ np.array(ret.mean() * 252)[:, np.newaxis]
        port_vol = np.sqrt(np.linalg.multi_dot([weights.T, ret.cov()*252, weights]))
    
        # Portfolio statistics
        rets.append(port_ret)        
        vols.append(port_vol)
        wts.append(weights.flatten())

        # Record values     
        port_rets = np.array(rets).flatten()
        port_vols = np.array(vols).flatten()
        port_wts = np.array(wts)
        
    return pd.DataFrame({'returns': port_rets,
                      'volatility': port_vols,
                      'sharpe_ratio': (port_rets-r_free)/port_vols,
                      'weights': list(port_wts)})
    
    
def msr_sim(n_port,r_free=0):
    '''
    Generate random weights arrays and find the maximum Sharpe ratio within the simulation. 
    Return the results in portfolio returns, volatility, Sharpe ratio, and weights for maximum Sharpe ration.
    Function is sensitive to n_port.'''
    
    data = port_sim(n_port,r_free)
    msrp = data.iloc[data['sharpe_ratio'].idxmax()]
    return msrp


def port_sim_vis(n_port, r_free=0):
    '''
    Generate random weights arrays and plot the portfolio volatility on x-axis and the portfolio return on y-axis.
    The portfolio with maximum Sharpe ratio is marked.
    Function is sensitive to n_port.'''
    
    port = port_sim(n_port,r_free)
    port_rets = port['returns']
    port_vols = port['volatility']
    port_sr = port['sharpe_ratio']
    
    msrp = msr_sim(n_port,r_free)
    
    # Visualize the simulated portfolio for risk and return
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()

    ax.set_title('Monte Carlo Simulated Allocation')

    # Simulated portfolios
    fig.colorbar(ax.scatter(port_vols, port_rets, c=port_sr, 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

    # Maximum sharpe ratio portfolio
    ax.scatter(msrp['volatility'], msrp['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')

    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.grid(True)
    
    
def msr_opt(r_free=0):
    '''Find the maximum Sharpe ratio through optimization.
    Return the results: [0:p_stats][0: the portfolio return, 1: the portfolio volatility, 3: Sharpe ratio], [1:p_wts]
    '''
    n_asset = ret.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(n_asset))
    initial_wts = n_asset*[1./n_asset]
    def neg_sr(wts):
        return -port_sr(wts)
    wts = minimize(neg_sr, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
    p_stats = np.around(port_stats(wts['x']),4)
    p_wts = np.around(wts['x'],4)
    return [p_stats, p_wts]


def mv_opt(r_free=0):
    '''Find the minimum portfolio volatility through optimization.
    Return the results in the portfolio return, volatility, and Sharpe ratio.
    '''
    n_asset = ret.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for x in range(n_asset))
    initial_wts = n_asset*[1./n_asset]
    
    def min_variance(wts):
        return port_vol(wts)
    wts = minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
    p_stats = np.around(port_stats(wts['x']),4)
    p_wts = np.around(wts['x'],4)
    return [p_stats, p_wts]


def ef_sim_vis(r_free=0,show_msr=False,show_mv=False):
    
    # Minimize the volatility
    def min_volatility(wts):
        return port_vol(wts)
    n_asset = ret.shape[1]
    targetrets = np.linspace(0.20,0.45,100)
    tvols = []
    for tr in targetrets:
        initial_wts = n_asset*[1./n_asset]
        bnds = tuple((0, 1) for x in range(n_asset))
        ef_cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tr},
               {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        opt_ef = minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
        tvols.append(opt_ef['fun'])
    targetvols = np.array(tvols)
    # Visualize the simulated portfolio for risk and return
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes()
    ax.set_title('Efficient Frontier Portfolio')
    # Efficient Frontier
    fig.colorbar(ax.scatter(targetvols, targetrets, c=(targetrets-r_free) / targetvols, 
                        marker='x', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 
    
    # Maximum Sharpe Portfolio
    if show_msr:
        p_msr = msr_opt(r_free)
        ax.plot(p_msr[0][1], p_msr[0][0], 'r*', markersize =15.0)
        
    # Minimum Variance Portfolio
    if show_mv:
        p_mv = mv_opt(r_free)
        ax.plot(p_mv[0][1], p_mv[0][0], 'b*', markersize =15.0)
    
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.grid(True)
    

#______________________________________________________________________________________________#
# VaR
    

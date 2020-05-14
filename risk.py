from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

print('What is the start date? ie 2016-01-01')
start_date = input()
#end_date = str(datetime.now().strftime('%Y-%m-%d'))
print('What is the end date? ie 2020-01-01')
end_date = input()
print('What is the ticker? You can look it up on yahoo finance. Tips: Its case sensitive and make sure you use the exact same symbol')
ticker = input()
ticker = ticker.upper()

# Getting the data set

stock_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
stock_data['Returns'] = stock_data['Adj Close'].pct_change()


risk = stock_data.copy()
risk.index = risk.index.astype(str)

StockReturns_perc = risk[['Returns']].dropna().copy()
StockReturns_perc = StockReturns_perc*100

# Historical VaR(90) quantiles
var_90 = np.percentile(StockReturns_perc, 10)

# Historical expected shortfall CVaR(90) quantiles
cvar_90 = StockReturns_perc[StockReturns_perc <= var_90].mean()

# Historical VaR(95) quantiles
var_95 = np.percentile(StockReturns_perc, 5)
                                                                                    
# Historical expected shortfall CVaR(95) quantiles
cvar_95 = StockReturns_perc[StockReturns_perc <= var_95].mean()

# Historical VaR(99) quantiles
var_99 = np.percentile(StockReturns_perc, 1)

# Historical expected shortfall CVaR(99) quantiles
cvar_99 = StockReturns_perc[StockReturns_perc <= var_99].mean()


def plot_daily_return():
    ## Plot daily returns over time
    plt.figure(figsize=(12,12))
    stock_data['Returns'].plot()
    plt.title("Returns for {} from {} to {}".format(ticker, start_date, end_date))
    plt.ylabel('Returns')
    plt.show()


def plot_return_distribution():
    ## Return distributions
    plt.figure(figsize=(12,12))
    percent_return = stock_data['Returns']*100

    plt.hist(percent_return.dropna(), bins=100)
    plt.title("Distribution for {} returns from {} to {}".format(ticker, start_date, end_date))
    plt.ylabel('Count')
    plt.show()

def avgReturn():
    # Calculate the average daily return of the stock
    mean_return_daily = np.mean(stock_data['Returns'])
    print('Start Date : {}'.format(start_date))
    print('End Date : {}'.format(end_date))

    print('{} Average Daily Return: '.format(ticker) + str(round(mean_return_daily*100,2)) + '%')
    # Calculate the implied annualized average return
    mean_return_annualized = ((1+mean_return_daily)**252)-1
    print('{} Average Annualized Return: '.format(ticker) + str(round(mean_return_annualized*100,2))+ '%')


def avgVolatility():
    # Calculate the standard deviation of daily return of the stock
    sigma_daily = np.std(stock_data['Returns'])
    print('The average daily volatility of {}(std) is : '.format(ticker) + str(round(sigma_daily*100,2)) + '%')

    # Calculate the daily variance
    variance_daily = sigma_daily**2
    print('Variance of {}: '.format(ticker)+ str(round(variance_daily*100,4)) + '%')

    # Annualize the standard deviation
    sigma_annualized = sigma_daily*np.sqrt(252)
    print('Annualized Volatility of {} is '.format(ticker) + str(round(sigma_annualized*100,0)) + '%')

    # Calculate the annualized variance
    variance_annualized = sigma_annualized**2
    print('Annualized Variance is {} is '.format(ticker) + str(round(variance_annualized*100,0)) + '%')
    print("Psst...A volatile stock has a high standard deviation, while the deviation of a stable blue-chip stock is usually rather low.")

def plot_var_scale():   
    # Plot the forecased vs time    
    plt.plot(forecasted_values[:,0], -1*forecasted_values[:,1])    
    plt.xlabel('Time Horizon T+i')    
    plt.ylabel('Forecasted VaR 95 (%)')    
    plt.title('VaR 95 Scaled by Time for {}'.format(ticker), fontsize=18, fontweight='bold')    
    plt.show()

# Aggregate forecasted VaR
forecasted_values = np.empty([100, 2])

    
def valueAtRisk():
    plt.figure(figsize=(12,12))
    plot_hist()
    print('VAR 90')
    print('With 90% confidence we expect that our worst daily loss will not exceed {0:.2f}%'.format(var_90))
    #print('The minimum loss this asset has sustained in the worst 10% is {0:.2f}%'.format(var_90)) 
    print('In the worst 10% of cases, losses were on average exceed {0:.2f}% historically'.format(cvar_90[0]))
    print('-----------------------------------')
    print('VAR 95')
    print('-----------------------------------')
    print('With 95% confidence we expect that our worst daily loss will not exceed {0:.2f}%'.format(var_95))
    #print('The minimum loss this asset has sustained in the worst 5% is {0:.2f}%'.format(var_95)) 
    print('In the worst 5% of cases, losses were on average exceed {0:.2f}% historically'.format(cvar_95[0]))
    print('VAR 99')
    print('-----------------------------------')
    print('With 99% confidence we expect that our worst daily loss will not exceed {0:.2f}%'.format(var_99))
    #print('The minimum loss this asset has sustained in the worst 1% is {0:.2f}%'.format(var_99))
    print('In the worst 1% of cases, losses were on average exceed {0:.2f}% historically'.format(cvar_99[0]))


    # Loop through each forecast period
    for i in range(100):
        # Save the time horizon i
        forecasted_values[i, 0] = i
        # Save the forecasted VaR 95
        forecasted_values[i, 1] = var_95*np.sqrt(i+1)
    
    # Plot var over time 
    plot_var_scale()
    print('with 95% confidence, we will not lose more than {0:.3f}% in any given month.'.format(var_95*np.sqrt(20)) )

def plot_hist():    
    plt.hist(StockReturns_perc['Returns'],bins=100, density=True)    
    # Charting parameters    
    plt.xlabel('Returns (%)')   
    plt.ylabel('Probability')
    plt.title('Historical Distribution of {} Returns'.format(ticker), fontsize=18, fontweight='bold')    
    plt.axvline(x=var_90, color='r', linestyle='-', label="VaR 90: {0:.2f}%".format(var_90))
    plt.axvline(x=var_95, color='g', linestyle='-', label="VaR 95: {0:.2f}%".format(var_95))
    plt.axvline(x=var_99, color='b', linestyle='-', label="VaR 99: {0:.2f}%".format(var_99))
    plt.axvline(x=cvar_90[0], color='r', linestyle='--', label="CVaR 90: {0:.2f}%".format(cvar_90[0]))
    plt.axvline(x=cvar_95[0], color='g', linestyle='--', label="CVaR 95: {0:.2f}%".format(cvar_95[0]))
    plt.axvline(x=cvar_99[0], color='b', linestyle='--', label="CVaR 99: {0:.2f}%".format(cvar_99[0]))
    plt.legend(loc='upper right')   
    plt.show()

def simulateRisk(tradingDays = 252,startingPrice = 1, numSim=500):
    """
    Based the mean and volatility of the stock, simulate how will the share price look like in a random walk / monte carlo
    252 is the average trading days per year
    """

    risk = stock_data.copy()
    risk.index = risk.index.astype(str)
    StockReturns = risk[['Returns']].dropna().copy()

    # average daily return
    mu = np.mean(StockReturns)

    # Estimate the daily volatility
    vol = np.std(StockReturns)

    # Set the VaR confidence level
    confidence_level = 0.05

    # Calculate Parametric VaR (Value at risk for a single day)
    var_95 = norm.ppf(confidence_level, mu, vol)
    print('Average Daily return: {0:.3f}%'.format(mu[0]*100),
        '\nAverage Daily Volatility: {0:.3f}%'.format(vol[0]*100),
        '\nThe minimum loss this asset has sustained in the worst 5% is {0:.3f}%'.format(var_95[0]*100))
    print("Number of trading days :{}".format(str(tradingDays)))
    print("Starting Price for simulation : {}".format(startingPrice))
    print("Number of times simulated :{}".format(str(numSim)))

    # Loop through the number of simulations
    for i in range(numSim):

        # Generate the random returns
        rand_rets = np.random.normal(mu, vol, tradingDays) + 1
        
        # Create the Monte carlo path
        forecasted_values = startingPrice*(rand_rets).cumprod()
        
        # Plot the Monte Carlo path
        plt.plot(range(tradingDays), forecasted_values)

    # Show the simulations
    plt.title('Simulation of stock price for {} over {} trading days based on mean & volatility for {} times'.format(ticker, str(tradingDays),str(numSim)))
    plt.show()

def annualised_sharpe(returns, N=252):
    """
    Calculate the annualised Sharpe ratio of a returns stream 
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily returns.

    The function assumes that the returns are the excess of 
    those compared to a benchmark.
    """
    return np.sqrt(N) * returns.mean() / returns.std()

def equity_sharpe(risk_free_rate=0.05):
    """
    Calculates the annualised Sharpe ratio based on the daily
    returns of an equity
    The ratio compares the mean average of the excess returns of the asset or strategy with the standard deviation of those returns. Thus a lower volatility of returns will lead to a greater Sharpe ratio, assuming identical returns.
    The "Sharpe Ratio" often quoted by those carrying out trading strategies is the annualised Sharpe, the calculation of which depends upon the trading period of which the returns are measured.
    Note that the Sharpe ratio itself MUST be calculated based on the Sharpe of that particular time period type
    """
    sharp_df = risk[['Returns']].dropna().copy()

    # assumming risk free rate of 5%
    sharp_df['excess_daily_ret'] = sharp_df['Returns'] - (risk_free_rate/252)

    print(annualised_sharpe(sharp_df['excess_daily_ret']))


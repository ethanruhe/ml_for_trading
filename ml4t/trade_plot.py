"""
Ethan Ruhe
"""
import matplotlib.pyplot as plt


def plot_rm(df, var='close', n=10, bollinger=True):
    """Plot rolling mean
        Assumes data formatted by TradeData().clean_df_s()
        Prints plot.

    Args: 
        df: TradeData().df_s, TradeData().df_m, TradeData().df_h, or TradeData().df_d
        var: variable to plot; 'low', 'high', 'open', 'close', 'coin_vol', 'usd_vol', 'return'
        n: periods over which to calculate rolling statistics
        bollinger: include "Bollinger Bands" (+/- 2 standard deviation trends)
        
    Returns: 
        None
    """
    ex = df.columns[0][:-4]  # Exhcange name
    col = ex + '_' + var  # var to plot

    ax = df[col].plot(title = (ex + ': ' + var.title()) \
        , label=ex, figsize = (16, 12), color='b')

    rm = df[col].rolling(window=n).mean()
    rm.plot(label=(str(n) + '-period' + ' Rolling Mean'), color='g', ax=ax)

    if bollinger:
        sd = df[col].rolling(window=n).std()
        upper_bound = rm + 2*sd
        lower_bound = rm - 2*sd

        upper_bound.plot(label=(str(n) + '-period' + ' Rolling Mean + 2sd'), color='r', ax=ax)
        lower_bound.plot(label=(str(n) + '-period' + ' Rolling Mean - 2sd'), color='r', ax=ax)

    ax.set_xlabel("datetime")
    ax.set_ylabel(var)
    ax.legend(loc='upper left')
    
    plt.show()

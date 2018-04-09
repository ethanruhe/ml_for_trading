"""
Ethan Ruhe
"""
import numpy as np
import pandas as pd
from sklearn.utils import resample
import datetime as dt

def calc_static_features(df_input):
    """Calcualte features that do not depend on rolling metrics:
        > return one period ago
        > number of periods with positive return in a row
        > number of periods with negative return in a row

    Args:
        df: candlestick trade data with low, high, open, close, vol, usd_vol variables in df with time index

    Return:
        df of static features; index is same as input
    """
    df = pd.DataFrame(index=df_input.index)
    ex = df_input.columns[0][:-4]  # Exhcange name

    low_var     = ex + '_low'
    high_var    = ex + '_high'
    open_var    = ex + '_open'
    close_var   = ex + '_close'
    return_var  = ex + '_return'
    vol_var     = ex + '_coin_volume'
    usd_vol_var = ex + '_usd_volume'

    ############################################################
    # Return feature creation
    ############################################################
    # lagged return
    return_var_lag1 = return_var + '_lag1'
    df[return_var_lag1] = df_input[return_var]  # will be a lag when shifted

    positive_return = (df_input[return_var] > 0)
    negative_return = (df_input[return_var] <= 0)  # note: includes 0% return

    # momentum from period return: number of periods in same direction
    return_momentum = \
        positive_return.groupby(positive_return.ne(positive_return.shift()).cumsum()).cumcount()

    pos_flag = return_var + '_pos'
    pos_momentum_var = return_var + '_pos_momentum'
    neg_momentum_var = return_var + '_neg_momentum'

    df[pos_momentum_var] = positive_return * return_momentum
    df[neg_momentum_var] = negative_return * return_momentum
    #df[pos_flag] = positive_return

    ############################################################
    # Time Features
    ############################################################
    # Hour of day
    #df['hour'] = df.index.hour

    #return df.loc[:, [return_var_lag1, pos_momentum_var, neg_momentum_var, pos_flag]]
    return df.loc[:, [return_var_lag1, pos_momentum_var, neg_momentum_var]]


def calc_rolling_features(df_input, n=10):
    """Calculate normalized features for clean candlestick data.
    > Normalized dispersion of closing price
    > STD's from mean of closing price
    > Normalized dispersion of USD volume
    > STD's from mean of USD volume

    Args:
        df: candlestick trade data with low, high, open, close, vol, usd_vol variables in df with time index
        n: list of periods overwhich to calculate rolling stats

    Return:
        df of features
    """
    df = pd.DataFrame(index=df_input.index)
    ex = df_input.columns[0][:-4]  # Exhcange name

    low_var     = ex + '_low'
    high_var    = ex + '_high'
    open_var    = ex + '_open'
    close_var   = ex + '_close'
    return_var  = ex + '_return'
    vol_var     = ex + '_coin_vol'
    usd_vol_var = ex + '_usd_vol'

    ############################################################
    # Normalized rolling price feature creation
    ############################################################
    rm = df_input[close_var].rolling(window=n, min_periods=n).mean()  # rolling mean
    rstd = df_input[close_var].rolling(window=n, min_periods=n).std()  # rolling std

    # normalized dispersion: std magnitude compared to meaxn
    stdmag_var = close_var + '_stdmag_' + str(n)
    df[stdmag_var] = rstd / rm

    # normalized price: std's from mean
    stds_var = close_var + '_stds_' + str(n)
    df[stds_var] = (df_input[close_var] - rm) / rstd


    ############################################################
    # Rolling return feature creation
    ############################################################
    rm_returns_var = return_var + '_rm_' + str(n)
    df[rm_returns_var] = df_input[return_var].rolling(window=n, min_periods=n).mean()


    ############################################################
    # Normalized rolling volume (USD) feature creation
    ############################################################
    rm_usd_vol = df_input[usd_vol_var].rolling(window=n, min_periods=n).mean()
    rstd_usd_vol = df_input[usd_vol_var].rolling(window=n, min_periods=n).std()

    # normalized dispersion: std magnitude compared to mean
    usd_vol_stdmag_var = usd_vol_var + '_stdmag_' + str(n)
    df[usd_vol_stdmag_var] = rstd_usd_vol / rm_usd_vol

    # normalized volume: std's from mean
    usd_vol_stds_var = usd_vol_var + '_stds_' + str(n)
    df[usd_vol_stds_var] = (df_input[usd_vol_var] - rm_usd_vol) / rstd_usd_vol


    ############################################################
    # Stochastic Oscillator
    ############################################################
    rlow = df_input[low_var].rolling(window=n, min_periods=n).min()  # rolling low price
    rhigh = df_input[high_var].rolling(window=n, min_periods=n).max()  # rolling high price
    lag_close = df_input[close_var].shift()
    k = 100 * (lag_close - rlow) / (rhigh - rlow)

    so_var = ex + '_so_' + str(n)
    df[so_var] = k.rolling(window=3, min_periods=1).mean()  # consider cross validating window, min,s


    return df.loc[:, [stdmag_var, stds_var, rm_returns_var, usd_vol_stdmag_var, usd_vol_stds_var, so_var]]


def calc_features(df, rolling_periods=[10], y_binary=True):
    """For input df, return static features, rolling features for each period in list, and dependent var
    X is shifted so that X_i-1 can be used to predict Yi
    Args:
        df: trade_data.df_m most likely; assumes similar format
        rolling_periods: list windows over which to calculate rolling stats
        y_binary: use actual period return % as dependent var, or is-positive Boolean
    Return:
        Y, X
    """
    ex = df.columns[0][:-4]  # Exhcange name
    y  = ex + '_return'  # Var to be predicted
    # Calc static features
    # Includes '_return_pos' flag, used as Y
    output = calc_static_features(df)
    
    # For each rolling period window, calc rolling features
    for i in rolling_periods:
        output = output.join(calc_rolling_features(df, n=i))
    # Remove leading obs for which full rolling stats couldn't be calculated
    output = output.iloc[max(rolling_periods)-1:,:]
    # Set NaNs to 0.0
    # In _stdmag_, occurs when rollling mean is 0; e.g., in vol data when no trading
    # In _stds_, can occur when std is 0; e.g., when price or vol are stable
    #TODO: log what was NaN (output.isna().sum()) before updating
    output[output.isna()] = 0.0

    # Join return column from df
    output = output.join(df[y])

    Y = output[y]  # only y with indicies in output
    if y_binary:
        Y = Y > 0.0

    X = output.drop(columns=[y])

    # Shift X obs one period, so Yi aligns with X data from period i-1
    # Ensures not using functions of Yi to predict Yi
    # Then, drop first obs of Y, X since X[0] is NaN
    X = X.shift(1).iloc[1:,:]
    Y = Y.iloc[1:]

    return Y, X


def split_test_train(Y, X, test_pct=0.2, upsample=True):
    """Split Y, X into test and train sets where test data always come after
        train data. If upsample, then ensure test set has same number of
        positive and negative return periods by oversampling minority class.
    Args:
        Y, X: input data to be split
        test_pct: proportion of data to be set aside as test data 
        upsample: whether to balance the +/- return classes in test data
    Return:
        Y_train, X_train, Y_test, X_test

    """
    y  = Y.name  # Var to be predicted
    
    df = X.join(Y)

    # Split into test/ train: last test_pct are test
    i_split = int(df.shape[0] * (1 - test_pct))
    df_train = df.iloc[0:i_split,]
    df_test = df.iloc[i_split:,]

    # Resample minority class to achieve 50/50 split
    df_train_return_pos = df_train[df_train[y] > 0.0]
    df_train_return_neg = df_train[df_train[y] <= 0.0]
    
    # Resample minority class to as many obs as majority
    #TODO: log positive/negative return split in training data
    negatives = df_train_return_neg.shape[0]
    positives = df_train_return_pos.shape[0]
    if negatives > positives:
        df_train_return_pos_resampled = resample(df_train_return_pos, 
                                         replace=True,        # with replacement
                                         n_samples=negatives, # create 50-50 spilt
                                         random_state=42)     # reproducible results 
        df_train_resampled = pd.concat([df_train_return_neg, df_train_return_pos_resampled])
    elif positives > negatives:
        df_train_return_neg_resampled = resample(df_train_return_neg, 
                                         replace=True,        # with replacement
                                         n_samples=positives, # create 50-50 spilt
                                         random_state=42)     # reproducible results   
        df_train_resampled = pd.concat([df_train_return_pos, df_train_return_neg_resampled])
    else:
        df_train_resampled = pd.concat([df_train_return_pos, df_train_return_neg])

    Y_train = df_train_resampled[y]
    X_train = df_train_resampled.drop(columns=[y])

    Y_test = df_test[y]
    X_test = df_test.drop(columns=[y])

    return Y_train, X_train, Y_test, X_test


def build_orders(dfprediction, abs_threshold=0.02, startin=False, symbol='USD-BTC'):
    """Create df of orders to make (at period close)
    Expects input to have time index and 'y_hat' col that corresponds to predicted return for i
    Output: columns=['exchange', 'order', 'order_vol'], index=df.index
    Single currency, net holdings possible at -10, 0, 10
    
    TODO: vectorize once settled on format; bit more intuitive to manipulate non-vectorized but slow
    """
    # Build order info
    tsymbol = pd.Series(index=dfprediction.index)
    torder = pd.Series(index=dfprediction.index)
    tshares = pd.Series(index=dfprediction.index)
    
    if startin: 
        shares = 1
    else:
        shares = 0

    # if predicted return in i+1 suggests trade, input order in i
    for i in dfprediction.index[:-1]:
        
        # Sell when prediction < -(abs_threshold)
        if dfprediction.loc[i + dt.timedelta(hours=1), 'y_hat'] < -1.0 * abs_threshold:      

            # if currently own -1: pass
            if shares == -1:
                pass

            # if currently own 0: get to net -1 share position
            elif shares == 0:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'SELL'
                tshares.loc[i] = 1
                shares -= 1

            # if currently own 1: get to net -1 share position
            elif shares == 1:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'SELL'
                tshares.loc[i] = 2
                shares -= 2

            else:
                raise Exception("Unexpected share holdings. 0")

        # Buy when prediction > abs_threshold
        elif dfprediction.loc[i + dt.timedelta(hours=1), 'y_hat'] > abs_threshold:      

            # if currently own -1: get to net 1 share position
            if shares == -1:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'BUY'
                tshares.loc[i] = 2
                shares += 2

            # if currently own 0: get to net 1 share position
            elif shares == 0:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'BUY'
                tshares.loc[i] = 1
                shares += 1

            # if currently own 1: pass
            elif shares == 1:
                pass 

            else:
                raise Exception("Unexpected share holdings. 1")            

        # if predicted return is too small, pass
        else:
            pass

        df_orders = pd.concat({'Symbol': tsymbol, 'Order': torder, 'Shares': tshares}, axis=1).dropna()

    return df_orders

def port_strategy(dfprediction, abs_threshold=0.02, symbol='USD-BTC'):
    """Create df of orders to make (at period close)
    Expects input to have time index and 'y_hat' col that corresponds to predicted return for i
    Output: columns=['exchange', 'order', 'order_vol'], index=df.index
    Single currency, net holdings possible at -10, 0, 10
    
    TODO: vectorize once settled on format; bit more intuitive to manipulate non-vectorized but slow
    """
    # Build order info
    tsymbol = pd.Series(index=dfprediction.index)
    torder = pd.Series(index=dfprediction.index)
    tport_pct = pd.Series(index=dfprediction.index)

    net_position = 0
    # if predicted return in i+1 suggests trade, input order in i
    for i in dfprediction.index[:-1]:
        
        # Sell when prediction < -(abs_threshold)
        if dfprediction.loc[i + dt.timedelta(hours=1), 'y_hat'] < -1.0 * abs_threshold:      

            # if currently own -1: pass
            if net_position == -1:
                pass

            # if currently own 0: get to net -1 share position
            elif net_position == 0:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'SELL'
                tshares.loc[i] = 1
                shares -= 1

            # if currently own 1: get to net -1 share position
            elif shares == 1:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'SELL'
                tshares.loc[i] = 2
                shares -= 2

            else:
                raise Exception("Unexpected share holdings. 0")

        # Buy when prediction > abs_threshold
        elif dfprediction.loc[i + dt.timedelta(hours=1), 'y_hat'] > abs_threshold:      

            # if currently own -1: get to net 1 share position
            if shares == -1:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'BUY'
                tshares.loc[i] = 2
                shares += 2

            # if currently own 0: get to net 1 share position
            elif shares == 0:
                tsymbol.loc[i] = symbol  # trade symbol
                torder.loc[i] = 'BUY'
                tshares.loc[i] = 1
                shares += 1

            # if currently own 1: pass
            elif shares == 1:
                pass 

            else:
                raise Exception("Unexpected share holdings. 1")            

        # if predicted return is too small, pass
        else:
            pass

        df_orders = pd.concat({'Symbol': tsymbol, 'Order': torder, 'Shares': tshares}, axis=1).dropna()

    return df_orders

def compute_portvals(dforders, dfprices, trend, start_val=10000, commission=0.0029, impact=0.005):
    """Calculate by-period porfolio values.
    	Note this generalizes to portfolios of multiple assets.
    	Note this assumes free, unlimited margin trading (e.g., can have negative cash balance)
    	Note starting portfolio is assumed to be only cash, no assets
	Args:
		dforders: dataframe of orders, in the same format as output from .build_orders()
		dfprices: dataframe of prices for all assets in dforders
		start_val: cash with which to trade before any market orders
		commission: commission rate of market order trades
		impact: price impact of market orders; scalar to increase price of buys, decrease price of sells
	Return:
		portvals: by-period value of portfolio
    """
    # Subset to only close price data; drop '_close' suffix
    dfprices = dfprices.filter(regex='_close$').rename(columns=lambda x: x[:7])
    dfprices['Cash'] = 1.0  # add cash as unit cost

    # Assets to trade
    #symbols = [i + '_close' for i in pd.unique(dforders['Symbol'])]

    # dftrades
    dftrades = pd.DataFrame(data = np.zeros((len(trend), dfprices.shape[1]))
            , index = trend
            , columns = dfprices.columns)

    # Fill in trades
    for order in range(dforders.shape[0]):
        # Change dftrades of dforders['Symbol'] at dforders.index to dforders['Shares']
        if dforders['Order'][order] == 'BUY':
            delta_shares = dforders['Shares'][order]
            dftrades.loc[dforders.index[order], dforders['Symbol'][order]] += delta_shares
            dftrades.loc[dforders.index[order], 'Cash'] += -(1.0 + commission + impact) * delta_shares * dfprices.loc[dforders.index[order], dforders['Symbol'][order]]

        else:
            delta_shares = dforders['Shares'][order]
            dftrades.loc[dforders.index[order], dforders['Symbol'][order]] += delta_shares * -1
            dftrades.loc[dforders.index[order], 'Cash'] += (1.0 - impact - commission) * delta_shares * dfprices.loc[dforders.index[order], dforders['Symbol'][order]]

    # dfholdings
    dfholdings = pd.DataFrame(data = np.zeros((len(trend), dfprices.shape[1]))
            , index = trend
            , columns = dfprices.columns)

    start_date = trend[0]
    dfholdings.loc[start_date, 'Cash'] = start_val

    dfholdings.iloc[0,] += dftrades.iloc[0,]
    for i in range(dfholdings.shape[0] - 1):
        dfholdings.iloc[i+1,] = dfholdings.iloc[i,] + dftrades.iloc[i+1,]

    # dfvalue
    dfvalue = dfholdings * dfprices

    # portval
    portvals = pd.DataFrame(index=trend, data=dfvalue.sum(axis=1))

    return portvals



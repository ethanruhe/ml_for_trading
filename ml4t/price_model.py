"""
Ethan Ruhe
#TODO: Add logs to save data cleaning details
#TODO: Add tests
#TODO: Refactor to allow for predicting +/-= return, or % return
#TODO: Refactor to allow for predicting return X periods into the future
"""
import numpy as np
import pandas as pd
from sklearn.utils import resample

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
	df[return_var_lag1] = df_input[return_var].shift()

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

	#return df.loc[:, [return_var_lag1, pos_momentum_var, neg_momentum_var, pos_flag]]
	return df.loc[:, [return_var_lag1, pos_momentum_var, neg_momentum_var]]


def calc_rolling_features(df_input, n=10):
	"""Calculate normalized features for clean candlestick data.
		> Normalized dispersion of closing price
		> STD's from mean of closing price
		> Normalized dispersion of USD volume
		> STD's from mean of USD volume
		> Rolling mean return

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

	return df.loc[:, [stdmag_var, stds_var, rm_returns_var, usd_vol_stdmag_var, usd_vol_stds_var]]


def calc_features(df, rolling_periods=[10], upsample=True, y_binary=True):
	"""For input df, return static features, rolling features for each period in list, and dependent var

	Args:
		df: trade_data.df_m most likely; assumes similar format
		rolling_periods: list windows over which to calculate rolling stats
		upsample: balance number of positive and negative return obs by upsampling minority class in data
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

	# Remove leading obs for which fulll rolling stats couldn't be calculated
	output = output.iloc[max(rolling_periods)-1:,:]

	# Set NaNs to 0.0
	# In _stdmag_, occurs when rollling mean is 0; e.g., in vol data when no trading
	# In _stds_, can occur when std is 0; e.g., when price or vol are stable
	#TODO: log what was NaN (output.isna().sum()) before updating
	output[output.isna()] = 0.0

	# Join return column from df
	output = output.join(df[y])

	# Resample minority class to achieve 50/50 split
	df_return_pos = output[output[y] > 0.0]
	df_return_neg = output[output[y] <= 0.0]

	# Resample minority class to as many obs as majority
	#TODO: log positive/negative return split in training data
	negatives = df_return_neg.shape[0]
	positives = df_return_pos.shape[0]

	if negatives > positives:
	    df_return_pos_resampled = resample(df_return_pos, 
	                                     replace=True,        # with replacement
	                                     n_samples=negatives, # create 50-50 spilt
	                                     random_state=42)     # reproducible results 
	    output_resampled = pd.concat([df_return_neg, df_return_pos_resampled])

	elif positives > negatives:
	    df_return_neg_resampled = resample(df_return_neg, 
	                                     replace=True,        # with replacement
	                                     n_samples=positives, # create 50-50 spilt
	                                     random_state=42)     # reproducible results   
	    output_resampled = pd.concat([df_return_pos, df_return_neg_resampled])
	else:
	    output_resampled = pd.concat([df_return_pos, df_return_neg])


	Y = output_resampled[y]
	if y_binary:
		Y = Y > 0.0

	X = output_resampled.drop(columns=[y])

	return Y, X
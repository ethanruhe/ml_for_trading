"""
Ethan Ruhe
"""
import numpy as np
import pandas as pd

def calc_static_features(df_input):
	"""Calcualte features that do not depend on rolling metrics.

		Args:
			df: candlestick trade data with low, high, open, close, vol, usd_vol variables in df with time index

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
	df[return_var + '_pos'] = positive_return

	return df.loc[:, [return_var_lag1, pos_momentum_var, neg_momentum_var, pos_flag]]

def calc_rolling_features(df_input, n=10):
	"""Calculate normalized features for clean candlestick data.

		Args:
			df: candlestick trade data with low, high, open, close, vol, usd_vol variables in df with time index
			n: periods overwhich to calculate rolling stats

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
	# Correct
	############################################################
	#df.replace([np.inf, -np.inf], np.nan, inplace=True)
	#df.fillna(0.0, inplace=True)

	return df.loc[:, [stdmag_var, stds_var, rm_returns_var, usd_vol_stdmag_var, usd_vol_stds_var]]


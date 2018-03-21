""" 
Ethan Ruhe
"""

import numpy as np
import pandas as pd

class TradeData(object):
	"""Candle stick trade data object.

	Takes by-second candle stick trade data (aggregated every 60 seconds) for single currency.
	Assumes format is Pandas df with second index and columns in the following order:
		0: low_price
		1: high_price
		2: open_price
		3: close_price
		4: coin volume 
	Prices assumed in USD.

	TODO: test that df_s is in correct format
	TODO: test low <= high; low <= open; low <= close; high >= open; high >= close; volume >= 0
	TODO: test implicit asusmption that one column's row value is NaN iff all in row are NaN
	TODO: methods for initializing from CSV or request JSON

	Attributes:
		df_s: Pandas df of by-second trades
	"""


	def __init__(self, df_s):
		"""Init TradeData assuming properly formatted data"""
		self.df_s = df_s


	def fill_forward(self):
		"""Fill pricing data forward to candles in which there is no trading.
			
			All prices for period w/o trades equal close of most recent period
			with trading and volume is 0.

			Leading NaNs in series are possible.

			TODO: add test that after transformations, only leading NaNs
		"""
		# Fill forward closing price
		self.df_s.iloc[:,3].fillna(method='ffill', inplace=True)
		
		# low, high, open equal close where filled forward
		self.df_s.loc[self.df_s.iloc[:,0].isnull(), self.df_s.columns[0:3]] = self.df_s.iloc[:,3]

		# volume = 0 where filled forward
		self.df_s.loc[self.df_s.iloc[:,4].isnull(), self.df_s.columns[4]] = 0.0

		print('Note: df has {} leading NaN periods.'.format(self.df_s.iloc[:,0].isnull().sum()))


	def calc_usd_vol(self):
		"""Calculate the approximate USD value of coins traded in period.
			Assumes a uniform distribution of price from open to close in period.
			Large movements from open to close may result in inaccurate
			calculations of USD value.
		"""
		ex = self.df_s.columns[0][:-4]  # Exhcange name
		self.df_s[ex + '_usd_vol'] = (self.df_s.iloc[:,2] + self.df_s.iloc[:,3]) / 2.0 * self.df_s.iloc[:,4]


	def calc_pct_return(self):
		ex = self.df_s.columns[0][:-4]  # Exhcange name
		self.df_s[ex + '_return'] = (self.df_s.iloc[:,3] - self.df_s.iloc[:,2]) / self.df_s.iloc[:,2]


	def aggregate_df(self, level):
		"""Aggregate data into by-minute, by-hour, and by-day resolutions"""
		ex = self.df_s.columns[0][:-4]  # Exhcange name

		lowp   = self.df_s.iloc[:,0].resample(level).min()  # period low
		highp  = self.df_s.iloc[:,1].resample(level).max()  # period high
		openp  = self.df_s.iloc[:,2].resample(level).first()  # period open
		closep = self.df_s.iloc[:,3].resample(level).last()  # period close
		vol    = self.df_s.iloc[:,4].resample(level).sum()  # period volume
		usdvol = self.df_s.loc[:, ex+'_usd_vol'].resample(level).sum()  # period usd volume

		aggregated = lowp.to_frame().join(highp).join(openp).join(closep).join(vol).join(usdvol)

		aggregated[ex + '_return'] = (aggregated.iloc[:,3] - aggregated.iloc[:,2]) / aggregated.iloc[:,2]

		return aggregated


	def clean_df_m(self):
		"""Clean raw by-minute data."""
		self.fill_forward()
		self.calc_usd_vol()
		self.calc_pct_return()

		self.df_m = self.aggregate_df('min')
		self.df_h = self.aggregate_df('H')
		self.df_d = self.aggregate_df('D')
		print("""Minute-level data have been cleaned.\nMinute, hour, and day-level data are available as .df_m, .df_h, and .df_d, respectively.""")

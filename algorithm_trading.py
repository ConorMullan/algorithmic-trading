import datetime
import pandas as pd
import numpy as np
import glob
import talib as ta

from pandas.plotting import scatter_matrix
from pandas_datareader import data as pdr
from pandas import tseries

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn import utils

from scipy.cluster import hierarchy
from scipy.spatial import distance


print(pd.__version__)
# Data Preparatiom
# Removed the two all stocks .csv files, causing duplicates
path = r'C:/Users/User/Documents/Jupyter Notebooks/stock data/stock-20050101-to-20171231'
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
	# print(filename)
	dfs.append(pd.read_csv(filename))
	# Choose how many datasets you want
	if len(dfs) == 3:
		break

# print(dfs[-1:])
stocks = pd.concat(dfs, ignore_index=True)

stocks = stocks.set_index(['Date', 'Name'])
stocks.dropna(inplace=True)
#print(stocks.head())
# Features / X values
features = pd.DataFrame(index=stocks.index)

features.index = features.index.rename(['date', 'name'])

# Feature engineering
features['open'] = stocks.Open
features['close'] = stocks.Close
features['volume'] = stocks.Volume
features['high'] = stocks.High
features['low'] = stocks.Low
# Momentum over 5 days
features['momentum_5_day'] = stocks.groupby(level='Name').Close.pct_change(5)
# Change between the days
features['intraday_change'] = (stocks.groupby(level='Name').Close.shift(0) - stocks.groupby(level='Name').Open.shift(0))/stocks.groupby(level='Name').Open.shift(0)
# Daily percentage increase/decrease in stocks within open and close
features['daily_pct_return'] = stocks.Close/stocks.Open-1
daily_close_px = stocks.Close.groupby(level='Name')
#Odd behaviour with pct_change() and groupby (?)
pct_change_fctn = lambda x: x.pct_change()
# Change from closing to next closing
features['daily_pct_change']  = daily_close_px.apply(pct_change_fctn)
features['daily_return'] = stocks.Open/stocks.groupby(level='Name').Close.shift(1)-1
ma_5 = lambda x: x.rolling(5).mean()
ma_20 = lambda x: x.rolling(20).mean()
ma_100 = lambda x: x.rolling(100).mean()
ma_200 = lambda x: x.rolling(200).mean()
std_20 = lambda x: x.rolling(20).std()
# Close moving average over 20 days
features['close_ma_20'] = stocks.groupby(level='Name').Close.apply(ma_20)# Log of 5 day moving average of volume
# features.close_ma_20.unstack().plot(title='Close Short Moving Average')
# plt.ylabel('Close SMA')
# plt.show()
features['close_std_20'] = stocks.groupby(level='Name').Close.apply(std_20)
features['close_ma_20_log'] = stocks.groupby(level='Name').Close.apply(ma_20).apply(np.log) # Log of 5 day moving average of volume
# features.close_ma_20_log.unstack().plot(title='Log Close Short Moving Average')
# plt.ylabel('Log Close SMA')
# plt.show()
features['log_daily_volume'] = stocks.Volume.apply(np.log) # log of daily volume
features['day_volume_diff'] = stocks.groupby(level='Name').Volume.diff() # Volume change since the prior day
features['50_day_volume_diff'] = stocks.groupby(level='Name').Volume.diff(50) # Volume change since 50 days
features['volume_change_rate'] = stocks.groupby(level='Name').Volume.apply(pct_change_fctn) # Daily percentage change of the volume
features['volume_change_ratio'] = stocks.groupby(level='Name').Volume.diff(1) / stocks.groupby(level='Name').shift(1).Volume # Daily ratio change of the volume

features['volume_ma_5'] = stocks.groupby(level='Name').Volume.apply(ma_5).apply(np.log) # Log of 5 day moving average of volume
features['volume_ma_20'] = stocks.groupby(level='Name').Volume.apply(ma_20).apply(np.log) # Log of 20 day moving average of volume
features['volume_ma_100'] = stocks.groupby(level='Name').Volume.apply(ma_100).apply(np.log) # Log of 100 day moving average of volume
features['volume_ma_200'] = stocks.groupby(level='Name').Volume.apply(ma_200).apply(np.log) # Log of 200 day moving average of volume
min_periods = 75

# Statistical measure of dispersion of returns for given stocks
features['volatility'] = stocks.Close.groupby(level='Name').apply(pct_change_fctn).rolling(min_periods).std() * np.sqrt(min_periods)
# features.volatility.unstack().plot(figsize=(10,8))
# plt.show()

# Key error: uses the mean and sd of whole time frame to calculate each datapoint
# Used rolling so each data point doesn't use the whole time frame
zscore = lambda x: (x - x.rolling(window=200, min_periods=20).mean()) / x.rolling(window=200, min_periods=20).std()
features['z_score'] = stocks.groupby(level='Name').Close.apply(zscore)
features.z_score.unstack().plot.kde(title='z-scores')
plt.show()


# Technical indicators
# Money flow index for 200 periods of data
# above 80 is considered overbought and below 20 is oversold
features['money_flow_ta'] = ta.MFI(stocks.High,stocks.Low,stocks.Close,stocks.Volume, timeperiod=200)
fig, ax = plt.subplots()
features.money_flow_ta.unstack().plot(ax=ax,legend=True, figsize=(12,10), title='Money Flow Index')
plt.xlabel('Date')
plt.show()

# Williams %R
# Momentum indicator that moves between 0 and -100 and measures overbought, over 500 periods
features['will_r_ta'] = ta.WILLR(stocks.High.rolling(14).mean(), stocks.Low.rolling(14).mean(),stocks.Close.rolling(13).mean(), timeperiod=14)
fig, ax = plt.subplots()
ax.set_xlabel('Date')
features.will_r_ta.unstack().plot(ax=ax, legend=True, figsize=(12,10))
plt.xlabel('Date')
plt.ylabel
plt.show()

# Bollinger Bands %b (volitility indicator)
# Errors and slow loading, works better with one dataset
features['BB_up'], features['BB_mid'], features['BB_low'] = ta.BBANDS(stocks.Close, timeperiod=200, nbdevup=2, nbdevdn=2, matype=0)
fig, ax = plt.subplots(sharex=True, figsize=(12, 10))

for key, grp in stocks.groupby(level='Name'):
	ax.plot(stocks.index.get_level_values(level='Date'), features.BB_up, label='BB_up')
	ax.plot(stocks.index.get_level_values(level='Date'), stocks.Close.rolling(window=20).mean(), label='Close')
	ax.plot(stocks.index.get_level_values(level='Date'), features.BB_low, label='BB_low')
	ax.fill_between(stocks.index.get_level_values(0), y1=features.BB_low, y2=features.BB_up, color='#adccff', alpha='0.3')

ax.set_xlabel('Date')
ax.set_ylabel('Bollinger Bands')
plt.show()

features.dropna(inplace=True)

# Scatter matrix showing percentage change of daily closing prices
pd.plotting.scatter_matrix(features.daily_pct_change.unstack(),diagonal='kde', alpha=0.1, figsize=(12,12))
plt.show()

# Outcomes / Y values
outcomes = pd.DataFrame(index=stocks.index).sort_index()
outcomes['close_1'] = stocks.groupby(level='Name').Close.pct_change(-1) # next day's closing price
outcomes['close_5'] = stocks.groupby(level='Name').Close.pct_change(-5) # 5 days ahead closing price
outcomes['close_10'] = stocks.groupby(level='Name').Close.pct_change(-10) # 10 days ahead closing price
outcomes['close_20'] = stocks.groupby(level='Name').Close.pct_change(-20) # 20 days ahead closing price
outcomes['close_50'] = stocks.groupby(level='Name').Close.pct_change(-50) # 50 days ahead closing price
outcomes['open_10'] = stocks.groupby(level='Name').Open.pct_change(-10)
outcomes['volume_1'] = stocks.groupby(level='Name').Volume.pct_change(-1)
outcomes['volume_10'] = stocks.groupby(level='Name').Volume.pct_change(-10)
outcomes['volume_20'] = stocks.groupby(level='Name').Volume.pct_change(-20)

outcomes.index = outcomes.index.rename(['date', 'name'])
#print(outcomes.head())

# Seperate dataframes for individual stocks and create return column
all_returns = np.log(features['close'] / features['close'].shift(1))
aaba_df = features.iloc[features.index.get_level_values('name') == 'AABA']
aaba_df['returns'] = all_returns.iloc[all_returns.index.get_level_values('name')== 'AABA']
aaba_df.index = aaba_df.index.droplevel('name')

aapl_df = features.iloc[features.index.get_level_values('name') == 'AAPL']
aapl_df['returns'] = all_returns.iloc[all_returns.index.get_level_values('name')== 'AAPL']
aapl_df.index = aapl_df.index.droplevel('name')
aapl_df = aapl_df.sort_index()
#
aapl_df.dropna(inplace=True)
amzn_df = features.iloc[features.index.get_level_values('name') == 'AMZN']
amzn_df['returns'] = all_returns.iloc[all_returns.index.get_level_values('name')== 'AMZN']
amzn_df.index = amzn_df.index.droplevel('name')
# amzn_returns = all_returns.iloc[all_returns.index.get_level_values('name')== 'AABA']

amzn_df.dropna(inplace=True)

# New dataframe with AAPL and AMZN returns
return_data = pd.concat([aapl_df.returns, amzn_df.returns], axis=1)[1:]
return_data.columns = ['AAPL', 'AMZN']

# Constant
C = sm.add_constant(return_data['AAPL'])

# OLS model
ols_model = sm.OLS(return_data['AMZN'],C).fit()

print(ols_model.summary())

plt.plot(return_data['AAPL'], return_data['AMZN'], 'r.')

ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)

# Parameters for the plot
plt.plot(x, ols_model.params[0] + ols_model.params[1] * x, 'b', lw=2)

plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Amazon returns')

plt.show()

return_data['AMZN'].rolling(window=252).corr(return_data['AAPL']).plot()
plt.show()


# Building a Trading Strategy
# Short windows and long windows for window of rolling
short_window=40
long_window = 100
# Initialize the 'signals' DataFrame with signal column
signals = pd.DataFrame(index=aapl_df.index)
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = aapl_df['close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = aapl_df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   
# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price in $')

# Plot the closing price
aapl_df['close'].plot(ax=ax1, color='r', lw=2.)

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
         
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
         
# Show the plot
plt.show()


# Backtesting the Strategy
# Set the initial capital
initial_capital= float(100000.0)
signals= signals.sort_index()
# Create a DataFrame `positions`
positions = pd.DataFrame(index=signals.index).fillna(0.0)

# Buy a 100 shares
positions['AAPL'] = 100*signals['signal']   
  
# Initialize the portfolio with value owned   
portfolio = positions.multiply(aapl_df['close'], axis=0)

# Store the difference in shares owned 
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(aapl_df['close'], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl_df['close'], axis=0)).sum(axis=1).cumsum()   

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

fig = plt.figure()

ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)

# Plot the "buy" trades against the equity curve
ax1.plot(portfolio.loc[signals.positions == 1.0].index, 
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')

# Plot the "sell" trades against the equity curve
ax1.plot(portfolio.loc[signals.positions == -1.0].index, 
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')

# Show the plot
# plt.show()

# Evaluating Moving Average Crossover Strategy
# isolate returns
returns = portfolio['returns']

# annualized sharpe ratio
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

#display
print("Sharpe ratio: ",sharpe_ratio)

#Maximum drawdown
# 252 trading day window
window = 252

# Max drawdown in the past window days for each day
rolling_max = aapl_df['close'].rolling(window, min_periods=1).max()
daily_drawdown = aapl_df['close']/rolling_max - 1.0

# Minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()


# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()

# Show the plot
plt.show()

# End of trading strategy

# Standardize or normalize X
std_scaler = StandardScaler()
features_scaled = std_scaler.fit_transform(amzn_df)
x_scaled = pd.DataFrame(features_scaled, index=amzn_df.index)
x_scaled.columns = amzn_df.columns
# print(x_scaled.tail())

# preprocess Y
outcomes_df = outcomes.iloc[outcomes.index.get_level_values('name') == 'AMZN']
# print("Outcomes_df", outcomes_df.shape)
y_scaled = std_scaler.fit_transform(outcomes_df.dropna())
y_scaled = pd.DataFrame(y_scaled, index=outcomes_df.dropna().index)
y_scaled.columns = outcomes_df.columns

# print(y_scaled.tail())


# Pearson coefficient, measuring the strength of the linear relationship between X values and the Y value
corr = x_scaled.corrwith(y_scaled.volume_10)
corr.sort_values().plot.barh(color = 'blue', title= 'Correlation Strength')
plt.show()

 # Correlated features with the highest pearson correlaion
correlated_features = corr[corr>0.1].index.tolist()
corr_matrix = x_scaled[correlated_features].corr()
correlations_array = np.asarray(corr_matrix)
linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
g = sns.clustermap(corr_matrix, row_linkage=linkage, col_linkage=linkage, row_cluster=True, col_cluster=True, \
	figsize=(10,10), cmap='Greens')
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
label_order = corr_matrix.iloc[:, g.dendrogram_row.reordered_ind].columns
print("Correlation Strength:")
print(corr[corr>.1].sort_values(ascending=False))
plt.show()

#Highest correlation with y and minimal covariance
selected_features = ['volume', 'volume_change_ratio', 'daily_pct_change']
sns.pairplot(x_scaled[selected_features], size=1.5)
plt.show()

Y = y_scaled

# print("Y before join: {} \n {}".format(Y.shape, Y.head()))
XY = x_scaled.join(y_scaled)
XY.dropna(inplace=True)
X = XY[selected_features].values

# print(XY.head())
Y =  XY["volume_10"].values

plt.scatter(XY['volume'], XY['volume_10'])
plt.show()

# never used encoder, instead used int()
# lab_enc = LabelEncoder()
# X_encoded = X
# X_encoded = X_encoded.apply(lab_enc.fit_transform)
# Y_encoded = lab_enc.fit_transform(Y)
# X_encoded = X_encoded.values


# Floats can't be inserted into model selections, so it is cast as an int instead
Y = Y.astype(int)
X = X.astype(int)

# 20% is left out as unseen data for testing
validation_size = 0.2
# Seed for repeatability
seed = 10
# Model selection and parameters
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(5)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

knn = KNeighborsClassifier(5)
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, knn_predictions))
print(confusion_matrix(Y_validation, knn_predictions))
print(classification_report(Y_validation, knn_predictions))


ols = sm.OLS(Y_train, X_train).fit()
print(ols.summary())

ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, ols.params[0] + ols.params[1] * x, 'b', lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel('Selected_features')
plt.ylabel('Volume in 10 days')
plt.show()

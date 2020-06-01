import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use('ggplot')
df = pd.read_csv("./India.csv")
for i in range(32):
    df = df.drop([i], axis = 0)


""" df.dropna(inplace = True) """
new = df['date'].str.split("-", expand= True)
df['Day'] = new[1]
df = df[['Day', 'totalconfirmed', 'totaldeceased', 'totalrecovered', 'dailyconfirmed', 'dailydeceased','dailyrecovered']]

""" print(df.head()) """

df['PCT_CHANGE'] = df.groupby('Day')['totalconfirmed'].apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))
df['DAILY_CHANGE'] = df.groupby('Day')['dailyconfirmed'].apply(lambda x: x.div(x.iloc[0]).subtract(1).mul(100))

df = df[['Day', 'totalconfirmed', 'dailyconfirmed', 'PCT_CHANGE', 'DAILY_CHANGE']]


forecast_col = 'totalconfirmed'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head())

X = np.array(df.drop(['Day','label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)
last_unix = last_date
one_day = 1
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = next_unix
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['totalconfirmed'].plot()
df['dailyconfirmed'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Increase')
plt.show()



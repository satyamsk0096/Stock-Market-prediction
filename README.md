# Stock-Market-prediction
Stock market is a Place where buying and selling of shares  happen for publicly listed companies. Stock exchnage is the  mediator that allows buying and selling of shares.
#Importing Libraries
import pandas as pd
import numpy as np
from sklearn import metrics
%matplotlib inline
import matplotlib.pyplot as plt


#Reading the csv file and putting it into the dataset
dataset=pd.read_csv(r'C:\Users\Satyam Singh Sittu\Downloads\Tesla.csv')

dataset.head()

dataset['Date'] = pd.to_datetime(dataset.Date)

dataset.shape

dataset.drop('Adj Close', axis = 1, inplace = True)

dataset.head()

dataset.isnull().sum()

dataset.isna().any()

dataset.info()

dataset.describe()

print(len(dataset))

dataset['Open'].plot(figsize=(16,6))

# Putting variable to X
X = dataset[['Open','High','Low','Volume']]
# Putting response variable to y
y = dataset['Close']

# Spliting the data into train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

X_train.shape

X_test.shape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()

regressor.fit(X_train,y_train)

print(regressor.coef_)

print(regressor.intercept_)

predicted= regressor.predict(X_test)

print(X_test)

predicted.shape

dframe= pd.DataFrame(y_test,predicted)

dfr=pd.DataFrame({'Actual Price':y_test,'Predicted Price':predicted})

print(dfr)

dfr.head(25)

from sklearn.metrics import confusion_matrix, accuracy_score

regressor.score(X_test,y_test)

import math

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicted))

print('Mean Squared Error:',metrics.mean_squared_error(y_test,predicted))

print(' Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicted)))

graph=dfr.head(20)

graph.plot(kind='bar')

# -*- coding: utf-8 -*-


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
data = pd.read_csv("odev_tenis.csv")

# Getting the data as array

# Categoric data
outlook = data.iloc[:,0:1].values
windy = data.iloc[:,-2:-1].values
play = data.iloc[:,-1:].values

# Numeric data
others = data.iloc[:,1:3].values


from sklearn import preprocessing

# Short way for Label Encoder (All dataset)
"""
data2 = data.apply(preprocessing.LabelEncoder().fit_transform)
"""

# Other way for Label Encoder
le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(data.iloc[:,0])
windy[:,0] = le.fit_transform(data.iloc[:,-2])
play[:,0] = le.fit_transform(data.iloc[:,-1])


# OneHotEncoder 
ohe = preprocessing.OneHotEncoder()

outlook = ohe.fit_transform(outlook).toarray()
windy = ohe.fit_transform(windy).toarray()
play = ohe.fit_transform(play).toarray()


# transform from array to dataframe
d1 = pd.DataFrame(data = outlook, columns = ["Overcast","Rainy","Sunny"])
d2 = pd.DataFrame(data = windy[:,0:1], columns = ["Windy(False)"])
d3 = pd.DataFrame(data = play[:,0:1], columns = ["Play(No)"])
d4 = pd.DataFrame(data = others, columns = ["Temperature","Humidity"])


# Combine to dataframe
s = pd.concat([d1,d2,d3], axis = 1)
s2 = pd.concat([s,d4], axis = 1)


# Splitting the dataset as Train and Test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s2.iloc[:,:-1],s2.iloc[:,-1:], test_size = 0.3, random_state = 11)


# Model building (Multiple-Linear-Regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)


humidity = s2.iloc[:,-1:]
new_data = s2.iloc[:,:-1]

# importing sm for seeing p-values (Backward-Elimination)
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values = new_data, axis = 1)

X_l = new_data.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype = float)

model = sm.OLS(humidity,X_l).fit()
print(model.summary())


# Drop 3. column (Backward-Elimination)
X_l = new_data.iloc[:,[0,1,2,4,5]].values
X_l = np.array(X_l, dtype = float)

model = sm.OLS(humidity,X_l).fit()
print(model.summary())


new_x_train = x_train.drop("Windy(False)", axis = 1)
new_x_test = x_test.drop("Windy(False)", axis = 1)

lr.fit(new_x_train,y_train)

new_y_pred = lr.predict(new_x_test)







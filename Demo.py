from LinearRegression import LinearRegression
import pandas as pd
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')
X =  dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

regressor = LinearRegression()
result ,X_train,X_test, Y_Train,Y_test, Y_train_predict ,Y_test_predict = regressor.cal_R(X,Y,6.23)

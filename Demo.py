#If dataset is small 
from LinearRegression import LinearRegression
import pandas as pd
import numpy as np

#Importing Dataset
dataset = pd.read_csv('style.csv')
X =  dataset.iloc[:,:-1].values   #seperation of Independent variable
Y = dataset.iloc[:,1].values      #seperation of Dependent variable

regressor = LinearRegression()   #creating Object
result ,X_train,X_test, Y_Train,Y_test, Y_train_predict ,Y_test_predict = regressor.cal_R( X, Y, 6.2 """X_value""")

#If dataset is large

"""from LinearRegression import LinearRegression
import pandas as pd
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')
X =  dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

regressor = LinearRegression()
result ,X_train,X_test, Y_Train,Y_test, Y_train_predict ,Y_test_predict = regressor.cal_R( X, Y, 6.2 """X_value""")  """

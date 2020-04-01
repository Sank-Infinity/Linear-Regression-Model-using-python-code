# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:58:06 2020

@author: Sanket Kale
"""
#Importing Required libraties
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Regression Class 
class LinearRegression:
  @staticmethod
  #Calculating regression coefficient ....
  def cal_R(X, Y, x_value):
    #Splitting the dataset into training set and test set
    x_list = []
    y_list = []
    x_test = []
    y_test = []
    length = int(len(X)*(1))   #Replace 1 by appropriate split no. as per data availability (0.8 recommended). If Dataset is Large.
    for i in range(0,length):
      x_list.append(X[i])
      y_list.append(Y[i])
    
    for i in range(length,(len(X))):
      x_test.append(X[i])
      y_test.append(Y[i])
      
    def square(n): #Function for calculating squares
     ls = []
     for i in range(0,len(n)):
       ele = n[i]**2
       ls.append(ele)
     return ls
    #function for calculating coefficient on line y on x
    def Byx(summation_XY , N , XbarYbar, summation_Xsquare, Xbar_square): 
       return ((summation_XY/N)-XbarYbar)/((summation_Xsquare/N)-Xbar_square)
    
    #Function for calculating value of Y using equation
    def B_yx_equation(X_bar, Y_bar, value_X, B_yx):
      return ((B_yx*value_X)-(B_yx*X_bar)+Y_bar)
    
    #Addition of elements of Independent Variables...
    X_addition = sum(x_list)
    
    #Taking square of Independent variables...
    X_square = square(x_list)
    
    #Addition of Elements of Dependent Variables...
    Y_addition = sum(y_list) 
    
    #Taking square of Dependent variables...
    Y_square = square(y_list)
    
    #calculting lenth of dataset
    N = len(x_list)
    
    XY = [] 
    
    for i in range(0,len(x_list)):  #Calculating X*Y
      xy_mul = x_list[i]*y_list[i]
      XY.append(xy_mul)
      
    #taking summation of X*Y
    XY_addition  =  (sum(XY))  
    
    #taking summation of X_square
    sum_X_square = sum(X_square)  
    
    #Now lets calculate covarience  
    X_bar = (X_addition/N)
    X_bar_square = (X_bar**2)
    Y_bar = (Y_addition/N)
    XBarYBar = (X_bar*Y_bar)
    
    #calculating value of Y using equation
    B_yx = Byx(XY_addition, N, XBarYBar, sum_X_square,X_bar_square)
    
    #calculating value of Y using equation
    prediction = B_yx_equation(X_bar,Y_bar,x_value,B_yx )
    
    #List of Predicted values by machine learning model
    y_pred_test= []
    for i in range(len(x_test)):       
      pred = B_yx_equation(X_bar,Y_bar,x_test[i],B_yx )
      y_pred_test.append(pred)
      
    #visualising results of X_test and y_test
    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_test, y_pred_test, color='blue')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    #List of Predicted values by machine learning model
    y_pred_list= []
    for i in range(len(x_list)):       
      pred = B_yx_equation(X_bar,Y_bar,x_list[i],B_yx )
      y_pred_list.append(pred)
       
    #visualising results of X_list and y_list
    plt.scatter(x_list, y_list, color='red')
    plt.plot(x_list, y_pred_list, color='blue')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    #Output
    print("Predicted value for Independent variable {} is {} ".format(x_value, prediction))
    
    return(prediction,x_list,x_test ,y_list, y_test, y_pred_list, y_pred_test)
    


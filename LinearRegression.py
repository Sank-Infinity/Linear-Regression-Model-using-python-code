# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:58:06 2020

@author: Sanket Kale
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class LinearRegression:
  @staticmethod
  #Calculating regression coefficient ....
  def cal_R(X, Y,x_value):
    #Splitting the dataset into training set and test set
    x_list = []
    y_list = []
    x_test = []
    y_test = []
    length = int(len(X)*(1/3))
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
   
    def Byx(summation_XY , N , XbarYbar, summation_Xsquare, Xbar_square): #function for calculating coefficient on line y on x
       return ((summation_XY/N)-XbarYbar)/((summation_Xsquare/N)-Xbar_square)
    
    def B_yx_equation(X_bar, Y_bar, value_X, B_yx):#Function for calculating value of Y using equation
      return ((B_yx*value_X)-(B_yx*X_bar)+Y_bar)
    
    X_addition = sum(x_list) #Addition of elements of Independent Variables...
    X_square = square(x_list)#Taking square of Independent variables...
    Y_addition = sum(y_list) #Addition of Elements of Dependent Variables...
    Y_square = square(y_list)#Taking square of Dependent variables...
    N = len(x_list)
    XY = [] 
    
    for i in range(0,len(x_list)):  #Calculating X*Y
      xy_mul = x_list[i]*y_list[i]
      XY.append(xy_mul)
    XY_addition  =  (sum(XY))   
    sum_X_square = sum(X_square)
    
    #Now lets calculate covarience  
    X_bar = (X_addition/N)
    X_bar_square = (X_bar**2)
    Y_bar = (Y_addition/N)
    XBarYBar = (X_bar*Y_bar)
    B_yx = Byx(XY_addition, N, XBarYBar, sum_X_square,X_bar_square)#calculating value of Y using equation
    prediction = B_yx_equation(X_bar,Y_bar,x_value,B_yx )#calculating value of Y using equation
    
   
    y_pred_test= []
    for i in range(len(x_test)):       #List of Predicted values by machine learning model
      pred = B_yx_equation(X_bar,Y_bar,x_test[i],B_yx )
      y_pred_test.append(pred)
    
    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_test, y_pred_test, color='blue')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    y_pred_list= []
    for i in range(len(x_list)):       #List of Predicted values by machine learning model
      pred = B_yx_equation(X_bar,Y_bar,x_list[i],B_yx )
      y_pred_list.append(pred)
    
    plt.scatter(x_list, y_list, color='red')
    plt.plot(x_list, y_pred_list, color='blue')
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    print(prediction)
    return(prediction,x_list,x_test ,y_list, y_test, y_pred_list, y_pred_test)
    


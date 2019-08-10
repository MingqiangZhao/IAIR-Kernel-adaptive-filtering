# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:24:37 2019

@author: Zhao Mingqiang
"""

import math
import numpy as np

def GausseKernel(sigma,x):
    
    y = math.exp(-x**2/(2*(sigma**2)))
    return y   

def gradient_gausse(sigma,x):
    
    gradient = GausseKernel(sigma,x)*(x/(2*sigma**3))#注意这里没有负号
    return gradient

#regularization
def gausse_regularization_1(sigma,w):
    N = w.shape[1]
    
    dw = np.zeros((1,N))
    for i in range(N):
       dw[0,i] = dw[0,i] + gradient_gausse(sigma,w[0,i])
    
    return dw        

def gausse_regularization_2(sigma,w,c):
    N = w.shape[1]
    
    dw = np.zeros((1,N))
    for i in range(N):
        #for j in range(N):
        dw[0,i] = dw[0,i] + gradient_gausse(sigma,w[0,i]-c[0,i])
    
    return dw     

def gausse_regularization_3(sigma,w):
    N = w.shape[1]
    
    dw = np.zeros((1,N))
    for i in range(N):
        for j in range(N):
            dw[0,i] = dw[0,i] + gradient_gausse(sigma,w[0,i]-w[0,j])
    
    return dw      


#gradient function 梯度函数
def gradient_function_entropy_1(w, x_train, y_train, M, sigma, lamda):
    diff = np.dot(w, x_train) - y_train
    dw = (1./M) * np.dot(diff,np.transpose(x_train)) + 1/M*lamda*gausse_regularization_1(sigma,w)
    
    return dw

def gradient_function_entropy_2(w, x_train, y_train, M, sigma, lamda, c):
    diff = np.dot(w, x_train) - y_train
    dw = (1./M) * np.dot(diff,np.transpose(x_train)) + 1/M*lamda*gausse_regularization_2(sigma,w,c)
    
    return dw

def gradient_function_entropy_3(w, x_train, y_train, M, sigma, lamda):
    diff = np.dot(w, x_train) - y_train
    dw = (1./M) * np.dot(diff,np.transpose(x_train)) + 1/M*lamda*gausse_regularization_3(sigma,w)
    
    return dw



#gradient decent algorithm 梯度下降法算法 
#algorithm 1
def gradient_decent_entropy_1(x_train, y_train, M, w_pre, w, alpha, iteration, sigma, lamda ):
    
    results = np.zeros(iteration)
    dw = gradient_function_entropy_1(w_pre,x_train,y_train,M, sigma, lamda)
    
  
    for i in range(iteration):
      
        w_pre = w_pre - alpha* dw
        results[i] = np.log(np.sum(np.multiply(w-w_pre,w-w_pre)))
        
        dw = gradient_function_entropy_1(w_pre,x_train,y_train,M, sigma, lamda)
                
    return results,w_pre

#algorithm 2
def gradient_decent_entropy_2(x_train, y_train, M, w_pre, w, alpha, iteration, sigma, lamda, c ):
    
    results = np.zeros(iteration)
    dw = gradient_function_entropy_2(w_pre,x_train,y_train,M, sigma, lamda, c)
    
  
    for i in range(iteration):
      
        w_pre = w_pre - alpha* dw
        results[i] = np.log(np.sum(np.multiply(w-w_pre,w-w_pre)))
        
        dw = gradient_function_entropy_2(w_pre,x_train,y_train, M, sigma, lamda, c)
                
    return results,w_pre

#algorithm 3
def gradient_decent_entropy_3(x_train, y_train, M, w_pre, w, alpha, iteration, sigma, lamda ):
    
    results = np.zeros(iteration)
    dw = gradient_function_entropy_3(w_pre,x_train,y_train,M, sigma, lamda)
    
  
    for i in range(iteration):
      
        w_pre = w_pre - alpha* dw
        results[i] = np.log(np.sum(np.multiply(w-w_pre,w-w_pre)))
        
        dw = gradient_function_entropy_3(w_pre,x_train,y_train,M, sigma, lamda)
                
    return results,w_pre
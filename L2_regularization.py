# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:20:04 2019

@author: Zhao Mingqiang
"""

import numpy as np

def L2_regularization(w,c):
    return w - c


def gradient_function_L2(w,x_train,y_train,M,lamda,c):
    diff = np.dot(w, x_train) - y_train
    dw = (1./M) * np.dot(diff,np.transpose(x_train)) + 1/M*lamda*L2_regularization(w,c)
    
    return dw


def gradient_decent_L2(x_train, y_train, M, w_pre, w, alpha ,iteration,lamda, c):
    
    results = np.zeros(iteration)
    dw = gradient_function_L2(w_pre,x_train,y_train,M,lamda,c)
     
    for i in range(iteration):
      
        w_pre = w_pre - alpha* dw
        results[i] = np.log(np.sum(np.multiply(w-w_pre,w-w_pre)))
        dw = gradient_function_L2(w_pre,x_train,y_train,M,lamda,c)
                
    return results,w_pre
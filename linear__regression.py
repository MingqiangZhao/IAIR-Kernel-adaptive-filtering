# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:51:41 2019

@author: Zhao Mingqiang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
from scipy import stats, optimize, interpolate

import L2_regularization as l2
import entropy_regularization as entropy


#实验结论：使用模板作为先验并不会降低误差，因为lamda值过大才会使结果更偏向于正则项的模板，如果太小，正则项影响又太小

#如果用先验模板去优化，其实本质上就相当于优化精度，而不是自适应！！！

#definition of global varibale

M=10000 #the amount of train data

N=12 #dimension of x 

iteration1 = 300 #iteration of gradient descent 
iteration2 = 50

alpha = 0.05 # learning rate
alpha1 = 0.05 # learning rate for entropy algorithm 1
alpha2 = 0.05 # learning rate for entropy algorithm 2

lamda = 0.5
lamda0 = 1000

lamda1 = 0.3 #lamda for entropy algorithm 1
lamda2 = 0.3 #lamda for entropy algorithm 2


sigma1 = 0.02 # sigma for entropy algorithm 1
sigma2 = 0.02 # sigma for entropy algorithm 2

low = -1
high = 1

#初始化权重
w = np.array([1,0,0,0,0,0,0,0,0,0,0,0])
w = w.reshape((1,N))

#定义模块

# w 向量维度 [1，N]，x_trian 维度 [N,M] M为训练数据量
def inicialize(w,M,N,low,high):
    #alpha stable distribution
    noise = sp.stats.levy_stable.rvs(1.2,0,0,0.2,[1,M])
    #noise = np.random.randn(1,M)
    
    #均匀分布初始化x
    x_train = np.random.uniform(low, high, size=[N,M])
  
    #初始化y
    y_train = np.dot(w,x_train)+noise
    
    return x_train, y_train

#main function
results_L2_1 = np.zeros((iteration2,iteration1))
results_L2_2 = np.zeros((iteration2,iteration1))
results_entropy1 = np.zeros((iteration2,iteration1))
results_entropy2 = np.zeros((iteration2,iteration1))

for i in range(iteration2):

    x_train, y_train = inicialize(w,M,N,low,high)
    
    w_pre = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
    w_pre = w_pre.reshape((1,N))
    
    #w1 = np.array([0.95,0.95,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.95,0.95])
    #w1 = w1.reshape((1,N))
    c = w
    
    results_L2_1[i], w_L2 = l2.gradient_decent_L2(x_train, y_train, M, w_pre, w, alpha, iteration1,lamda,0)
    results_L2_2[i], w_L21 = l2.gradient_decent_L2(x_train, y_train, M, w_pre, w, alpha, iteration1,lamda0, c)
    #results_entropy1[i], w_entropy1 = entropy.gradient_decent_entropy_1(x_train, y_train, M, w_pre, w, alpha1, iteration1, sigma1, lamda1)
    results_entropy2[i], w_entropy2 = entropy.gradient_decent_entropy_2(x_train, y_train, M, w_pre, w, alpha2, iteration1, sigma2, lamda2, c)
    
itera = range(iteration1)
results_L2_1 = np.mean(results_L2_1, axis=0)
results_L2_2 = np.mean(results_L2_2, axis=0)
results_entropy1 = np.mean(results_entropy1, axis=0)
results_entropy2 = np.mean(results_entropy2, axis=0)

'''
print(w_L2)
print(w_L21)
#print(w_entropy1)
print(w_entropy2)
'''

#绘图
plt.plot(itera, results_L2_1*10, color='blue', label='L2')
plt.plot(itera, results_L2_2*10, color='red', label='L2 template')
#plt.plot(itera, results_entropy1*10, color='black', label='Entropy1')
plt.plot(itera, results_entropy2*10, color='green', label='Entropy2')

plt.legend() # 显示图例

plt.xlabel('iteration times')
plt.ylabel('error')
plt.show()





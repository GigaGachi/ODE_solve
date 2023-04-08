import numpy as np
import matplotlib.pyplot as plt
import scipy 
import math
def func(u,t):
    y = np.array([u[1],t*t*u[0]],dtype=np.float32)  
    return y
def ODE_solve_euler(u0,M,t0,tend,f):
    #схема эйлера
    tau = (tend-t0)/M
    t = np.zeros(M)
    t[0] = t0
    shape_ = len(u0)
    u = np.zeros((M,shape_),np.float32)
    u[0] = u0
    for i in range (M-1):
        t[i+1] = t[i] + tau
        u[i+1] = u[i] + tau*f(u[i],t[i])
    return u,t
def ODE_solve_ERK2(u0,M,t0,tend,f):
    #схема рунге-кутты 2 порядка
    tau = (tend-t0)/M
    t = np.zeros(M)
    t[0] = t0
    shape_ = len(u0)
    u = np.zeros((M,shape_),np.float32)
    u[0] = u0
    for i in range (M-1):
        w1 = f(u[i],t[i])
        w2 = f(u[i]+tau*w1*(2/3),t[i]+(2/3)*tau)
        t[i+1] = t[i] + tau
        u[i+1] = u[i] + tau*((1/4)*w1+(3/4)*w2)
    return u,t
def ODE_solve_ERK4(u0,M,t0,tend,f):
    #схема рунге-кутты 4 порядка
    tau = (tend-t0)/M
    t = np.zeros(M)
    t[0] = t0
    shape_ = len(u0)
    u = np.zeros((M,shape_),np.float32)
    u[0] = u0
    for i in range (M-1):
        w1 = f(u[i],t[i])
        w2 = f(u[i]+tau*w1/2,t[i]+tau/2)
        w3 = f(u[i]+tau*w2/2,t[i]+tau/2)
        w4 = f(u[i]+tau*w3,t[i]+tau)
        t[i+1] = t[i] + tau
        u[i+1] = u[i] + tau*(w1+2*w2+2*w3+w4)/6
    return u,t

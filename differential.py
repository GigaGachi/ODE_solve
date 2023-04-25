import numpy as np
import matplotlib.pyplot as plt
import scipy 
import math
def func(u,t):
    #функция правой части уравнения
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
def f_to_len(u,t,f):
    dot = np.dot(f(u,t),f(u,t))
    vec = func(u,t)/(2+dot)**0.5
    return vec
def t_to_len(u,t,f):
    dot = np.dot(f(u,t),f(u,t))
    vec = 1/(2+dot)**0.5
    return vec
def ODE_solve_ERK4_qu(u0,t0,tend,f,tau):
    #схема эйлера
    t = []
    t.append(t0)
    shape_ = len(u0)
    u = []
    u.append(u0)
    while t[-1]<tend:
        w1 = f_to_len(u[-1],t[-1],f)
        w2 = f_to_len(u[-1]+tau*w1/2,t[-1]+tau/2,f)
        w3 = f_to_len(u[-1]+tau*w2/2,t[-1]+tau/2,f)
        w4 = f_to_len(u[-1]+tau*w3,t[-1]+tau,f)
        tw1 = t_to_len(u[-1],t[-1],f)
        tw2 = t_to_len(u[-1]+tau*tw1/2,t[-1]+tau/2,f)
        tw3 = t_to_len(u[-1]+tau*tw2/2,t[-1]+tau/2,f)
        tw4 = t_to_len(u[-1]+tau*tw3,t[-1]+tau,f)
        t.append(t[-1] + tau*(tw1+2*tw2+2*tw3+tw4)/6)
        u.append(u[-1] + tau*(w1+2*w2+2*w3+w4)/6)
    return u,t

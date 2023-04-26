import numpy as np
import matplotlib.pyplot as plt
import scipy 
from math import sqrt
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
def f_to_len(u,f):
    #замена правой части дифура для перехода к длине дуги
    t = u[-1]
    dot = np.dot(f(u,t),f(u,t))
    vec = f(u,t)/sqrt(1+dot)
    time = 1/sqrt(1+dot)
    return np.concatenate((vec,[time]),0)
def ODE_solve_ERK4_qu(u0,t0,tend,f,tau,Num_points):
    #рунге-кутта 4 порядка с квазиравномерной сеткой
    shape_ = len(u0)
    u_0 =  np.concatenate((u0,[t0]),0)
    u = np.zeros((Num_points,shape_+1),np.float32)
    u[0] =u_0
    i=0
    while u[i][-1]<tend:
        w1 = f_to_len(u[i],f)
        w2 = f_to_len(u[i]+tau*w1/2,f)
        w3 = f_to_len(u[i]+tau*w2/2,f)
        w4 = f_to_len(u[i]+tau*w3,f)
        u[i+1] = u[i] + tau*(w1+2*w2+2*w3+w4)/6
        i = i+1
    return u[0:i,0:shape_],u[0:i,-1]
def ODE_solve_ERK2_qu(u0,t0,tend,f,tau,Num_points):
    #рунге-кутта 2 порядка с квазиравномерной сеткой
    shape_ = len(u0)
    u_0 =  np.concatenate((u0,[t0]),0)
    u = np.zeros((Num_points,shape_+1),np.float32)
    u[0] =u_0
    i=0
    while u[i][-1]<tend:
        w1 = f_to_len(u[i],f)
        w2 = f_to_len(u[i]+tau*w1*(2/3),f)
        u[i+1] = u[i] + tau*((1/4)*w1+(3/4)*w2)
        i = i+1
    return u[0:i,0:shape_],u[0:i,-1]
def delta_l(y,t):
    #возвращает массив длин дуг разности двух соседних точек
    return [sqrt((y[i+1][0]-y[i][0])**2+(t[i+1]-t[i])**2) for i in range(len(t)-1)]
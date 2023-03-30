import numpy as np
import matplotlib.pyplot as plt
import scipy 
import math
def func(u,t):
    y = np.array([u[1],5*u[1]-4*u[0]+4*t*math.exp(2*t)],dtype=np.float32)  
    return y
def ODE_solve_euler(u0,M,t0,tend,f):
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



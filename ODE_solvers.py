import numpy as np
import matplotlib.pyplot as plt
import scipy 
from math import sqrt


def ODE_solve_euler(u0,M,t0,tend,f):
    """
    Решение обыкновенного дифференциального уравнения методом Эйлера
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях
        M: число точек
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши

    """
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
    """
    Решение обыкновенного дифференциального уравнения методом Рунге-Кутты 2 порядка
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях
        M: число точек
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши

    """
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
    """
    Решение обыкновенного дифференциального уравнения методом Рунге-Кутты 4 порядка
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях
        M: число точек
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши

    """
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
    """
    Переводит правую часть уравнения в задаче Коши к виду с квазиравномерной сеткой

    """
    t = u[-1]
    dot = np.dot(f(u,t),f(u,t))
    vec = f(u,t)/sqrt(1+dot)
    time = 1/sqrt(1+dot)
    return np.concatenate((vec,[time]),0)
def ODE_solve_ERK4_qu(u0,t0,tend,f,tau,Num_points):
    """
    Решение обыкновенного дифференциального уравнения методом Рунге-Кутты 4 порядка с квазиравномерной сеткой
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши
        tau: шаг по длине дуги
        Num_points: максимальное число точек

    """
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
    """
    Решение обыкновенного дифференциального уравнения методом Рунге-Кутты 2 порядка с квазиравномерной сеткой
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши
        tau: шаг по длине дуги
        Num_points: число точек

    """

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
    """
    Возвращает массив длин дуг между соседними точками по массиву значений и аргументов

    """
    return [sqrt((y[i+1][0]-y[i][0])**2+(t[i+1]-t[i])**2) for i in range(len(t)-1)]
def ODE_solve_ROS(u0,M,t0,tend,f,jac,alpha):
    """
    Решение обыкновенного дифференциального уравнения методом Розенброка 1 порядка 
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях
        M: число точек
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши
        jac: якобиан правой части уравнения
        alpha: коэфициент альфа в схеме розенброка

    """
    tau = (tend-t0)/M
    t = np.zeros(M)
    t[0] = t0
    shape_ = len(u0)
    u = np.zeros((M,shape_),np.float32)
    u[0] = u0
    E = np.eye(shape_)
    for i in range(M-1):
        A =  E - alpha*tau*jac(u[i],t[i])
        B = f(u[i],t[i] + tau/2)
        w = np.linalg.solve(A,B).real
        u[i+1] = u[i] + tau*w[0:shape_]
        t[i+1] = t[i] + tau
    return u,t

def ODE_solve_ROS_algebraic(u0,M,t0,tend,f,jac,alpha):
    """
    Решение обыкновенного дифференциально-алгебраического уравнения методом Розенброка 1 порядка 
    
    Args:
        u0: значение вектор-функции или скалярной функции при данных начальных условиях с учетом алгебраической замены
        M: число точек
        t0: значение переменной при данных начальных условия
        tend: значение переменной конца интервала решения
        f: функция правой части уравнения в задаче Коши с учетом алгебраической замены
        jac: якобиан правой части уравнения по переменной-вектору с учетом алгебраической замены
        alpha: коэфициент альфа в схеме розенброка

    """
    tau = (tend-t0)/M
    t = np.zeros(M)
    t[0] = t0
    shape_ = len(u0)
    u = np.zeros((M,shape_),np.float32)
    u[0] = u0
    D = np.eye(shape_)
    D[-1][-1] = 0
    for i in range(M-1):
        A = D - alpha*tau*jac(u[i],t[i])
        B = f(u[i],t[i] + tau/2)
        w = np.linalg.solve(A,B).real
        u[i+1] = u[i] + tau*w[0:shape_]
        t[i+1] = t[i] + tau
    return u,t

def DichotomyMethod(f,eps,N_max,x_l,x_r):
    """
    Решение нелинейного уравнения методом дихотомии
    
    Args:
        f: функция 
        eps: точность корня
        N_max: максимальное число итераций
        x_l: левое начальное приближение
        x_r: правое начальное приближение

    """
    gamma = np.zeros(N_max)
    function_value = np.zeros(N_max)
    gamma[0] = x_l
    gamma[1] = x_r
    function_value[0] = f(gamma[0])
    function_value[1] = f(gamma[1])
    s = 1
    while abs(gamma[s] - gamma[s-1])>eps:
        gamma[s+1] = (gamma[s] + gamma[s-1])/2
        function_value[s+1] = f(gamma[s+1])
        if function_value[s+1]*function_value[s-1] < 0:
            gamma[s] = gamma[s-1]
            function_value[s] = function_value[s-1]
        elif function_value[s+1] == 0:
            s = s+1
            break
        s = s + 1
    return gamma[s]
class EDE_solver_ROS_DCH:
    """
    Класс, реализующий решение краевой задачи методом розенброка и дихотомии алгоритмом стрельбы
    
    """
    def __init__(self,ul,ur,M,t0,tend,f,jac,alpha,eps,N_max,x_l,x_r):
        """
        Инициализация параметров алгоритма 
    
        Args:
            ul: значение вектор-функции или скалярной функции на левом конце
            ur: значение вектор-функции или скалярной функции на правом конце
            M: число точек
            t0: значение переменной при данных начальных условия
            tend: значение переменной конца интервала решения
            f: функция правой части уравнения в задаче Коши
            jac: якобиан правой части уравнения с учетом автономизации
            alpha: коэфициент альфа в схеме розенброка
            f: функция 
            eps: точность корня
            N_max: максимальное число итераций
            x_l: левое начальное приближение
            x_r: правое начальное приближение
        """
        self.ul = ul
        self.ur = ur
        self.M = M
        self.t0 = t0
        self.tend = tend
        self.f = f
        self.jac = jac
        self.alpha = alpha
        self.eps = eps
        self.N_max = N_max
        self.x_l = x_l
        self.x_r = x_r
    def f_gamma(self,gamma):
        u,t = ODE_solve_ROS(np.array([self.ul,gamma]),self.M,self.t0,self.tend,self.f,self.jac,self.alpha)
        return (u[-1][0] - self.ur)
    def get_gamma(self):
        """
        Вычисляет значение гамма в методе стрельбы
        
        """
        gamma = DichotomyMethod(self.f_gamma,self.eps,self.N_max,self.x_l,self.x_r)
        return gamma
    def solve(self,gamma):
        """
        Решение обыкновенного дифференциального уравнения методом Розенброка 1 порядка 
    
        Args:
        gamma: найденное значение гамма в методе стрельбы

        """
        u,t = ODE_solve_ROS(np.array([self.ul,gamma]),self.M,self.t0,self.tend,self.f,self.jac,self.alpha)
        return u,t

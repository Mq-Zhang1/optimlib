import numpy as np
import math
import sympy as sy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize
from utils import *
'''---------------------Descent Methods--------------------'''

''' Implement Gradient Descent Algorithm(require learning rate)'''
class Gradient(object):
    def __init__(self,lr=0.001): 
        '''
        :param lr: learning rate [float]
        '''
        self.lr = lr
    def setlr(self,lr): 
        # used for linear search
        self.lr = lr
    def step(self,gradient):
        '''
        :param gradient: array, gradient of the function
        '''
        return self.lr*gradient

''' Implement Newton Direction Descent Algorithm'''
class Newton(object):
    def __init__(self): self.name = "newton"
    def step(self,gradient,heissian):
        '''
        :param gradient: array, first-order derivative of the function
        :param heissian: array, second-order derivative of the function
        '''
        eig = np.linalg.eigvals(heissian)
        # judge whether invertible
        if np.any(eig)<=0: 
            raise Exception("Heissian is not positive definite")
        else:
            direction = np.dot(np.linalg.inv(heissian),gradient)
        return direction

''' Implement Momentum Direction,different from above, use directly by x-direction'''
class Momentum(object):
    def __init__(self,velocity=0.9,lr=0.01):
        ''' 
        :param velocity: constant
        :param lr: learning rate
        '''
        self.beta = velocity
        self.lr = lr
        self.dir = 0
    def step(self,gradient):
        '''
        :param gradient: current gradient value
        '''
        self.dir = self.beta*self.dir+self.lr*gradient
        return self.dir

''' Implement Nesterov Accelerated Gradient,different from Momentum, gradient is not of the current point,
    but x-velocity*predir, different from others'''
class Nesterov(object):
    def __init__(self,velocity,lr):
        '''
        :param velocity: constant
        :param lr: learning rate
        '''
        self.vt = 0
        self.beta = velocity
        self.lr = lr
    def step(self,x,point,gradFunc):
        '''
        :param x: symbolic x
        :param point: array
        :param gradFunc: gradient function
        '''
        delta = self.beta*self.vt
        dfunc = getGrad(x,point-delta,gradFunc)
        self.vt = delta+self.lr*dfunc
        return self.vt


''' Implement Adagrad'''
#https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827
class Adagrad(object):
    def __init__(self,lr=0.01,epsilon=1e-6):
        '''
        :param lr: learning rate #very important hyperparameter
        :param epsilon: constant for avoiding dividing 0
        '''
        self.lr = lr
        self.Gt = epsilon
    def step(self,gradient):
        '''
        :param gradient: first-order derivative of the function
        '''
        self.Gt = self.Gt+np.square(gradient)
        Gt_1 = 1/np.sqrt(self.Gt)
        direction = self.lr*Gt_1*gradient
        return direction

''' Implement RMSprop'''
class Rmsprop(object):
    def __init__(self,beta=0.9,epsilon=1e-7,lr=0.5):
        '''
        :param beta: Exponential decay rate for past gradient info
        :param epsilon: constant for avoiding dividing 0
        :param lr: learning rate
        '''
        self.lr = lr
        self.Gt = epsilon
        self.beta = beta
    
    def step(self,gradient):
        '''
        :param gradient: first-order derivative of the function
        '''
        self.Gt = self.beta*self.Gt+(1-self.beta)*np.square(gradient)
        Gt_1 = 1/np.sqrt(self.Gt)
        direction = self.lr*Gt_1*gradient
        return direction

'''Implement AdaDelta'''
class AdaDelta(object):
    def __init__(self,beta=0.5,epsilon=1e-7):
        '''
        :param beta, epsilon: [float] related parameters
        '''
        self.beta = beta
        self.Gt = epsilon
        self.dx = epsilon
    def step(self,gradient):
        '''
        :param gradient: first-order derivative of the function
        '''
        self.dx = self.beta*self.dx+(1-self.beta)*np.square(gradient)
        ratio = self.Gt/self.dx
        direction = np.sqrt(ratio)*gradient
        self.Gt = self.beta*self.Gt+(1-self.beta)*np.square(direction)
        return direction

'''Implement Adam Algorithm'''
class Adam(object):
    def __init__(self,velocity = 0.9,beta=0.99,epsilon=1e-7,lr=0.01): #velocity = 0.7 beta = 0.9
        '''
        :param velocity,beta,epsilon: [float] algo related parameters
        :param lr: [float] learning rate
        '''
        self.v_t=0
        self.Gt = 0
        self.epsilon = epsilon
        self.lr = lr
        self.beta = beta
        self.velocity = velocity
        self.t = 1
    def step(self,gradient):
        '''
        :param gradient: [array] first-order derivative of the function
        '''
        self.v_t = self.velocity*self.v_t+(1-self.velocity)*gradient
        self.Gt = self.beta*self.Gt+(1-self.beta)*np.square(gradient)
        v_hat = self.v_t/(1-self.velocity**self.t)
        G_hat = self.Gt/(1-self.beta**self.t)
        direction = self.lr/np.sqrt(G_hat+self.epsilon)*v_hat
        self.t = self.t+1
        return direction

'''Implement AdaMax Algorithm'''
class AdaMax(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.01):
        '''
        :param alpha,beta: [float] related parameters
        :param lr: [float] learning rate
        '''
        self.S = 0
        self.Rt = 0
        self.t = 1
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
    def step(self,gradient):
        '''
        :param gradient: [array] first-order derivative of the function
        '''
        self.S = self.alpha*self.S+(1-self.alpha)*gradient
        St = self.S/(1-self.alpha**self.t)
        self.Rt = np.maximum(self.beta*self.Rt,np.abs(gradient))
        direction = self.lr*St/self.Rt
        self.t = self.t+1
        return direction

'''Implement Nadam Algorithm'''
class Nadam(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.01,epsilon=1e-7):
        '''
        :param alpha,beta,epsilon: [float] algo related parameters
        :param lr: [float] learning rate
        '''
        self.St = 0
        self.Rt = epsilon
        self.a = 1
        self.b = 1
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.t = 1
    def step(self,gradient):
        '''
        :param gradient: [array] first-order derivative of the function
        '''
        self.a = self.a*self.alpha
        self.b = self.b*self.beta
        self.t = self.t+1
        self.St = self.alpha*self.St+(1-self.alpha)*gradient
        self.Rt = self.beta*self.Rt+(1-self.beta)*np.square(gradient)
        direction = self.lr*math.sqrt(1-self.b)/(1-self.a)*(self.alpha*self.St+(1-self.alpha)*gradient)/np.sqrt(self.Rt)
        return direction

'''Implement Nadamax Algorithm'''
class Nadamax(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.01):
        '''
        :param alpha,beta: [float] algo related parameters
        :param lr: [float] learning rate
        '''
        self.alpha = alpha
        self.a = 1
        self.beta = beta
        self.lr = lr
        self.St = 0
        self.Rt = 0
    
    def step(self,gradient):
        '''
        :param gradient: [array] first-order derivative of the function
        '''
        self.a = self.a*self.alpha
        self.St = self.alpha*self.St+(1-self.alpha)*gradient
        self.Rt = np.maximum(self.beta*self.Rt,np.abs(gradient))
        direction = self.lr/(1-self.a)*(self.alpha*self.St+(1-self.alpha)*gradient)/self.Rt
        return direction

'''This is for fixed learning rate, above optimization methods all can used'''
def update(x,init,func,thresh,method,param,grad_func,heissian_func=[]):
    '''
    :param init: [array] initial point
           func: symbolic function
           thresh: [float] error threshod
           method: optimization class such as Newton, Gradient etc
           param: [tuple] required parameters for initialize class
           grad_func: symbolic gradient function
           heissian_fun: symbolic heissian function

    :return point_list: [list] updated point path len([[x1 x2 x3...xn],...,]) = iteration_num
            value: [list] all the function value along the path len([v0,v1,...]) = iteration_num
    '''
    point = init
    func_value = getFunc(x,point,func) #get initial function value
    diff_value = getGrad(x,point,grad_func) #get initial grad function
    heissian_value = getHeissian(x,point,heissian_func)
    point_list = [point] 
    value = [func_value]
    error = np.linalg.norm(diff_value)
    if param: opt = method(*param) # whether have initialization
    else: opt = method()
    while (error>thresh):
        if method == Newton:
            direction = opt.step(diff_value,heissian_value)
        elif method == Nesterov:
            direction = opt.step(x,point,grad_func)
        else:
            direction = opt.step(diff_value)
        point = point-direction
        point_list.append(point)
        value.append(getFunc(x,point,func))
        diff_value = getGrad(x,point,grad_func)
        error = np.linalg.norm(diff_value)
    return point_list, value #(point[array],value[float])

'''
Implement linear seach algorithm with backtracking(Armijo condition)
'''
def linear_search(x,init,thresh,func,grad_func):
    '''
    :param  x: symbolic x
            init: intial point
            thresh: error threshold
            func: symbolic function
            grad_func: symbolic gradient
    :return point_list: [list] updated point path len([[x1 x2 x3...xn],...,]) = iteration_num
            value: [list] all the function value along the path len([v0,v1,...]) = iteration_num
    '''
    # initialization
    point = init
    point_list = [point]
    func_value = getFunc(x,point,func)
    diff_value = getGrad(x,point,grad_func)
    error = np.linalg.norm(diff_value)
    value = [func_value]
    alpha = 0.1
    sigma = 0.8
    cparam = 0.5
    # suitable for using gradient descent
    opt = Gradient(alpha)
    while (error>thresh):
        direction = opt.step(diff_value)
        tmp = point-direction
        # Armijo condition to choose learning rate
        if getFunc(x,tmp,func)>value[-1]-cparam*np.sum(diff_value*direction): 
            alpha = alpha*sigma
            opt.setlr(alpha)
            continue
        point = tmp
        point_list.append(point)
        value.append(getFunc(x,point,func))
        diff_value = getGrad(x,point,grad_func)
        error = np.linalg.norm(diff_value)
        alpha = 0.1
    return point_list,value

'''
Implement CG method
'''
def CG(x,init,func,grad,heissian,methods='Newton-CG'):
    '''
    :param  x: [array] sybolic x
            init: [array] initial point
            func: [array] symbolic function
            grad: [array] symbolic grad function
            heissian: [array] heissian function matrix
            methods: support "Newton-CG","trust-ncg" "trust-krylov"
    :return res: final optimal result
    '''
    f = lambdify([tuple(x)],func)
    derivative = lambdify([tuple(x)],grad)
    hess = lambdify([tuple(x)],heissian)
    res = minimize(f, init, method=methods,
                jac=derivative, hess=hess,
                options={'xtol': 1e-8, 'disp': True})
    return res

'''
Implement TrustRegion method
'''
def TrustRegion(x,init,func,grad,heissian,methods="trust-exact"):
    '''
    :param methods: support "trust-exact","dogleg"
    '''
    f = lambdify([tuple(x)],func)
    derivative = lambdify([tuple(x)],grad)
    hess = lambdify([tuple(x)],heissian)
    res = minimize(f, init, method=methods,
                jac=derivative, hess=hess,
                options={'xtol': 1e-8, 'disp': True})
    return res

'''
Function for testing CG method using simple test function x1^2+x2^2, could define yours here
'''
def test_CG():
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([1,0])
    function = (symbol[0])**2+(symbol[1])**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
    CG(symbol,init,function,diff1,diff2)

'''
Function for testing trust-region method using simple test function x1^2+x2^2, could define yours here
'''
def test_tr():
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([1,0])
    function = (symbol[0])**2+(symbol[1])**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
    TrustRegion(symbol,init,function,diff1,diff2,"dogleg")

'''
Function for testing line search method using simple test function x1^2+x2^2, could define yours here
'''
def test_line(): #using backtracking line search
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([1,0])
    thresh = 0.01
    # build 2D function
    function = (symbol[0])**2+(symbol[1])**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    plt.figure()
    _,value = linear_search(symbol,init,thresh,function,diff1)
    print("Linear search final value:",value[-1])
    print("Linear Search iteration times:",len(value))

'''
Function for testing different optimization class and compare convergence rate
'''
def test_update():
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([2,2])
    thresh = 0.01
    # build 2D function
    function = symbol[0]**2-symbol[0]*symbol[1]+3*symbol[1]**2+5
    # build test function Rosenbrock
    # function = (1-symbol[0])**2+100*(symbol[1]-symbol[0]**2)**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
    figure = plt.figure()
    X1 = np.arange(-5,5,0.05)
    X2 = np.arange(-5,5,0.05)
    X1,X2 = np.meshgrid(X1,X2)
    Z = X1**2-X1*X2+3*X2**2+5
    # Z = (1-X1)*(1-X1)+100*(X2-X1*X1)*(X2-X1*X1)
    contour1 = plt.contour(X1,X2,Z)
    plt.plot([2],[2],'o') # initial
    plt.plot([0],[0],"x") # optimal
    value = dict()
    # 11 algorithms
    _type = ['gradient','Newton','Momentum','Nesterov','Adagrad','RMSPROP']
    color = dict()
    _color = ['r','g','b','y','c','m']
    counter = dict()
    value['gradient'] = update(symbol,init,function,thresh,Gradient,(0.01,),diff1)
    value['Newton'] = update(symbol,init,function,thresh,Newton,(),diff1,diff2)
    value['Momentum'] = update(symbol,init,function,thresh,Momentum,(0.9,0.001),diff1)
    value['Nesterov'] = update(symbol,init,function,thresh,Nesterov,(0.9,0.001),diff1)
    value['Adagrad'] = update(symbol,init,function,thresh,Adagrad,(0.1,),diff1)
    value['RMSPROP'] = update(symbol,init,function,thresh,Rmsprop,(0.9,1e-7,0.1),diff1)
    for key in _type:
        counter[key] = len(value[key][1])-1
        color[key] = _color[_type.index(key)]
    max_len = max(counter.values()) # max value
    for i in range(max_len):
        for key in _type:
            if counter[key] == 0: continue
            x=[value[key][0][i][0],value[key][0][i+1][0]]
            y=[value[key][0][i][1],value[key][0][i+1][1]]
            #z = [value[key][1][i],value[key][1][i+1]]
            plt.plot(x,y,color[key],label = key)
            counter[key]-=1 #TODO: improve visualization, too ugly now
        if i == 0:
            plt.legend()
        plt.pause(0.01)
    plt.show()

    '''for test: could test following algo and see the iteration plot'''
    # value['AdaDelta'] = update(symbol,init,function,thresh,AdaDelta,(0.9,1e-7),diff1)
    # value['Adam'] = update(symbol,init,function,thresh,Adam,(),diff1)
    # value['AdaMax'] = update(symbol,init,function,thresh,AdaMax,(),diff1)
    # value['Nadam'] = update(symbol,init,function,thresh,Nadam,(),diff1)
    # value['Nadamax'] = update(symbol,init,function,thresh,Nadamax,(),diff1)
    # ite = [i for i in range(len(value['Adagrad']))]
    # plt.plot(ite,value['Adagrad'],'r--')

import argparse
parser = argparse.ArgumentParser(description='Descent')
parser.add_argument('--test', '-t', dest="test", type=int, default=0, \
    help='1: test line search, \
          2: test trust region, \
          3: test CG default: Newton-CG,\
          4: test descent methods such as gd, momentum on certain function'
)
args = parser.parse_args()

if __name__  == "__main__":
    if args.test == 1:
        test_line()
    elif args.test == 2:
        test_tr()
    elif args.test == 3:
        test_CG()
    else:
        test_update()

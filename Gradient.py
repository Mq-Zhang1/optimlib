import numpy as np
import math
import sympy as sy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities.lambdify import lambdify
from scipy.optimize import minimize
'''---------------------Descent Methods--------------------'''

''' Implement Gradient Descent Algorithm(require learning rate)
    Args: 
        -gradient: array, gradient of the function
        -lr: float, learning rate
'''
class Gradient(object):
    def __init__(self,lr=0.001): self.lr = lr
    def setlr(self,lr): self.lr = lr
    # return one step
    def step(self,gradient):
        #print(gradient)
        return self.lr*gradient

''' Implement Newton Direction Descent Algorithm
    Args: 
        - gradient: array, first-order derivative of the function
        - heissian: array, second-order derivative of the function
'''
class Newton(object):
    def __init__(self): self.name = "newton"
    def step(self,gradient,heissian): #return 1 step
        eig = np.linalg.eigvals(heissian)
        if np.any(eig)<=0:
            raise("Heissian is not positive definite")
        else:
            direction = np.dot(np.linalg.inv(heissian),gradient)
        return direction

''' Implement Momentum Direction,different from above, use directly by 
    x-direction
    Args: 
        - predir: the direction of last step
        - gradient: first-order derivative of the function
        - velocity: constant
        - lr: learning rate
    Returns:
        - momentum direction
'''
class Momentum(object):
    def __init__(self,velocity=0.9,lr=0.01):
        self.beta = velocity
        self.lr = lr
        self.dir = 0
    def step(self,gradient):
        self.dir = self.beta*self.dir+self.lr*gradient
        return self.dir

''' Implement Nesterov Accelerated Gradient,different from Momentum, gradient is not of the current point,
    but x-velocity*predir, different from others
    Args: 
        - predir: the direction of last step
        - point: array
        - velocity: constant
        - lr: learning rate
    Returns:
        - Nesterov direction
'''
class Nesterov(object):
    def __init__(self,velocity,lr):
        self.vt = 0
        self.beta = velocity
        self.lr = lr
    def step(self,x,point,gradFunc):
        delta = self.beta*self.vt
        dfunc = getGrad(x,point-delta,gradFunc)
        self.vt = delta+self.lr*dfunc
        return self.vt


''' Implement Adagrad
    Args: 
        - gradient: first-order derivative of the function
        - epsilon: constant for avoiding dividing 0
        - lr: learning rate #very important  cannot be too small
    Returns:
        - adagrad direction
'''
#https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827
class Adagrad(object):
    def __init__(self,epsilon=1e-6,lr=0.8):
        self.lr = lr
        self.Gt = epsilon
    def step(self,gradient):
        self.Gt = self.Gt+np.square(gradient)
        Gt_1 = 1/np.sqrt(self.Gt)
        direction = self.lr*Gt_1*gradient
        return direction

''' Implement RMSprop
    Args: 
        - beta: Exponential decay rate for past gradient info
        - gradient: first-order derivative of the function
        - epsilon: constant for avoiding dividing 0
        - lr: learning rate
    Returns:
        - adagrad direction
'''  
class Rmsprop(object):
    def __init__(self,beta=0.9,epsilon=1e-7,lr=0.5):
        self.lr = lr
        self.Gt = epsilon
        self.beta = beta
    
    def step(self,gradient):
        self.Gt = self.beta*self.Gt+(1-self.beta)*np.square(gradient)
        Gt_1 = 1/np.sqrt(self.Gt)
        direction = self.lr*Gt_1*gradient
        return direction

'''Implement AdaDelta
   Args:
       -beta, epsilon: float, related parameters
       -gradient: first-order derivative of the function
   Returns:
       -step: 1 step forward
'''
class AdaDelta(object):
    def __init__(self,beta=0.5,epsilon=1e-7):
        self.beta = beta
        self.Gt = epsilon
        self.dx = epsilon
    def step(self,gradient):
        self.dx = self.beta*self.dx+(1-self.beta)*np.square(gradient)
        ratio = self.Gt/self.dx
        direction = np.sqrt(ratio)*gradient
        self.Gt = self.beta*self.Gt+(1-self.beta)*np.square(direction)
        return direction

'''Implement Adam Algorithm
   Args:
       -velocity,beta,epsilon: float, algo related parameters
       -lr: float, learning rate
       -gradient: array, first-order derivative of the function
   Returns:
       -step: 1 step forward
'''
class Adam(object):
    def __init__(self,velocity = 0.9,beta=0.99,epsilon=1e-7,lr=0.01): #velocity = 0.7 beta = 0.9
        self.v_t=0
        self.Gt = 0
        self.epsilon = epsilon
        self.lr = lr
        self.beta = beta
        self.velocity = velocity
        self.t = 1
    def step(self,gradient):
        self.v_t = self.velocity*self.v_t+(1-self.velocity)*gradient
        self.Gt = self.beta*self.Gt+(1-self.beta)*np.square(gradient)
        v_hat = self.v_t/(1-self.velocity**self.t)
        G_hat = self.Gt/(1-self.beta**self.t)
        direction = self.lr/np.sqrt(G_hat+self.epsilon)*v_hat
        self.t = self.t+1
        return direction

'''Implement AdaMax Algorithm
   Args:
       -alpha,beta: float, related parameters
       -lr: float, learning rate
       -gradient: array, first-order derivative of the function
   Returns:
       -step: 1 step forward
'''
class AdaMax(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.01):
        self.S = 0
        self.Rt = 0
        self.t = 1
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
    def step(self,gradient):
        self.S = self.alpha*self.S+(1-self.alpha)*gradient
        St = self.S/(1-self.alpha**self.t)
        self.Rt = np.maximum(self.beta*self.Rt,np.abs(gradient))
        direction = self.lr*St/self.Rt
        self.t = self.t+1
        return direction

'''Implement Nadam Algorithm
   Args:
       -alpha,beta,epsilon: float, algo related parameters
       -lr: float, learning rate
       -gradient: array, first-order derivative of the function
   Returns:
       -step: 1 step forward
'''
class Nadam(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.01,epsilon=1e-7):
        self.St = 0
        self.Rt = epsilon
        self.a = 1
        self.b = 1
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.t = 1
    def step(self,gradient):
        self.a = self.a*self.alpha
        self.b = self.b*self.beta
        self.t = self.t+1
        self.St = self.alpha*self.St+(1-self.alpha)*gradient
        self.Rt = self.beta*self.Rt+(1-self.beta)*np.square(gradient)
        direction = self.lr*math.sqrt(1-self.b)/(1-self.a)*(self.alpha*self.St+(1-self.alpha)*gradient)/np.sqrt(self.Rt)
        return direction

'''Implement Nadamax Algorithm
   Args:
       -alpha,beta: float, algo related parameters
       -lr: float, learning rate
       -gradient: array, first-order derivative of the function
   Returns:
       -step: 1 step forward
'''
class Nadamax(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.01):
        self.alpha = alpha
        self.a = 1
        self.beta = beta
        self.lr = lr
        self.St = 0
        self.Rt = 0
    
    def step(self,gradient):
        self.a = self.a*self.alpha
        self.St = self.alpha*self.St+(1-self.alpha)*gradient
        self.Rt = np.maximum(self.beta*self.Rt,np.abs(gradient))
        direction = self.lr/(1-self.a)*(self.alpha*self.St+(1-self.alpha)*gradient)/self.Rt
        return direction


# Establish a new function for test
'''
symbol: array
'''
def test_func(symbol,type):
    function = 0
    diff1 = 0
    diff2 = 0
    if type == "sphere":
        func = np.sum(symbol*symbol)
        diff1 = 2*symbol
        diff2 = np.eye(symbol.size)*2
    return func,diff1,diff2

'''Return function value of single point
   Args:
        - point: 1D array of point
        - func: symbolic function
    Returns:
        - value: function value of this point
'''
def getFunc(x,point,func):
    dim = point.size
    func_value = func
    for i in range(dim):
        func_value = func_value.subs(x[i],point[i])
    return float(func_value)

'''Return the first order gradient of single point
   Args:
        - point: 1D array of point
        - gradient: symbolic function of gradient
    Returns:
        - value: gradient value of this point
'''
def getGrad(x,point,grad):
    dim = point.size
    path = [0]*dim
    for i in range(dim):
        path[i] = getFunc(x,point,grad[i])
    return np.array(path)

'''Return Hessian Matrix of single point
   Args:
        - point: 1D array of point
        - heissian: symbolic function of heissian matrix
   Returns:
        - path: return heissian value of this point
'''
def getHeissian(x,point,heissian):
    if heissian == None: return None
    dim = point.size
    path = np.ones((dim,dim))
    for i in range(dim):
        path[i] = getGrad(x,point,heissian[i])
    return path
'''This is for fixed learning rate, above opt methods all can used
init: array
x: array
param = param tuple
'''
def update(x,init,func,thresh,method,param,grad_func,heissian_func=None):
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
    point = init
    point_list = [point]
    func_value = getFunc(x,point,func)
    diff_value = getGrad(x,point,grad_func)
    error = np.linalg.norm(diff_value)
    value = [func_value]
    alpha = 0.1
    sigma = 0.8
    cparam = 0.5
    opt = Gradient(alpha)
    while (error>thresh):
        direction = opt.step(diff_value)
        tmp = point-direction
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
x: sybolic array
init: initial point array
func: function array
grad: grad function array
heissian: heissian dunction matrix
methods: support "Newton-CG","trust-ncg" "trust-krylov"
method:
'''

def CG(x,init,func,grad,heissian,methods='Newton-CG'):
    f = lambdify([tuple(x)],func)
    derivative = lambdify([tuple(x)],grad)
    hess = lambdify([tuple(x)],heissian)
    res = minimize(f, init, method=methods,
                jac=derivative, hess=hess,
                options={'xtol': 1e-8, 'disp': True})
    return res
    #print(f(*tuple(init)))
'''
Implement TrustRegion method
methods: support "trust-exact","dogleg"
'''
def TrustRegion(x,init,func,grad,heissian,methods="trust-exact"):
    f = lambdify([tuple(x)],func)
    derivative = lambdify([tuple(x)],grad)
    hess = lambdify([tuple(x)],heissian)
    res = minimize(f, init, method=methods,
                jac=derivative, hess=hess,
                options={'xtol': 1e-8, 'disp': True})
    return res
def test_CG():
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([1,0])
    function = (symbol[0])**2+(symbol[1])**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
    CG(symbol,init,function,diff1,diff2)
def test_tr():
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([1,0])
    function = (symbol[0])**2+(symbol[1])**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
    TrustRegion(symbol,init,function,diff1,diff2,"dogleg")
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

def main():
    symbol = np.array(sy.symbols('x1 x2'))
    init = np.array([1,0])
    thresh = 0.1
    # build 2D function
    function = (symbol[0])**2+(symbol[1])**2
    # build test function Rosenbrock
    #function = (1-symbol[0])**2+100*(symbol[1]-symbol[0]**2)**2
    diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
    diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
    figure = plt.figure()
    ax = Axes3D(figure)
    X1 = np.arange(-0.5,0.5,0.05)
    X2 = np.arange(-1,2,0.05)
    X1,X2 = np.meshgrid(X1,X2)
    Z = X1*X1+X2*X2
    #Z = (1-X1)*(1-X1)+100*(X2-X1*X1)*(X2-X1*X1)
    ax.plot_surface(X2,X1,Z,cmap = "rainbow")
    value = dict()
    # 11 algorithms
    _type = ['gradient','Momentum','Nesterov']
    color = dict()
    _color = ['r','g','b']
    #_type = ['gradient']
    counter = dict()
    value['gradient'] = update(symbol,init,function,thresh,Gradient,(0.001,),diff1)
    print(1)
    # value['Newton'] = update(symbol,init,function,thresh,Newton,(),diff1,diff2)
    value['Momentum'] = update(symbol,init,function,thresh,Momentum,(0.9,0.001),diff1)
    value['Nesterov'] = update(symbol,init,function,thresh,Nesterov,(0.9,0.001),diff1)
    #print(value["gradient"])
    for key in _type:
        counter[key] = len(value[key][1])-1
        color[key] = _color[_type.index(key)]
    max_len = max(counter.values()) # max value
    for i in range(max_len):
        for key in _type:
            if counter[key] == 0: continue
            x=[value[key][0][i][0],value[key][0][i+1][0]]
            y=[value[key][0][i][1],value[key][0][i+1][1]]
            z = [value[key][1][i],value[key][1][i+1]]
            ax.plot3D(x,y,z,color[key],linewidth = 2)
            counter[key]-=1 #TODO: improve visualization, too ugly now
        plt.pause(0.01)
    plt.show()
    '''for test: could test following algo and see the plot'''
    # value['Adagrad'] = update(symbol,init,function,thresh,Adagrad,(),diff1)
    # value['RMSPROP'] = update(symbol,init,function,thresh,Rmsprop,(0.9,1e-7,1),diff1) #not good
    # value['AdaDelta'] = update(symbol,init,function,thresh,AdaDelta,(0.9,1e-7),diff1)
    # value['Adam'] = update(symbol,init,function,thresh,Adam,(),diff1)
    # value['AdaMax'] = update(symbol,init,function,thresh,AdaMax,(),diff1)
    # value['Nadam'] = update(symbol,init,function,thresh,Nadam,(),diff1)
    # value['Nadamax'] = update(symbol,init,function,thresh,Nadamax,(),diff1)
    #print(value['Adagrad'])
    # ite = [i for i in range(len(value['Adagrad']))]
    # plt.plot(ite,value['Adagrad'],'r--')

if __name__  == "__main__":
    test_tr()

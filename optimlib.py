from functools import partialmethod
import numpy as np
import math
import random

''' Implement Gradient Descent Algorithm
    Args: 
        -gradient: gradient of the function
    Returns:
        -descent direction
'''
class Gradient(object):
    def __init__(self,lr): self.lr = lr
    def step(self,gradient):
        return self.lr*gradient

''' Implement Newton Direction Descent Algorithm
    Args: 
        - gradient: first-order derivative of the function
        - heissian: second-order derivative of the function
    Returns:
        - newton direction
'''
class Newton(object):
    def __init__(self): self.name = "newton"
    def step(self,gradient,heissian):
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
'''
velocity =0.7
beta = 0.9
'''
class Adam(object):
    def __init__(self,velocity = 0.9,beta=0.99,epsilon=1e-7,lr=0.01):
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

class Nadamax(object):
    def __init__(self,alpha=0.9,beta=0.9,lr=0.1):
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

def test_func(symbol,type):
    function = 0
    diff1 = 0
    diff2 = 0
    if type == "sphere":
        func = np.sum(symbol*symbol)
        diff1 = 2*symbol
        diff2 = np.eye(symbol.size)*2
    return func,diff1,diff2
'''
get symbolic function value
'''
def getFunc(x,point,func):
    dim = point.size
    func_value = func
    for i in range(dim):
        func_value = func_value.subs(x[i],point[i])
    return float(func_value)

def getGrad(x,point,grad):
    dim = point.size
    path = [0]*dim
    for i in range(dim):
        path[i] = getFunc(x,point,grad[i])
    return np.array(path)

def getHeissian(x,point,heissian):
    dim = point.size
    path = np.ones((dim,dim))
    for i in range(dim):
        path[i] = getGrad(x,point,heissian[i])
    return path
'''
init: array
x: array
'''
def update(x,init,func,thresh,grad_func,heissian_func=None):
    point = init
    func_value = getFunc(x,point,func) #get initial function value
    diff_value = getGrad(x,point,grad_func) #get initial grad function
    heissian_value = heissian_func
    value = [func_value]
    error = np.linalg.norm(diff_value)
    opt = Momentum()
    while (error>thresh):
        direction = opt.step(diff_value)
        point = point-direction
        value.append(getFunc(x,point,func))
        diff_value = getGrad(x,point,grad_func)
        error = np.linalg.norm(diff_value)
    return value

''' ---------------------Stochastic Searching-------------------------'''
'''
Implement Simulated Annealing
x array
init array
method (name,related_param)
'''
def SimulatedAnnealing(x,init,temp,func,method,thresh):
    f_mean = init
    var = np.eye(init.shape[0])
    value = [getFunc(x,init,func)]
    error = 100
    k = 1 #current time
    while (error>thresh):
        x_new = np.random.multivariate_normal(mean = f_mean,cov = var)
        f_new = getFunc(x,x_new,func)
        error = f_new-value[-1]
        if error>=0:
            if method[0] == "Fast":
                cooling = temp/k
            elif method[0] == "Exponential":
                if k == 1: cooling = temp
                else: cooling = method[1]*cooling
            elif method[0] == "Log":
                cooling = temp*math.log(2)/math.log(k+1)
            else:
                cooling = temp/k
            if random.uniform(0,1)<math.exp((-error)/cooling):
                f_mean = x_new #accept
            else: continue
        else:
            f_mean = x_new
            error = -error
        k+=1
        value.append(f_new)
    return value

'''
Implement Cross Entropy
x array
init array
point_num 
rho
'''
def CrossEntropy(x,init,point_num,rho,func,thresh):
    f_mean = init
    var = np.eye(init.shape[0])
    value = [getFunc(x,init,func)]
    error = 1
    elite_num = int(rho*point_num)
    while(error<thresh):
        f_new = {} # all the new sample values
        temp_mean = 0 # mean value of samples' function value
        x_sample = [] # x_values
        x_new = np.random.multivariate_normal(mean = f_mean,cov = var,size = point_num)
        for i in range(point_num):
            f_new[i] = getFunc(x,x_new[i],func)
            temp_mean+=f_new[i]
        f_new = sorted(f_new.items(),key=lambda item:item[1])[0:elite_num]
        for num in f_new:
            x_sample.append(x_new[num[0]])
        x_sample = np.array(x_sample)
        f_mean = np.sum(x_sample,axis = 0)/elite_num
        var = np.dot((x_sample-f_mean).T,(x_sample-f_mean))/elite_num
        value.append(temp_mean/point_num)
        error +=1
    return value

def SearchGradient(x,init,point_num,lr,norm,func,thresh):
    dim = init.shape[0]
    np.random.seed(1)
    f_mean = init
    var = np.eye(dim)
    value = [getFunc(x,init,func)]
    stop_var = 0 # stop the update of variance
    error = 1
    while (error<thresh):
        dir_mean=0
        dir_cov = 0
        f_value = []
        x_new = np.random.multivariate_normal(mean = f_mean,cov=var,size = point_num)
        for i in range(point_num):
            f_value.append(getFunc(x,x_new[i],func)) #TODO: other faster way
        value.append(sum(f_value)/point_num)
        f_value = np.array(f_value)
        
        delta = (x_new-f_mean).T
        cov_1 = np.linalg.inv(var)
        dir_mean = np.sum(np.dot(cov_1,delta*f_value.reshape((1,-1))),axis = 1)/point_num
        dir_mean = dir_mean.reshape(dim,)
        #print("dir_mean:",dir_mean)
        for i in range(point_num):
            dvalue = delta[:,i].reshape(-1,1)
            dir_cov+=((-0.5)*cov_1+0.5*np.linalg.multi_dot([cov_1,dvalue,dvalue.T,cov_1]))*f_value[i]
        dir_cov = dir_cov/point_num
        if norm:
            dir_mean = dir_mean/np.linalg.norm(dir_mean)
            dir_cov = dir_cov/np.linalg.norm(dir_cov)
        f_mean -=lr*dir_mean
        error+=1
        if stop_var:continue
        var_tm = var-lr*dir_cov
        eig = np.linalg.eigvals(var_tm)
        if np.all(eig>=0): var=var_tm
        else: stop_var=1
        #print(error,": ",var)
    return value




'''
    Minimax Algorithm
    Temporarily use 2048 Game Engine, Add some other engines later
'''
import copy
from game import Game
MAX_PLAYER, MIN_PLAYER, CHANCE_PLAYER = 0, 1, 2 
MiniMax, ExpectiMax = 0,1
class Node:
    def __init__(self,state,player_type):
        self.state = (copy.deepcopy(state[0]), state[1])
        self.children = []
        self.player_type = player_type
    def is_terminal(self):
        if len(self.children)==0:
            return True
        pass
''' supoort MiniMax, ExpectiMax'''
class Adversial:
    def __init__(self,root_state,player,moves,adtype,search_depth=3):
        self.root = Node(root_state,player)
        self.search_depth = search_depth
        self.moves = moves # the steps we could take
        self.simulator = Game(*root_state)
        self.adtype = adtype
    def build_tree(self,node = None, depth = 0):
        if node == None:
            node = self.root
        if depth == self.search_depth:
            return
        # find all the children availble
        if node.player_type == MAX_PLAYER or node.player_type == MIN_PLAYER:
            self.simulator.reset(*(node.state))
            for key in self.moves.keys():
                if (self.simulator.move(key)): # available next step
                    children_state = self.simulator.get_state()
                    if node.player_type == MAX_PLAYER and self.adtype==MiniMax:
                        children_node = Node(children_state,MIN_PLAYER)
                    elif node.player_type == MAX_PLAYER and self.adtype==ExpectiMax:
                        children_node = Node(children_state,CHANCE_PLAYER)
                    else:
                        children_node = Node(children_state,MAX_PLAYER)
                    node.children.append((key,children_node))
                self.simulator.undo()
        if node.player_type == CHANCE_PLAYER:
            #TODO update chance player rule
            self.simulator.reset(*(node.state)) #current state
            tm = copy.deepcopy(self.simulator.tile_matrix) # original matrix
            avai_tiles = self.simulator.get_open_tiles()
            for tile in avai_tiles:
                self.simulator.tile_matrix[tile[0]][tile[1]]=2 #TODO: update here
                children_state = self.simulator.get_state()
                children_node = Node(children_state,MAX_PLAYER)
                node.children.append((tile,children_node))
                self.simulator.tile_matrix = copy.deepcopy(tm)
            pass
        # build a tree for each child
        for child in node.children:
            self.build_tree(node = child[1], depth = depth+1)

    def getRoad(self,node = None):
        if node ==None:
            node = self.root
        if node.is_terminal():
            return None, node.state[1] # (path,node_value)
        elif node.player_type ==MAX_PLAYER:
            dir = -1
            best_score = -100
            for child in node.children:
                (cdir,cscore) = self.getRoad(child[1])
                if cscore>best_score:
                    best_score = cscore
                    dir = child[0]
            return dir,best_score
        elif node.player_type == MIN_PLAYER:
            dir = -1
            least_score = 1e8
            for child in node.children:
                (cdir,cscore) = self.getRoad(child[1])
                if cscore<least_score:
                    least_score = cscore
                    dir = child[0]
            return dir,least_score
        elif node.player_type == CHANCE_PLAYER:
            dir = None
            ex_best_score = 0
            num_children = len(node.children)  # number of all children
            for child in node.children:
                (cdir,cscore) = self.expectimax(child[1])
                ex_best_score+=cscore
            return dir, ex_best_score/num_children
    
    def getDecision(self):
        self.build_tree()
        direction,value = self.getRoad(self.root)
        return direction


import sympy as sy
import matplotlib.pyplot as plt
if __name__ =="__main__":
    symbol = np.array(sy.symbols('x1 x2'))
    function,diff1,diff2 = test_func(symbol,"sphere")
    x = np.array([2.0,2.0])
    plt.figure()
    #print(function,diff1)
    value = SearchGradient(symbol,x,10,0.01,0,function,100)
    #value = SG(symbol,100,10,x,0.01,function,0,0)
    ite = [i for i in range(len(value))]
    plt.plot(ite,value,'r--')
    plt.show()
    
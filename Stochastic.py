import numpy as np
from utils import getFunc,test_func
import random
import math
import sympy as sy
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify

''' ---------------------Stochastic Searching-------------------------'''
'''
Implement Simulated Annealing
'''
def SimulatedAnnealing(x,init,temp,func,method,thresh,is_visual = False):
    '''
    :param  x: symbolic x
            init: intial point
            temp: annealing temperature
            func: symbolic function
            method: annealing method (name, related_param) support Fast,Exponential,Log
            thresh: iteration time
            is_visual: whether visualize the process of converge---only used for testing!
    :return value: updated function value
    '''
    f_mean = init
    is_change = 1
    var = np.eye(init.shape[0])
    value = [getFunc(x,init,func)]
    error = 100
    cooling = temp
    k = 1 #current time
    if is_visual: #just for testing!
        random.seed(0)
        plt.figure()
        X1 = np.arange(-5,5,0.05)
        X2 = np.arange(-3,6,0.05)
        X1,X2 = np.meshgrid(X1,X2)
        Z = (1-X1)*(1-X1)+100*(X2-X1*X1)*(X2-X1*X1)
        contour1 = plt.contour(X1,X2,Z,10)
        plt.plot([1],[1],'o')
    while (k<thresh):
        x_new = np.random.multivariate_normal(mean = f_mean,cov = var)
        f_new = getFunc(x,x_new,func)
        error = f_new-value[-1]
        fold = f_mean
        # f(x_t)>f(x_t+1)
        if error>=0:
            if method[0] == "Fast":
                cooling = temp/k
            elif method[0] == "Exponential":
                if k == 1: cooling = temp
                elif is_change: cooling = method[1]*cooling
            elif method[0] == "Log":
                cooling = temp*math.log(2)/math.log(k+1)
            else:
                cooling = temp/k
            # decide whether choose point
            if random.uniform(0,1)<math.exp((-error)/cooling):
                f_mean = x_new #accept
                is_change = 1
            else: 
                is_change = 0
                continue #reject
        else:
            f_mean = x_new
            error = -error
            is_change = 1
        if is_visual:
            plt.plot([fold[0],f_mean[0]],[fold[1],f_mean[1]],'r--')
            if k == thresh-1:
                plt.plot([f_mean[0]],[f_mean[1]],'x')
            plt.pause(0.01)
        k+=1
        value.append(f_new)
    return value

'''
Implement Cross Entropy
'''
def CrossEntropy(x,init,point_num,rho,func,thresh,is_plot = False):
    '''
    :param  x: symbolic x
            init: intial point
            point_num: the number of sampling
            rho: elite point ratio
            func: symbolic function
            thresh: iteration time
    :return value: updated function value
    '''
    f_mean = init
    var = np.eye(init.shape[0])
    value = [getFunc(x,init,func)]
    error = 1
    elite_num = int(rho*point_num)
    if is_plot: #just for testing!
        random.seed(0)
        plt.figure()
        X1 = np.arange(-5,5,0.05)
        X2 = np.arange(-3,6,0.05)
        X1,X2 = np.meshgrid(X1,X2)
        Z = (1-X1)*(1-X1)+100*(X2-X1*X1)*(X2-X1*X1)
    while(error<thresh):
        f_new = {} # all the new sample values
        temp_mean = 0 # mean value of samples' function value
        x_sample = [] # x_values
        x_new = np.random.multivariate_normal(mean = f_mean,cov = var,size = point_num)
        if is_plot:
            plt.contour(X1,X2,Z,10)
            plt.plot([1],[1],'bo')
            plt.plot([2],[2],'yx')
            plt.plot(x_new[:,0],x_new[:,1],'g+',markersize = 2)
        #choose elite samples
        for i in range(point_num):
            f_new[i] = getFunc(x,x_new[i],func)
            temp_mean+=f_new[i]
        f_new = sorted(f_new.items(),key=lambda item:item[1])[0:elite_num]
        for num in f_new:
            x_sample.append(x_new[num[0]])
        x_sample = np.array(x_sample)
        if is_plot:
            plt.plot(x_sample[:,0],x_sample[:,1],'r+',markersize = 2)
            plt.pause(0.1)
            plt.cla()
        # update sampling function mean and covariance
        f_mean = np.sum(x_sample,axis = 0)/elite_num
        var = np.dot((x_sample-f_mean).T,(x_sample-f_mean))/elite_num
        value.append(temp_mean/point_num)
        error +=1
    return value

'''
Implement SearchGradient
'''
def SearchGradient(x,init,point_num,lr,func,thresh,norm=False,is_plot = False):
    '''
    :param  x: symbolic x
            init: intial point
            point_num: the number of sampling
            lr: learning rate
            func: symbolic function
            thresh: iteration time
            norm: whether normalization
    :return value: updated function value
    '''
    dim = init.shape[0]
    np.random.seed(2)
    f_mean = init
    var = np.eye(dim)
    value = [getFunc(x,init,func)]
    stop_var = 0 # stop the update of variance sign
    error = 1
    if is_plot: #just for testing!
        plt.figure()
        X1 = np.arange(-5,5,0.05)
        X2 = np.arange(-3,6,0.05)
        X1,X2 = np.meshgrid(X1,X2)
        Z = (1-X1)*(1-X1)+100*(X2-X1*X1)*(X2-X1*X1)
    while (error<thresh):
        dir_mean=0
        dir_cov = 0
        f_value = []
        #sampling
        x_new = np.random.multivariate_normal(mean = f_mean,cov=var,size = point_num)
        if is_plot:
            plt.contour(X1,X2,Z,10)
            plt.plot([1],[1],'bo')
            plt.plot([2],[2],'yx')
            plt.plot(x_new[:,0],x_new[:,1],'g+',markersize = 2)
        #get samples' value
        for i in range(point_num):
            f_value.append(getFunc(x,x_new[i],func))
        value.append(sum(f_value)/point_num)
        f_value = np.array(f_value)
        # update direction of Gaussian mean and covariance
        delta = (x_new-f_mean).T
        cov_1 = np.linalg.inv(var)
        dir_mean = np.sum(np.dot(cov_1,delta*f_value.reshape((1,-1))),axis = 1)/point_num
        dir_mean = dir_mean.reshape(dim,)
        # print("dir_mean:",dir_mean)
        for i in range(point_num):
            dvalue = delta[:,i].reshape(-1,1)
            dir_cov+=((-0.5)*cov_1+0.5*np.linalg.multi_dot([cov_1,dvalue,dvalue.T,cov_1]))*f_value[i]
        dir_cov = dir_cov/point_num
        # For normalization if necessary
        if norm:
            dir_mean = dir_mean/np.linalg.norm(dir_mean)
            dir_cov = dir_cov/np.linalg.norm(dir_cov)
        #update sampling function mean
        f_mean -=lr*dir_mean
        if is_plot:
            plt.plot([f_mean[0]],[f_mean[1]],'ro',markersize = 3)
            plt.pause(0.01)
            plt.cla()
        error+=1
        # judge whether update function covariance
        if stop_var:continue
        var_tm = var-lr*dir_cov
        eig = np.linalg.eigvals(var_tm)
        # if covariance is not positive-semi-definite, stop updating
        if np.all(eig>=0): var=var_tm
        else: stop_var=1
        
    return value

def plot():
    symbol = np.array(sy.symbols('x1 x2'))
    #Rosenbrock
    function = (1-symbol[0])**2+100*(symbol[1]-symbol[0]**2)**2
    # drop wave function
    # function2 = -(1+cos(12*(symbol[0]**2+symbol[1]**2)**0.5))/(0.5*(x1**2+x2**2)+2)
    # function,diff1,diff2 = test_func(symbol,"sphere")
    x = np.array([2.0,2.0])
    #print(function,diff1)
    value_sa = SimulatedAnnealing(symbol,x,10,function,("Fast"),100)
    value_ce = CrossEntropy(symbol,x,50,0.2,function,100)
    value_sg = SearchGradient(symbol,x,50,0.1,function,100,1)
    value = [value_sa,value_ce,value_sg]
    title = ["Simulated Annealing","Cross Entropy","Search Gradient"]
    plt.figure()
    plt.suptitle("Stochastic Search for Rosenbrock Function")
    for key in range(3):
        plt.subplot(1,3,key+1)
        ite = [i for i in range(len(value[key]))]
        plt.plot(ite,value[key],'r-')
        plt.title(title[key])
    '''if want to see dynamic plot:'''
    # value_sa = SimulatedAnnealing(symbol,x,10,function,("Fast"),100,True)
    # value_ce = CrossEntropy(symbol,x,50,0.2,function,30,True)
    # value_sg = SearchGradient(symbol,x,50,0.1,function,45,1,True)
    plt.show()


if __name__ =="__main__":
    
    plot()
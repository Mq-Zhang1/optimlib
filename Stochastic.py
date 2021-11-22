import numpy as np
from Gradient import getFunc,test_func
import random
import math
import sympy as sy
import matplotlib.pyplot as plt
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
    while (k<thresh):
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

def test():
    symbol = np.array(sy.symbols('x1 x2'))
    function,diff1,diff2 = test_func(symbol,"sphere")
    x = np.array([2.0,2.0])
    plt.figure()
    #print(function,diff1)
    value_sa = SimulatedAnnealing(symbol,x,10,function,"Fast",100)
    value_ce = CrossEntropy(symbol,x,50,0.2,function,100)
    value_sg = SearchGradient(symbol,x,10,0.01,0,function,100)
    value = [value_sa,value_ce,value_sg]
    #value = SG(symbol,100,10,x,0.01,function,0,0)
    for key in range(3):
        plt.subplot(1,3,key+1)
        ite = [i for i in range(len(value[key]))]
        plt.plot(ite,value[key],'r-')
    plt.show()

if __name__ =="__main__":
    test()
    # TODO add dynamic graphs
import numpy as np
'''Return the function value of a single point'''
def getFunc(x,point,func):
    '''
    :param x: symbolic x
    :param point: 1D array of point
    :param func: symbolic function
    :return function value of this point
    '''
    dim = point.size
    func_value = func
    for i in range(dim):
        func_value = func_value.subs(x[i],point[i])
    return float(func_value)

'''Return the first order gradient of single point'''
def getGrad(x,point,grad):
    '''
    :param x: symbolic x
    :param point: 1D array of point
    :param func: symbolic function of gradient
    :return gradient value of this point
    '''
    dim = point.size
    path = [0]*dim
    for i in range(dim):
        path[i] = getFunc(x,point,grad[i])
    return np.array(path)

'''Return Hessian Matrix of single point'''
def getHeissian(x,point,heissian):
    '''
    :param x: symbolic x
    :param point: 1D array of point
    :param func: symbolic function of heissian matrix
    :return heissian value of this point
    '''
    if heissian == []: return None
    dim = point.size
    path = np.ones((dim,dim))
    for i in range(dim):
        path[i] = getGrad(x,point,heissian[i])
    return path

# Establish a new function for test
'''Func: add function for testing, could yours here'''
def test_func(symbol,type):
    function = 0
    diff1 = 0
    diff2 = 0
    if type == "sphere":
        func = np.sum(symbol*symbol)
        diff1 = 2*symbol
        diff2 = np.eye(symbol.size)*2
    return func,diff1,diff2
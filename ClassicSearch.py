'''---------------------------Classic Search--------------------------'''

from matplotlib.colors import same_color
import numpy as np
import random
import math
import copy
from queue import Queue
import matplotlib.pyplot as plt
from geometry import *



'''
Implement BFS using queue
'''
def BFS(graph:Graph,start,end,is_plot = False):
    '''
    :param  graph: Graph object, abstract representation of our problem
            start: start point, size (n,)
            end: object point, size (n,)
            is_plot: used for testing, plz don't use
    :return path: the path from start to end
    '''
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    experienced = []
    # use queue structure to implement bfs
    frontier = Queue()
    frontier.put(startNode)
    while not frontier.empty():
        node = frontier.get()
        if is_plot:
            plt.plot([node.position[0]],[node.position[1]],'cx',markersize=5)
            plt.pause(0.01)
        experienced.append(node)
        if node == endNode:
            return findPath(experienced,startNode,endNode)
        index = graph.vertices.index(node)
        for child in graph.child[index]:
            if child[0] in experienced: continue # avoid circle
            child[0].parent = node
            frontier.put(child[0])
        
'''
Implement DFS using stack
'''
def DFS(graph:Graph,start,end,is_plot = False):
    '''
    :param  graph: Graph object, abstract representation of our problem
            start: start point, size (n,)
            end: object point, size (n,)
            is_plot: used for testing, plz don't use
    :return path: the path from start to end
    '''
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    # check whether start = end
    if startNode == endNode: print("No need to search!")
    experienced = []
    frontier = [startNode]
    # using stack structure to finish this problem
    while frontier:
        node = frontier.pop()
        if node in experienced: continue # avoid circle
        experienced.append(node)
        if node == endNode: return findPath(experienced,startNode,endNode)
        index = graph.vertices.index(node)
        for child in graph.child[index]:
            child[0].parent = node
            frontier.append(child[0])
            if is_plot:
                plt.plot([child[0].position[0]],[child[0].position[1]],'cx',markersize=5)
                plt.pause(0.01)

'''
Implement Dijkstra (almost the same as Astar)
'''
def Dijkstra(graph:Graph,start,end, is_plot = False):
    '''
    :param  graph: Graph object, abstract representation of our problem
            start: start point, size (n,)
            end: object point, size (n,)
            is_plot: used for testing, plz don't use
    :return path: the path from start to end
    '''
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    experienced = [] #experienced set
    frontier = [] # frontier set
    frontier.append(startNode)
    while len(frontier)>0:
        frontier = sorted(frontier,key = lambda t: t.G, reverse=True) #decreasing order, current closest point
        curNode = frontier.pop() #last one with lowest cost
        experienced.append(curNode)
        if curNode == endNode: # end of the search
            return findPath(experienced,startNode,endNode)
        index = graph.vertices.index(curNode)
        for child in graph.child[index]:
            if child[0] in experienced: continue #already in experienced set
            child[0].parent = curNode
            child[0].setG(curNode.G+child[1])
            overlap = [node for node in frontier if node == child[0]]
            if len(overlap) == 0 : 
                frontier.append(child[0]) # add new point
                if is_plot:
                    plt.plot([child[0].position[0]],[child[0].position[1]],'cx',markersize=5)
                    plt.pause(0.01)
            else:
                for node in overlap: # already in frontier, judge whether need to update
                    if node.G<= child[0].G: break
                    else: node = copy.deepcopy(child[0])

'''
    A* algorithm
'''
def Astar(graph:Graph,start,end,heuristic="Euclid",is_plot = False):
    '''
    :param  graph: Graph object, abstract representation of our problem
            start: start point, size (n,)
            end: object point, size (n,)
            heuristic: heuristic function, mostly used:"Euclid" and "Manhanttan"
            is_plot: used for testing, plz don't use
    :return path: the path from start to end
    '''
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    experienced = [] #experienced set
    frontier = [] # frontier set
    frontier.append(startNode)
    while len(frontier)>0:
        frontier = sorted(frontier,key = lambda t: t.F, reverse=True) #decreasing
        curNode = frontier.pop() #last one with lowest cost
        experienced.append(curNode)
        if curNode == endNode: # end of the search
            return findPath(experienced,startNode,endNode)
        index = graph.vertices.index(curNode)
        for child in graph.child[index]:
            if child[0] in experienced: continue #already in experienced set
            child[0].parent = curNode
            child[0].setG(curNode.G+child[1])
            if heuristic == "Euclid": # calculate heuristic value
                child[0].setH(np.linalg.norm(np.array(child[0].position)-np.array(end)))
            elif heuristic == "Manhattan":
                child[0].setH(np.linalg.norm(np.array(child[0].position)-np.array(end),1))
            else:
                raise ValueError("Don't support this heuristic function")
            child[0].calcF()
            overlap = [node for node in frontier if node == child[0]]
            if len(overlap) == 0 : #add new frontier point
                frontier.append(child[0])
                if is_plot:
                    plt.plot([child[0].position[0]],[child[0].position[1]],'cx',markersize=5)
                    plt.pause(0.01)
            else: # already in frontier, judge whether need to update
                for node in overlap:
                    if node.F<= child[0].F: break
                    else: node = copy.deepcopy(child[0])
''' Function to get path from startpoint to end point'''
def findPath(pathlist,startNode:ANode,endNode:ANode):
    path = []
    curNode = endNode
    for node in pathlist[::-1]:
        if node == curNode:
            path.append(node.position)
            curNode = node.parent # using parent attribute to get the path from back
    return path[::-1]
"----------------------RRT--------------------------"
'''
    Implement RRT Algorithm
'''

def RRT(space:Space,winsize,alpha,prob=0.2,maxiter=1000, is_plot = False):
    '''
    :param  space: the n-dimensional space where we wanna find the path
            winsize: windowsize for seaching, used we you wanna contract the region for searching
            alpha: the largest step size for each adding new poing
            prob: the prob to examine the endpoint
            maxiter: max time for iteration, avoid deadlock
            is_plot: used for testing, plz don't use this
    :return path: the path from start to end
    '''
    rrt_tree = Tree(space,winsize)
    iter = 1
    while True:
        while iter<maxiter: #for finding a point
            point = rrt_tree.generatePoint()
            #print("newpoint",point)
            iter+=1
            if not point.all(): continue
            if not rrt_tree.findNear(point,alpha,is_plot): continue
            break
        if random.uniform(0,1)<prob:
            if rrt_tree.findNear(space.end,alpha,is_plot): 
                path = findPath(rrt_tree.path,rrt_tree.start,rrt_tree.end)
                break
    return path

'''--------Below is the testing and plotting function----------'''
def rrt_test():
    start = [0.,0.]
    end = [5.,5.]
    obstacle = [Obstacle(2,[1.,1.],[2.,2.])]
    space= Space(2,[-1.,-1.],[6.,6.],start,end,obstacle)
    path = RRT(space,None,0.5)
    return path

def classic_test():
    height = 5
    width = 6
    obsX = [0,2,2,3,3,4]
    obsY = [1,1,3,1,4,4]
    map = Map(height,width,obsX,obsY)
    start = [0,0]
    end = [4,5]
    cost = 1
    graph = map.Map2Graph(start,end,None)
    path = DFS(graph,start,end)
    matrx = map.present(path)
    print(np.matrix(matrx))
from matplotlib.patches import Rectangle

def dijkstra_plot():
    height = 50
    width = 50
    start = [10,10]
    end = [40,40]
    obsx = (np.concatenate((np.array([15]*30),np.array([30]*30)),axis =None)).tolist()
    #print(obsx)
    obsy = (np.concatenate((np.array([i for i in range(30)]),np.array([i for i in range(20,50)])),axis = None)).tolist()
    #print(obsy)
    map = Map(height,width,obsx,obsy)
    graph = map.Map2Graph(start,end,None)
    plt.figure()
    ax = plt.subplot(111)
    ax.add_patch(Rectangle((0,0),height,width,fc = "none",ec = 'r',lw = 3))
    plt.plot(obsx,obsy,'go', markersize = 3)
    plt.plot([10],[10],'ro',markersize = 3)
    plt.plot([40],[40],'bo',markersize = 3)
    path = Dijkstra(graph,start,end,True)
    plt.show()

def astar_plot():
    height = 50
    width = 50
    start = [10,10]
    end = [40,40]
    obsx = (np.concatenate((np.array([15]*30),np.array([30]*30)),axis =None)).tolist()
    #print(obsx)
    obsy = (np.concatenate((np.array([i for i in range(30)]),np.array([i for i in range(20,50)])),axis = None)).tolist()
    #print(obsy)
    map = Map(height,width,obsx,obsy)
    graph = map.Map2Graph(start,end,None)
    plt.figure()
    ax = plt.subplot(111)
    ax.add_patch(Rectangle((0,0),height,width,fc = "none",ec = 'r',lw = 3))
    plt.plot(obsx,obsy,'go', markersize = 3)
    plt.plot([10],[10],'ro',markersize = 3)
    plt.plot([40],[40],'bo',markersize = 3)
    path = Astar(graph,start,end,"Euclid",True)
    plt.show()

def rrt_plot():
    plt.figure()
    ax = plt.subplot(111)
    # start from (-1,-1) to (11,11)
    # whole space
    ax.add_patch(Rectangle((-1,-1),12,12,fc = "none",ec = "b",lw = 3))
    start = [0.,0.]
    end = [8.,8.]
    plt.plot([0.],[0.],'go',markersize = 2)
    plt.plot([8.],[8.],'go',markersize = 2)
    # obstacle
    ax.add_patch(Rectangle((2,2),1,6,fc = "r",ec = "r",lw = 2))
    ax.add_patch(Rectangle((5,1),1,8,fc = "r",ec = "r",lw = 2))
    ax.add_patch(Rectangle((7,0),1,6,fc = "r",ec = "r",lw = 2))
    obstacle = [Obstacle(2,[2.,2.],[3.,8.]),Obstacle(2,[5.,1.],[6.,9.]),Obstacle(2,[7.,0.],[8.,6.])]
    space= Space(2,[-1.,-1.],[11.,11.],start,end,obstacle)
    path = RRT(space,None,0.5,is_plot = True)
    plt.show()

import argparse
parser = argparse.ArgumentParser(description='Descent')
parser.add_argument('--test', '-t', dest="test", type=int, default=0, \
    help='1: test Dijstra, \
          2: test A*, \
          3: test RRT,\
          '
)
args = parser.parse_args()
if __name__ =="__main__":
    if args.test == 1:
        dijkstra_plot()
    elif args.test == 2:
        astar_plot()
    elif args.test == 3:
        rrt_plot()
    else:
        classic_test()
    
'''---------------------------Classic Search--------------------------'''
'''
Define a class represent the map we want to apply grid search on
height: int, the height of the map
width: int, the width of the map
obstaclex: [x1,x2,x3...] x coordinates of obstacles
obstacley: [y1,y2,y3...] y coordinates of obstacles
'''
import numpy as np
import random
import math
import copy
from queue import Queue


class ANode:
    def __init__(self,parent,position):
        self.parent = parent
        self.position = position # relevant attri 
        self.H = 0
        self.G = 0
        self.F = 0
    def __eq__(self, other):
        return np.all(self.position == other.position)
    def setH(self,h):
        self.H = h
    def setG(self,g):
        self.G = g
    def calcF(self):
        self.F = self.H+self.G

'''
Implement Graph class
'''
class Graph: #using more general method to design the algorithm
    def __init__(self):
        self.vertices=[] #[]anode
        self.child = dict() #[](anode,cost)
        self.id = dict() # index--vertice
    def addVertice(self,Vertice):
        if Vertice not in self.vertices:
            index  = len(self.vertices)
            self.id[index] = Vertice
            self.vertices.append(Vertice)
            self.child[index] = [] #ANode is not hashable
    
    def addChild(self,parenVertice, childVertice, cost):
        index = self.vertices.index(parenVertice)
        self.child[index].append((childVertice,cost))

    
def findPath(pathlist,startNode:ANode,endNode:ANode):
    path = []
    curNode = endNode
    for node in pathlist[::-1]:
        if node == curNode:
            path.append(node.position)
            curNode = node.parent
    return path[::-1]

'''
Implement BFS using queue
'''
def BFS(graph:Graph,start,end):
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    experienced = []
    frontier = Queue()
    frontier.put(startNode)
    while not frontier.empty():
        node = frontier.get()
        experienced.append(node)
        if node == endNode:
            return findPath(experienced,startNode,endNode)
        index = graph.vertices.index(node)
        for child in graph.child[index]:
            if child[0] in experienced: continue
            child[0].parent = node
            frontier.put(child[0])
        
'''
Implement DFS using stack
'''
def DFS(graph:Graph,start,end):
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    if startNode == endNode: print("No need to search!")
    experienced = []
    frontier = [startNode]
    while frontier:
        node = frontier.pop()
        if node in experienced: continue
        experienced.append(node)
        if node == endNode: return findPath(experienced,startNode,endNode)
        index = graph.vertices.index(node)
        for child in graph.child[index]:
            child[0].parent = node
            frontier.append(child[0])

'''
Implement Dijkstra (almost the same as Astar)
'''
def Dijkstra(graph:Graph,start,end):
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    experienced = [] #experienced set
    frontier = [] # frontier set
    frontier.append(startNode)
    while len(frontier)>0:
        frontier = sorted(frontier,key = lambda t: t.G, reverse=True) #decreasing
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
            if len(overlap) == 0 : frontier.append(child[0])
            else:
                for node in overlap:
                    if node.G<= child[0].G: break
                    else: node = copy.deepcopy(child[0])

'''
    A* algorithm
'''
def Astar(graph:Graph,start,end,heuristic="Euclid"):
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
            if heuristic == "Euclid":
                child[0].setH(np.linalg.norm(np.array(child[0].position)-np.array(end)))
            elif heuristic == "Manhattan":
                child[0].setH(np.linalg.norm(np.array(child[0].position)-np.array(end),1))
            else:
                raise ValueError("Don't support this heuristic function")
            child[0].calcF()
            overlap = [node for node in frontier if node == child[0]]
            if len(overlap) == 0 : frontier.append(child[0])
            else:
                for node in overlap:
                    if node.F<= child[0].F: break
                    else: node = copy.deepcopy(child[0])

''' Implement class map for simple map game
height:int
width:int
obstaclex: list
obstacley: list
'''
class Map:
    def __init__(self,height,width,obstaclex,obstacley):
        '''
        width y
        -----------
        |          | height x
        |          |
        -----------
        '''
        self.height = height
        self.width = width
        if (len(obstaclex)!=len(obstacley)):
            raise ValueError("The dimenstion for X and Y doesn't match")
        self.obstaclex = obstaclex
        self.obstacley = obstacley
        self.instance = self.instance()
        self.direction = [[-1,0],[0,1],[1,0],[0,-1]]
    def instance(self):
        matrx = np.zeros((self.height,self.width))
        for n in range(len(self.obstaclex)):
            matrx[self.obstaclex[n]][self.obstacley[n]]=-1
        return matrx
    def getInstance(self):
        return self.instance
    def Map2Graph(self,start,end,windowsize):
        '''
        start: starting point
        end : ending point
        windowsize: n times height* n times width
        '''
        graph = Graph()
        # select the window
        width = abs(start[1]-end[1])
        height = abs(start[0]-end[0])
        winXmin,winXmax,winYmin,winYmax = 0,self.height-1,0,self.width-1
        if windowsize !=None:
            size = (windowsize-1)/2
            tmpXmin = min([start[0],end[0]])-int(height*size)-1
            tmpXmax = max([start[0],end[0]])+int(height*size)+1
            tmpYmin = min([start[1],end[1]])-int(width*size)-1
            tmpYmax = max([start[1],end[1]])+int(width*size)+1 #at least expand one grid
            if tmpXmin>winXmin: winXmin =  tmpXmin
            if tmpXmax<winXmax: winXmax = tmpXmax
            if tmpYmin>winYmin: winYmin = tmpYmin
            if tmpYmax<winYmax: winYmax = tmpYmax
        for h in range(winXmin,winXmax+1):
            for w in range(winYmin,winYmax+1):
                if self.instance[h][w] == -1: continue #obstacle
                curNode = ANode(None,[h,w])
                graph.addVertice(curNode)
                for dir in self.direction:
                    newX = h+dir[0]
                    newY = w+dir[1]
                    if newX<winXmin or newX>winXmax or newY<winYmin or newY>winYmax: continue
                    if self.instance[newX][newY] == -1: continue
                    childNode = ANode(None,[newX,newY])
                    graph.addChild(curNode,childNode,1)
                    graph.addVertice(childNode)
        return graph
    def present(self,path):
        curState = copy.deepcopy(self.getInstance())
        for point in path:
            curState[point[0]][point[1]]=1
        return curState
"----------------------RRT--------------------------"
class Obstacle:
    def __init__(self,dimension,lb_point,ru_point):
        self.dimension = dimension
        self.lb_point = lb_point
        self.ru_point = ru_point

    def checkInObs(self,point,thresh=0): #thresh: control not too close to obstacle can be used for close detection
        if np.all(np.array(point)>=np.array(self.lb_point)-thresh) and np.all(np.array(point)<=np.array(self.ru_point)+thresh):
            return True
        else:
            return False
    
    def checkInclude(self,space,other):
        # check whether this obstacle is included in other
        for _ in range(0.1*np.max(space.maxpoint-space.minpoint)*self.dimension):
            point = space.randomSample(other.lb_point,other.ru_point)
            if self.checkInObs(point): return True
        return False

    def checkLineThr(self,linepoint):
        for point in linepoint:
            if self.checkInObs(point): return True
        return False

class Space:
    def __init__(self,dimension,minpoint,maxpoint,start,end,obstacle=[]):
        '''
        param:
            minpoint:left-bottom of the space
            maxpoint: right-up of the space
            start: start point array
            end: end point array
            obstacle: all obstacle list [Obstacle1, Obstacle2]
        '''
        #TODO: add examine sanity program
        if dimension<2:
            raise Exception("Dimension cannot be less than 2 (must >=2)")
        self.dimension = dimension
        self.minpoint = np.array(minpoint)
        self.maxpoint = np.array(maxpoint)
        self.obstacle = obstacle
        self.start = np.array(start)
        self.end = np.array(end)
    
    def randomSample(self,_min,_max):
        '''
        min:array
        max:array
        '''
        point = []
        for d in range(self.dimension):
            point.append(random.uniform(max(self.minpoint[d],_min[d]),min(self.maxpoint[d],_max[d])))
        return np.array(point)
    
    def createRandomObs(self,n,maxlength):
        if maxlength == 0: maxlength = 0.1*(self.maxpoint-self.minpoint)
        obstacle = []
        for _ in range(n): #create n obs
            obs_lb = self.randomSample(self.minpoint,self.maxpoint) #array
            obs_ru = self.randomSample(obs_lb,obs_lb+maxlength)
            Obs = Obstacle(self.dimension,obs_lb,obs_ru)
            if Obs.checkInObs(self.start) or Obs.checkInObs(self.end): continue
            flag = 0
            for obs in obstacle:
                if Obs.checkInclude(obs): 
                    flag = 1
                    break
            if flag == 1: continue
            obstacle.append(Obs)
        return obstacle
    
    def randomLineSample(self,start,end,lr):
        '''
        start: array
        end:array
        lr = float
        '''
        point_list = []
        point = start
        dis = np.linalg.norm(start-end)
        delta = end-start
        point_num = int(dis/lr)-1
        for _ in range(point_num):
            increase = lr/dis*delta
            point = point+increase
            point_list.append(point)
        return point_list

class Tree:
    def __init__(self,space:Space,winsize=None):
        '''
        param:
            space: the space we wanna generate tree
            winsize: the genarate window
        '''
        self.space = space
        self.dimension = space.dimension
        self.start = ANode(None,space.start)
        self.end = ANode(None,space.end)
        self.path = [self.start]
        if winsize == None:
            self.window = (space.minpoint,space.maxpoint)
        else:
            diff = np.abs(space.start-space.end)
            size = (winsize-1)/2
            central = (space.start+space.end)/2
            self.window = (np.maximum(space.minpoint,central-size*diff),np.minimum(space.maxpoint,central+size*diff))

    def generatePoint(self):
        maxiter = 1000
        iter = 1
        while iter<maxiter:
            point = self.space.randomSample(*self.window)
            flag = 1
            for obs in self.space.obstacle:
                if obs.checkInObs(point):
                    flag = 0
                    break
            if flag ==1: return point
            iter+=1
        return None #no more point to sample or exceed sample interation limit

    def findNear(self, newpoint, alpha):
        '''
        param:
            newpoint: random new point array
            alpha: generate length
        '''
        #print("path,",self.path)
        success = False
        mindistance = float("inf")
        parent = None
        for node in self.path:
            valid = True
            old_pos = node.position
            line_sample = self.space.randomLineSample(old_pos,newpoint,0.01)
            for obs in self.space.obstacle:
                if obs.checkLineThr(line_sample): 
                    valid =False
                    break
            if valid == True:
                mindistance = np.linalg.norm(old_pos-newpoint)
                parent = (node,old_pos)
                success = True
        if success:
            if np.all(newpoint == self.space.end): alpha = mindistance
            cost = min(alpha,mindistance)
            true_pos = min(alpha,mindistance)/mindistance*(newpoint-parent[1])+parent[1]
            #print("nearest point",true_pos)
            newNode = ANode(parent[0],true_pos)
            newNode.setG(cost)
            self.path.append(newNode)
        return success

def RRT(space:Space,winsize,alpha,prob=0.2,maxiter=1000):
    '''
    space: the space where we wanna find find the path n-dimensional
    prob: the prob to examine the endpoint
    '''
    rrt_tree = Tree(space,winsize)
    iter = 1
    while True:
        while iter<maxiter: #for finding a point
            point = rrt_tree.generatePoint()
            #print("newpoint",point)
            iter+=1
            if not point.all(): continue
            if not rrt_tree.findNear(point,alpha): continue
            break
        if random.uniform(0,1)<prob:
            if rrt_tree.findNear(space.end,alpha): 
                path = findPath(rrt_tree.path,rrt_tree.start,rrt_tree.end)
                break
    return path

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
if __name__ =="__main__":
    # TODO visualization
    # path = rrt_test()
    # print(path)
    classic_test()
    
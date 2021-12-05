import numpy as np
import copy 
import matplotlib.pyplot as plt
import random

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

'''
Define a class represent the map we want to apply grid search on
height: int, the height of the map
width: int, the width of the map
obstaclex: [x1,x2,x3...] x coordinates of obstacles
obstacley: [y1,y2,y3...] y coordinates of obstacles
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

    def findNear(self, newpoint, alpha, is_plot):
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
                distance = np.linalg.norm(old_pos-newpoint)
                if distance<mindistance:
                    mindistance = distance
                    parent = (node,old_pos)
                success = True
        if success:
            if np.all(newpoint == self.space.end): alpha = mindistance
            cost = min(alpha,mindistance)
            true_pos = min(alpha,mindistance)/mindistance*(newpoint-parent[1])+parent[1]
            #print("nearest point",true_pos)
            if is_plot:
                plt.plot([parent[1][0],true_pos[0]],[parent[1][1],true_pos[1]],'b-')
                plt.pause(0.01)
            newNode = ANode(parent[0],true_pos)
            newNode.setG(cost)
            self.path.append(newNode)
        return success
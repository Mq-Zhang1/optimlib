'''---------------------------Grid Search--------------------------'''
'''
Define a class represent the map we want to apply grid search on
height: int, the height of the map
width: int, the width of the map
obstaclex: [x1,x2,x3...] x coordinates of obstacles
obstacley: [y1,y2,y3...] y coordinates of obstacles
'''
import numpy as np
import math
import copy
from queue import Queue
'''
height:int
width:int
obstaclex: list
obstacley: list
'''
class Map:
    def __init__(self,height,width,obstaclex,obstacley):
        self.height = height
        self.width = width
        if (len(obstaclex)!=len(obstacley)):
            raise ValueError("The dimenstion for X and Y doesn't match")
        self.obstaclex = obstaclex
        self.obstacley = obstacley
        self.instance = self.instance()
    def instance(self):
        matrx = np.zeros((self.height,self.width))
        for n in range(len(self.obstaclex)):
            matrx[self.obstaclex[n]][self.obstacley[n]]=-1
        return matrx
    def getInstance(self):
        return self.instance
'''
    A* algorithm
    Implement as map searching
    parent:Node
    position:[x,y]

'''
class ANode:
    def __init__(self,parent,position):
        self.parent = parent
        self.position = position # relevant attri
        self.H = 0
        self.G = 0
        self.F = 0
    def __eq__(self, other):
        return self.position == other.position
    def setH(self,h):
        self.H = h
    def setG(self,g):
        self.G = g
    def calcF(self):
        self.F = self.H+self.G
# start: the starting point [x,y]
# end: the target [x,y]
# f: cost function
# h: heuristic function
# cost: int 1
'''
Implement Graph class
'''
class Graph: #using more general method to design the algorithm
    def __init__(self):
        self.vertices=[] #[]anode
        self.child = dict() #[](anode,cost)

    def addVertice(self,Vertice):
        self.vertices.append(Vertice)
        self.child[Vertice]=[]
    
    def addChild(self,parenVertice, childVertice, cost):
        self.child[parenVertice] = (childVertice,cost)

    
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
def BFS(map:Map,start,end):
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    directions =[[-1,0],[0,1],[1,0],[0,-1]]
    experienced = []
    frontier = Queue()
    frontier.put(startNode)
    while not frontier.empty():
        node = frontier.get()
        experienced.append(node)
        if node == endNode:
            return findPath(experienced,startNode,endNode)
        for dir in directions:
            newx = dir[0]+node.position[0]
            newy = dir[1]+node.position[1]
            if newx<0 or newx>=map.height or newy<0 or newy>=map.width:
                continue
            if map.instance[newx][newy]<0: #obstacle
                continue
            curNode = ANode(node,[newx,newy])
            if curNode in experienced: continue
            frontier.put(curNode)
        
'''
Implement DFS using stack
'''
def DFS(map:Map,start,end):
    startNode = ANode(None,start)
    endNode = ANode(None,end)
    if startNode == endNode: print("No need to search!")
    direction = [[-1,0],[0,1],[1,0],[0,-1]]
    experienced = []
    frontier = [startNode]
    while frontier:
        node = frontier.pop()
        if node in experienced: continue
        experienced.append(node)
        if node == endNode: return findPath(experienced,startNode,endNode)
        for dir in direction:
            newx = dir[0]+node.position[0]
            newy = dir[1]+node.position[1]
            if newx<0 or newx>=map.height or newy<0 or newy>=map.width:
                continue
            if map.instance[newx][newy]<0: #obstacle
                continue
            curNode = ANode(node,[newx,newy])
            frontier.append(curNode)

'''
Implement Dijkstra (almost the same as Astar)
'''
def Dijkstra(map:Map,start,end,cost):
    directions =[[-1,0],[0,1],[1,0],[0,-1]] #up, right, down, left
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
        for dir in directions:
            newX = dir[0]+curNode.position[0]
            newY = dir[1]+curNode.position[1]
            if newX<0 or newX>=map.height or newY<0 or newY>=map.width:
                continue # out of range
            if map.instance[newX][newY] <0:
                continue #obstacle
            newNode = ANode(curNode,[newX,newY])
            if newNode in experienced: continue #already in experienced set
            newNode.setG(curNode.G+cost)
            overlap = [node for node in frontier if node == newNode]
            if len(overlap) == 0 : frontier.append(newNode)
            else:
                for node in overlap:
                    if node.G<= newNode.G: break
                    else: node = copy.deepcopy(newNode)

def Astar(map:Map,start,end,cost,heuristic="Euclid"):
    directions =[[-1,0],[0,1],[1,0],[0,-1]] #up, right, down, left
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
        for dir in directions:
            newX = dir[0]+curNode.position[0]
            newY = dir[1]+curNode.position[1]
            if newX<0 or newX>=map.height or newY<0 or newY>=map.width:
                continue # out of range
            if map.instance[newX][newY] <0:
                continue #obstacle
            newNode = ANode(curNode,[newX,newY])
            if newNode in experienced: continue #already in experienced set
            newNode.setG(curNode.G+cost)
            if heuristic == "Euclid":
                newNode.setH(math.sqrt((newX-end[0])**2+(newY-end[1])**2))
            elif heuristic == "Manhattan":
                newNode.setH(abs(newX-end[0])+abs(newY-end[1]))
            else:
                raise ValueError("Don't support this heuristic function")
            newNode.calcF()
            overlap = [node for node in frontier if node == newNode]
            if len(overlap) == 0 : frontier.append(newNode)
            else:
                for node in overlap:
                    if node.F<= newNode.F: break
                    else: node = copy.deepcopy(newNode)
def present(path,map:Map):
    curState = map.getInstance()
    for point in path:
        curState[point[0]][point[1]]=1
    return curState

if __name__ =="__main__":
    # TODO visualization
    # TODO generalization
    height = 5
    width = 6
    obsX = [0,2,2,3,3,4]
    obsY = [1,1,3,1,4,4]
    map = Map(height,width,obsX,obsY)
    start = [0,0]
    end = [4,5]
    cost = 1
    path = Dijkstra(map,start,end,cost)
    matrx = present(path,map)
    print(np.matrix(matrx))

    
'''
    Minimax Algorithm
    Use 2048 Game Engine for testing.
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
                (cdir,cscore) = self.getRoad(child[1])
                ex_best_score+=cscore
            return dir, ex_best_score/num_children
    
    def getDecision(self):
        self.build_tree()
        direction,value = self.getRoad(self.root)
        return direction

import numpy as np
import random
import copy

''' Action class'''
class Action:
    def __init__(self,name,outcome=None,prob=None):
        self.name = name
        if outcome == None and prob == None:
            self.outcome = []
            self.prob = []
        else:
            self.outcome = outcome
            self.prob = prob
    def addOut(self,nextState,prob):
        self.outcome.append(nextState)
        self.prob.append(prob)
    def getQvalue(self): #get qvalue of this state-action pair
        value = 0
        for i in range(len(self.outcome)):
            value+=self.prob[i]*self.outcome[i].value
        return value

''' State class'''
class State:
    def __init__(self,name,reward):
        self.name = name
        self.reward = reward
        self.policy = None
        self.action = []
        self.value = random.random()
    def addAction(self,act:Action):
        self.action.append(act)
    def addAction(self,act:list):
        self.action = act
    def changeValue(self,num):
        self.value = num

''' Function to evaluate the performance of current policy'''
def evaPolicy(states,gamma,thresh,type):
    length = len(states)
    flag = [1]*length
    #error = [10]*len(states)
    count = [0]*length
    while sum(flag) !=0:
        for i in range(length):
            if flag[i] == 0: continue
            #value = state.value
            if type == "policy":
                if states[i].policy == None: Qvalue=0
                else: 
                    Qvalue = states[i].policy.getQvalue()
            elif type == "value":
                qvalue = [action.getQvalue() for action in states[i].action]
                if len(qvalue) == 0:
                    Qvalue = 0
                else: Qvalue = max(qvalue)
            newValue = states[i].reward+gamma*Qvalue
            dvalue = newValue-states[i].value
            states[i].value = newValue
            if abs(dvalue) < thresh: count[i]+=1
            if count[i]>=4: flag[i] = 0
            #error[index] = abs(dvalue)
    return states

'''
Implement Value Iteration
states: list[State]
gamma: float, discount factor
thresh: float, threshold for convergence
'''
def ValueIter(state,gamma,thresh=1e-4):
    return evaPolicy(state,gamma,thresh,type = "value")

'''
Implement Policy Iteration
'''   
def PolicyIter(states,gamma,thresh):
    #initialize policy
    for state in states:
        if len(state.action) != 0:
            state.policy = state.action[0]
    length = len(states)
    flag = [0]*length
    while sum(flag) < length:
        #evaluation
        states = evaPolicy(states,gamma,thresh,"policy")
        #update
        for i in range(length):
            allQvalue = [action.getQvalue() for action in states[i].action]
            if len(allQvalue) == 0:
                maxValue = 0 #no action
                action = None
                Qvalue = 0
            else: 
                #print(allQvalue)
                maxValue = max(allQvalue)
                action = states[i].action[allQvalue.index(maxValue)]
                Qvalue = states[i].policy.getQvalue()
            if maxValue > Qvalue:
                states[i].policy = action
                flag[i]=0
            else: flag[i] = 1
    return states

if __name__ =="__main__":
    States=[]
    s1 = State("s1",-10)
    s1.value = 100
    
    s2 = State("s2",10)
    s2.value = -100
    a11 = Action("a11",[s1,s2],[0.8,0.2])
    a12 = Action("a12",[s1,s2],[0.5,0.5])
    s1.addAction([a11,a12])
    States=[s1,s2]
    
    # policy iteration test
    print("-----Policy Iteration-------")
    print("Best policy")
    States=PolicyIter(States,0.9,1e-4)
    for state in States:
        if state.policy != None:
            print(state.name,state.policy.name)
        else: 
            print(state.name,state.policy)
    
    # value iteration test
    print("-----Value Iteration-------")
    print("Optimal value")
    States = ValueIter(States,0.9,1e-4)
    for state in States:
        print(state.name,state.value)
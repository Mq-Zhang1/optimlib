import random 
from MDP import Action, State

'''
Function for generate sequence for monte carlo evaluation
states: [State] all the possible states in this path
'''
def generate_Seq(states,gamma):
    initState = random.choice(states)
    episode = [initState]
    num = random.randint(5,11)
    Gvalue = dict()
    count = dict()
    for i in range(num):
        if initState.policy == None or len(initState.policy.outcome)==0: break # no next step
        nextState = random.choice(initState.policy.outcome)
        episode.append(nextState)
        initState = nextState
    length = len(episode)
    for i in range(length): #TODO: not to the last one, bcs it's not good enough
        value = 0
        factor = 1
        for j in range(i,length):
            value+=factor*episode[j].reward
            factor*=gamma
        if episode[i] in Gvalue.keys(): 
            Gvalue[episode[i]]+=value
            count[episode[i]]+=1
        else: 
            Gvalue[episode[i]]=value
            count[episode[i]]=1
    return episode,Gvalue,count

def Monte_Carlo(states,gamma,time):
    length = len(states)
    value_dic = dict()
    counter = dict()
    for state in states:
        value_dic[state] = 0
        counter[state]=0
    k = 0
    while k < time:
        _,gvalue,count = generate_Seq(states,gamma)
        for key in gvalue.keys():
            value_dic[key]+=gvalue[key]
            counter[key]+=count[key]
        k+=1
    for state in states:
        state.changeValue(value_dic[state]/counter[state])
    return states

def TD_policy(states,gamma,time):
    counter = dict()
    gValue = dict()
    for state in states:
        counter[state] = 0
        gValue[state] = 0
    gValue[None] = 0
    k = 0
    while k<time:
        initState = random.choice(states) # randomly choose one start point
        while initState != None:
            counter[initState] +=1
            if initState.policy==None:
                nextState = None
            else:nextState = random.choice(initState.policy.outcome)
            gValue[initState] = gValue[initState]+(10/(9+counter[initState]))*(initState.reward+gamma*gValue[nextState]-gValue[initState])
            initState = nextState
        k+=1
    for state in states:
        state.changeValue(gValue[state])
    return states

def pick_action(state,qvalue,epsilon):
    max_value = -100000
    best_action = None
    for key in qvalue[state].keys():
        if qvalue[state][key]>max_value: 
            max_value = qvalue[state][key]
            best_action = key
    if random.random()<epsilon:
        return random.choice(state.action)
    else:
        return best_action
        
def Qlearning(states,gamma,time,eps = 0.4):
    counter = dict()
    qValue = dict()
    for state in states:
        counter[state] = 0
        qValue[state] = dict()
        if len(state.action) == 0:
            qValue[state][None] = 0
        for act in state.action:
            qValue[state][act] = 0
    k = 0
    while k<time:
        initState = random.choice(states) # randomly choose one start point
        while initState != None:
            counter[initState] +=1
            if len(initState.action)==0:
                act = None
                nextState = None
                qValue[initState][None] = qValue[initState][None]+(10/(9+counter[initState]))*(initState.reward-qValue[initState][None])
            else:
                act = pick_action(initState,qValue,eps)
                nextState = random.choice(act.outcome)
                qValue[initState][act] = qValue[initState][act]+(10/(9+counter[initState]))*(initState.reward+gamma*max(qValue[nextState].values())-qValue[initState][act])
            initState = nextState
        k+=1
    for state in states:
        state.changeValue(max(qValue[state].values()))
    return states

def test_mc(states,gamma,time):
        States = Monte_Carlo(states,gamma,time)
        for k in States:
            print(k.name,k.value)

def test_td(states,gamma,time):
        States = TD_policy(states,gamma,time)
        for k in States:
            print(k.name,k.value)

def test_q(states,gamma,time):
        States = Qlearning(states,gamma,time)
        for k in States:
            print(k.name,k.value)

import argparse
parser = argparse.ArgumentParser(description='Descent')
parser.add_argument('--test', '-t', dest="test", type=int, default=0, \
    help='1: test monte carlo, \
          2: test temporal difference, \
          3: test q learning,\
            '
)
args = parser.parse_args()

if __name__ =="__main__":
    States=[]
    s1 = State("s1",-10)
    s1.value = 100
    s2 = State("s2",10)
    s2.value = -100
    a11 = Action("a11",[s1,s1,s1,s1,s2])
    a12 = Action("a12",[s1,s2])
    s1.addAction([a11,a12])
    s1.policy = a11
    States=[s1,s2]
    if args.test == 1:
        test_mc(States,0.9,5000) 
    elif args.test == 2:
        test_td(States,0.9,10000)
    elif args.test == 3:
        test_q(States,0.9,10000)
          


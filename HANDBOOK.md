**This is a usage handbook for all optimization methods**
# Descent Algorithm

### <span style="color:darkred">CLASS Gradient(\*args,**kargs)</span>

> Parameters:
>> * lr: learning rate, default value 0.001

> Functions:
>> * setlr (lr): changing learning rate of gradient descent
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Newton(\*args,**kargs)</span>

> Parameters:
>> * None

> Functions:
>> * step (gradient,heissian): obtain update direction 
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* param heissian: array of shape`(n,n)`, heissian matrix of current point
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Momentum(\*args,**kargs)</span>
$$
v_t=\gamma v_{t-1} + \eta \nabla f(x)\\
x = x-v_t
$$
> Parameters:
>> * velocity: $\beta$ hyperparamter, reflects the degree the new direction rely on the previous one, $\in (0,1)$
>> * lr: learning rate, default value 0.001

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Nesterov(\*args,**kargs)</span>
Different from Momentum, Nesterov looks ahead to where we are going to be and evaluate the gradient there.
$$
v_t = \gamma v_{t-1} +\eta \nabla f(x-\gamma v_{t-1})\\
x = x-v_t
$$
> Parameters:
>> * velocity: $\beta$ hyperparamter, reflects the degree the new direction rely on the previous one, $\in (0,1)$
>> * lr: learning rate, default value 0.001

> Functions:
>> * step (x, point, gradFunc): obtain update direction once
>>	* param x: array of shape`(n,)` storing all the symbolic x variable, for example `[x1 x2]`, where x1, x2 aobtained by `sympy.symbols("x1 x2")`
>>	* param point: array of shape`(n,)` initial point
>>	* param gradFunc: symbolic function formula represented by `x`, for example: `x1*x1+x2*x2`
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Adagrad(\*args,**kargs)</span>
Adagrad automatically tune the learning rate for each dimension of the varaibles

$$
x_{t+1} = x_t-\frac{\eta}{\sqrt{G_t+\epsilon I}}\nabla f(x_t)
$$

> Parameters:
>> * epsilon: $\epsilon$ hyperparamter, used for avoid dividing 0, default $1e^{-6}$
>> * lr: nominator in $\frac{\eta}{\sqrt{G_t+\epsilon I}}$, default 0.01

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Rmsprop(\*args,**kargs)</span>
Compared with Adagrad, this algorithm tune the learning rate with normalization
$$
G_t = \beta G_{t-1}+(1-\beta)(\nabla f(x_t))^2\\
x_{t+1} = x_t-\frac{\eta}{\sqrt{G_t+\epsilon I}}\nabla f(x_t)
$$
> Parameters:
>> * beta: $\beta$ parameter in above function, default 0.9
>> * epsilon: $\epsilon$ hyperparamter, used for avoid dividing 0, default $1e^{-7}$
>> * lr: nominator in $\frac{\eta}{\sqrt{G_t+\epsilon I}}$

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS AdaDelta(\*args,**kargs)</span>
AdaDelta can be seen as a variation from RMSprop, this method doesn't require set learning rate.
$$
G_t = \beta G_{t-1}+(1-\beta)\nabla f(x_t)^2\\
\Delta x = -\frac{\sqrt{\epsilon+v_{t-1}}}{\sqrt{\epsilon+G_i}} \nabla f(x_t)\\
v_i = \beta v_{i-1}+(1-\beta) \Delta w^2\\
x_t = x_{t+1}+\Delta x
$$
> Parameters:
>> * beta: $\beta$ parameter in above function, default 0.9
>> * epsilon: $\epsilon$ hyperparamter, used for avoid dividing 0, default $1e^{-7}$

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Adam(\*args,**kargs)</span>
Combines Momentum with RMSprop
$$
v_t = \gamma v_{t-1}+(1-\gamma)\nabla f(x_t)\\
G_t = \beta G_{t-1}+(1-\beta)(\nabla f(x_t))^2\\
x_{t+1} = x_t-\frac{\eta}{\sqrt{\hat{G_t}+\epsilon I}}\hat{v_t}\\
where\,\,\,\hat{v_t} = \frac{v_t}{1-\gamma^t}\,\,and\,\,\hat{G_t} = \frac{G_t}{1-\beta^t}\
$$
> Parameters:
>> * velocity: $\gamma$ parameter in above function, default 0.9
>> * beta: $\beta$ parameter in above function, default 0.99
>> * epsilon: $\epsilon$ hyperparamter, used for avoid dividing 0, default $1e^{-7}$
>> * lr: nominator in $\frac{\eta}{\sqrt{G_t+\epsilon I}}$

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS AdaMax(\*args,**kargs)</span>
A variation from Adam
$$
v_t = \gamma v_{t-1}+(1-\gamma)\nabla f(x_t)\\
G_t = max\,(\beta G_{t-1},|\nabla f(x_t)|)\\
x_{t+1} = x_t-\frac{\eta}{G_t}\hat{v_t}\\
where\,\,\,\hat{v_t} = \frac{v_t}{1-\gamma^t}
$$
> Parameters:
>> * alpha: $\gamma$ parameter in above function, default 0.9
>> * beta: $\beta$ parameter in above function, default 0.99
>> * lr: nominator in $\frac{\eta}{G_t}$, default 0.01

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Nadam(\*args,**kargs)</span>
A combination of Nesterov Momentum and RMSprop
$$
v_t = \gamma v_{t-1}+(1-\gamma)\nabla f(x_t)\\
G_t = \beta G_{t-1}+(1-\beta)(\nabla f(x_t))^2\\
x_{t+1} = x_t-\frac{\eta}{\sqrt{\hat{G_t}+\epsilon I}}(\gamma \hat{v_t}+\frac{(1-\gamma)\nabla f(x_t)}{1-\gamma^t})\\
where\,\,\,\hat{v_t} = \frac{v_t}{1-\gamma^t}\,\,and\,\,\hat{G_t} = \frac{G_t}{1-\beta^t}
$$
> Parameters:
>> * alpha: $\gamma$ parameter in above function, default 0.9
>> * beta: $\beta$ parameter in above function, default 0.99
>> * lr: nominator in $\frac{\eta}{\sqrt{\hat{G_t}+\epsilon I}}$, default 0.01
>> * epsilon: $\epsilon$ hyperparamter, used for avoid dividing 0, default $1e^{-7}$

> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

### <span style="color:darkred">CLASS Nadamax(\*args,**kargs)</span>
A combination of Nesterov and AdaMax
$$
v_t = \gamma v_{t-1}+(1-\gamma)\nabla f(x_t)\\
G_t = max\,(\beta G_{t-1},|\nabla f(x_t)|)\\
\hat{v_t} = \frac{v_t}{1-\gamma^t}\\
x_{t+1} = x_t-\frac{\eta}{G_t(1-\gamma^t)}(\gamma v_t+(1-\gamma)\nabla f(x_t))\\
$$
> Parameters:
>> * alpha: $\gamma$ parameter in above function, default 0.9
>> * beta: $\beta$ parameter in above function, default 0.99
>> * lr: nominator in $\frac{\eta}{G_t(1-\gamma^t)}$, default 0.01


> Functions:
>> * step (gradient): obtain update direction once
>>	* param gradient: array of shape`(n,)` where n is the dimension of function 
>>	* return: direction `(n,)`

**NOTE:** To use above optimization class, we first need to create an object and feed it into update function

### <span style="color:darkred">FUNCTION update(\*args,**kargs)</span>
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * thresh: largest error threshold could stand
>> * method: optimization object we wanna create, such as `Adam`,`AdaMax`
>> * param: tuple containing all the required parameters like `lr`, `alpha`
>> * grad_func: symbolic gradient function
>> * heissian_func: symbolic gradient matrix, default `[]`

> Returns:
>> * point_list: list of all updated point size `(L,n)`
>> * value: list of all function value along the path size `(L)`
>> where L is the iteration time until converge, n is function dimension

#### EXAMPLE
~~~python
import sympy as sy
import numpy as np
symbol = np.array(sy.symbols('x1 x2'))
init = np.array([2,2])
thresh = 0.01
function = symbol[0]**2-symbol[0]*symbol[1]+3*symbol[1]**2+5
diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
diff2 = np.array([[sy.diff(diff1[0],symbol[0]),sy.diff(diff1[0],symbol[1])],[sy.diff(diff1[1],symbol[0]),sy.diff(diff1[1],symbol[1])]])
value = update(symbol,init,function,thresh,Gradient,(0.01,),diff1)
~~~

## OTHER FUNCTIONS
### <span style="color:darkred">FUNCTION linear_search(\*args,**kargs)</span>
Linear search algorithm with backtracking (Armijo condition)
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * thresh: largest error threshold could stand
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * grad_func: symbolic gradient function

> Returns:
>> * point_list: list of all updated point size `(L,n)`
>> * value: list of all function value along the path size `(L)`

#### Example:
~~~python
symbol = np.array(sy.symbols('x1 x2'))
init = np.array([1,0])
thresh = 0.01
function = (symbol[0])**2+(symbol[1])**2
diff1 = np.array([sy.diff(function,symbol[0]),sy.diff(function,symbol[1])])
_,value = linear_search(symbol,init,thresh,function,diff1)
~~~
### <span style="color:darkred">FUNCTION TrustRegion(\*args,**kargs)</span>
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * grad_func: symbolic gradient function `[x1 x2...]`size`(n,)`
>> * heissian_func: symbolic gradient matrix size`(n,n)`
>> * methods: support `"trust-exact"`,`"dogleg"`

> Returns:
>> * res: minimum value of function

#### Example:
~~~python
TrustRegion(symbol,init,function,diff1,diff2,"dogleg")
~~~

### <span style="color:darkred">FUNCTION CG(\*args,**kargs)</span>
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * grad: symbolic gradient function `[x1 x2...]`size`(n,)`
>> * heissian: symbolic gradient matrix size`(n,n)`
>> * methods: support `"Newton-CG"`,`"trust-ncg"` `"trust-krylov"`, default `"Newton-CG"`

> Returns:
>> * res: minimum value of function

#### Example:
~~~python
CG(symbol,init,function,diff1,diff2,"Newton-CG")
~~~

# Stochastic Search
**All use Gaussian Distribution to sample**
### <span style="color:darkred">FUNCTION SimulatedAnnealing(\*args,**kargs)</span>
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * temp: annealing temperature
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * method: annealing schedule, `tuple` includes two parts `("method_name", related_param)`, supports three methods `Fast`,`Exonential`,`Log`
>> * thresh: iteration times
>> * is_visual: whether visualize the process, now only support for testing

> Return:
>> * value: updated function value. type `list`, size `(n)` where `n` is the iteration times

> Annealing Schedule Intro:
>> * Fast annealing: $T_k = T/k$
>> * Exponential annealing: $T_{k+1} = \gamma T_k$
>> * Log Annealing: $T_k = \frac{Tlog(2)}{log(k+1)}$

>> where `T` is the annealing temperature corresponding to `temp` in parameters

#### Example:
~~~python
symbol = np.array(sy.symbols('x1 x2'))
function = (1-symbol[0])**2+100*(symbol[1]-symbol[0]**2)**2
x = np.array([2.0,2.0])
value_sa = SimulatedAnnealing(symbol,x,10,function,("Fast"),100)
~~~

### <span style="color:darkred">FUNCTION CrossEntropy(\*args,**kargs)</span>
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * point_num: the number of sampling points
>> * rho: elite point ratio
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * thresh: iteration times
>> * is_visual: whether visualize the process, now only support for testing, default false

> Return:
>> * value: updated function value. type `list`, size `(n)` where `n` is the iteration times

#### Example:
~~~python
value_ce = CrossEntropy(symbol,x,50,0.2,function,100)
~~~

### <span style="color:darkred">FUNCTION SearchGradient(\*args,**kargs)</span>
> Parameters:
>> * x: symbolic x array of size `(n,)`
>> * init: initial point array of size`(n,)`
>> * point_num: the number of sampling points
>> * lr: gradient learning rate
>> * func: symbolic function such as `x1*x1+x2*x2`
>> * thresh: iteration times
>> * norm: whether normalize the gradient direction of gaussian parameters $\Sigma$ and $\mu$
>> * is_visual: whether visualize the process, now only support for testing, default false

> Return:
>> * value: updated function value. type `list`, size `(n)` where `n` is the iteration times

#### Example:
~~~python
value_sg = SearchGradient(symbol,x,50,0.1,function,100,True)
~~~
# Classic Search

### <span style="color:darkred">FUNCTION BFS(\*args,**kargs)</span>
> Parameters:
>> * graph:  Graph object, abstract representation of our problem. Could convert any search problem into graph structure using `Class Graph` in geometry.py
>> * start: start point, size(n,)
>> * end: terminal point, size(n,)
>> * is_plot: used for testing, default False

> Return:
>> * path: the path from start to end

### <span style="color:darkred">FUNCTION DFS(\*args,**kargs)</span>
> Parameters:
>> * graph:  Graph object, abstract representation of our problem. Could convert any search problem into graph structure using `Class Graph` in geometry.py
>> * start: start point, size(n,)
>> * end: terminal point, size(n,)
>> * is_plot: used for testing, default False

> Return:
>> * path: the path from start to end

### <span style="color:darkred">FUNCTION Dijkstra(\*args,**kargs)</span>
> Parameters:
>> * graph:  Graph object, abstract representation of our problem. Could convert any search problem into graph structure using `Class Graph` in geometry.py
>> * start: start point, size(n,)
>> * end: terminal point, size(n,)
>> * is_plot: used for testing, default False

> Return:
>> * path: the path from start to end

### <span style="color:darkred">FUNCTION Astar(\*args,**kargs)</span>
> Parameters:
>> * graph:  Graph object, abstract representation of our problem. Could convert any search problem into graph structure using `Class Graph` in geometry.py
>> * start: start point, size(n,)
>> * end: terminal point, size(n,)
>> * heuristic: heuristic function, mostly used:`"Euclid"` and `"Manhanttan"`
>> * is_plot: used for testing, default False

> Return:
>> * path: the path from start to end

#### Example:
~~~python
from geometry import Map
import numpy as np
height,width,cost = 5,6,1
obsX = [0,2,2,3,3,4]
obsY = [1,1,3,1,4,4]
map = Map(height,width,obsX,obsY)
start,end = [0,0],[4,5]
graph = map.Map2Graph(start,end,None)
path = DFS(graph,start,end)
matrx = map.present(path)
print(np.matrix(matrx))
~~~

### <span style="color:darkred">FUNCTION RRT(\*args,**kargs)</span>
> Parameters:
>> * space: the n-dimensional space where we wanna find the path, you could define your own space using `Class Space` in geometry.py
>> * winsize: windowsize for seaching, used we you wanna contract the region for searching
>> * alpha: the largest step size for each adding new poing, like learning rate
>> * prob: the prob to examine the endpoint, default `0.2`
>> * maxiter: max time for iteration, avoid deadlock, default `1000`
>> * is_plot: used for testing, default False

> Return:
>> * path: the path from start to end

#### Example:
~~~python
from geometry import Space
import numpy as np
start,end = [0.,0.],[5.,5.]
obstacle = [Obstacle(2,[1.,1.],[2.,2.])]
space= Space(2,[-1.,-1.],[6.,6.],start,end,obstacle)
path = RRT(space,None,0.5)
~~~

### <span style="color:darkred">CLASS Adversial(\*args,**kargs)</span>

Need to use this adversial search under a certain game scenario, at least know:

1. How to define state of the game
2. The root state of adversial tree
3. Tree type, expectimax or minimax tree
4. What's the next move of certain game states

> Parameters:
>> * root_state: The game state of the root node, this parameter is defined by game type
>> * player: Player type of root node, support `MAX_PLAYER, MIN_PLAYER, CHANCE_PLAYER = 0, 1, 2` 
>> * moves: Move types of every state, for instance, in 2048 game, `MOVES = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}`
>> * adtype: Type of the adversial tree, support `MiniMax, ExpectiMax = 0,1`
>> * search_depth: The depth of the search tree, default 3

> Functions:
>> * build_tree: Build the adversial tree
>> * getRoad: Obtain decision path according to the adversial tree
>> * getDecision: Get next step decision of current state

#### Usage:
~~~python
MOVES = {0: 'up', 1: 'left', 2: 'down', 3: 'right'}
ai = Adversial(game.get_state(),0,MOVES,0,search_depth=3) 
# game.get_state() is self-defined game-state function
direction = ai.getDecision()
~~~

### <span style="color:darkred">FUNCTION ValueIter(\*args,**kargs)</span>
> Parameters:
>> * state: State of the decision process. You can use Class `Action` and `State` in `MDP.py` to convert and markov process into state-action structure.
>> * gamma: discount factor
>> * thresh: iteration threshold, defualt $1e^{-4}$

> Return:
>> * updated states list `[State]`

### <span style="color:darkred">FUNCTION PolicyIter(\*args,**kargs)</span>
> Parameters:
>> * state: State of the decision process. You can use Class `Action` and `State` in `MDP.py` to convert and markov process into state-action structure.
>> * gamma: discount factor
>> * thresh: iteration threshold, defualt $1e^{-4}$

> Return:
>> * updated states list `[State]`

### <span style="color:darkred">FUNCTION Monte_Carlo(\*args,**kargs)</span>
> Parameters:
>> * states: State of the decision process. You can use Class `Action` and `State` in `MDP.py` to convert and markov process into state-action structure.
>> * gamma: discount factor
>> * time: iteration time

> Return:
>> * updated states list `[State]`

### <span style="color:darkred">FUNCTION TD_policy(\*args,**kargs)</span>
> Parameters:
>> * states: State of the decision process. You can use Class `Action` and `State` in `MDP.py` to convert and markov process into state-action structure.
>> * gamma: discount factor
>> * time: iteration time

> Return:
>> * updated states list `[State]`

### <span style="color:darkred">FUNCTION Qlearning(\*args,**kargs)</span>
> Parameters:
>> * states: State of the decision process. You can use Class `Action` and `State` in `MDP.py` to convert and markov process into state-action structure.
>> * gamma: discount factor
>> * time: iteration time
>> * eps: epsilon paramter in Epsilon-Greedy Policy, default 0.4

> Return:
>> * updated states list `[State]`

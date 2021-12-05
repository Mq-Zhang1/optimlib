# Main Content
This is a python code collection for widely used optimization algorithms. 
Main otpimization types:

1. Descent Search 
	* Gradient descent, Newton, Momentum, Nesterov etc.
	* Linear Search, Trust Region and Conjugate Descent
2. Stochastic Search:
	* Simulated Annealing, Cross-Entropy Methods, Search Gradient
3. Classic Search:
	* BFS, DFS, Dijkstra, A*
	* Adversial Search (Minimax, Expectimax)
	* RRT (n-dimensional)
4. MDP:
	* Value Iteration, Policy Iteration
5. Reinforcement Learning
	* Monte Carlo Policy Evaluation, Temporal Difference Policy Evaluation, Q-learning

# Environment

# Usage
1. Clone the repo

	> git clone <https://github.com/Mq-Zhang1/optimlib>

2. Install the required library

	using pip
	
	> pip install -r requirements.txt
	
3. See function usage `HANDBOOK.md` in the folder 

# Test
## Descent Search
Using $f(x) = x^2+y^2$ to test line search, trust region and CG 

* To test line search algorithm:

```
python3 Gradient.py -t 1
```
Terminal output:

~~~python
Linear search final value: 2.230074519853063e-05
Linear Search iteration times: 25
~~~

* To test Trust Region algorithm:

```
python3 Gradient.py -t 2
```
Output:

```
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 1
         Function evaluations: 2
         Gradient evaluations: 2
         Hessian evaluations: 1
```

* To test Conjugate Gradient algorithm:

```
python3 Gradient.py -t 3
```
Output:

```
Optimization terminated successfully.
         Current function value: 0.000000
         Iterations: 2
         Function evaluations: 2
         Gradient evaluations: 2
         Hessian evaluations: 2
```

* To test Descent methods such as Newton, Momentum and RMSprop:

```
python3 Gradient.py -t 4
```
By running this test function, we could intuitively observe the convergence rate of different algorithm. Here choose 5 methods: `Gradient`,`Momentum`,`Nesterov`,`Adagrad`,`RMSprop`

![1](https://github.com/Mq-Zhang1/optimlib/blob/main/image/gradient.mp4 =250x250)

## Stochastic Search
Using Rosenbrock:  $ f(x) = (1-x_1)^2+100(x_2-x_1^2)^2$ as the test function.

* Simulated Annealing
![2](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/sa.mp4)

* Cross Entropy Search
![3](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/cg.mp4)

* Search Gradient
![4](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/sg.mp4)

To test performance and obtain convergence plot:

```
python3 Stochastic.py
```
![5](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/sa_better.png)

## Classical Search
Classical Search includes many important algorithms for solving maze/grid path/traffic problem and various game AI strategies. Many advanced algorithms are now widely used in robot area for navigating etc. Here mainly includes 3 parts:

* Basic Algorithm BFS, DFS, Dijkstra and A\*, Test these method under the classical problem scenario - grid search.
	
	* Dijkstra algorithm
	![6](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/dijstra.mp4)
	* A\* algorithm
	![7](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/A*.mp4)

To get Dijkstra graph, run:
```
python3 ClassicSearch.py -t 1
```
To get A\* graph, run:
```
python3 ClassicSearch.py -t 2
```

* General Path Finding: RRT algorithm, the core idea of this algorithm is quite simple, but it's still a popular way in robot path planning. In this library, the code supports more than 3 dimension search.
	
![8](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/RRT.mp4)

To get above graph, run:
```
python3 ClassicSearch.py -t 3
```

* Adversarial Search: Minimax and Expectimax

	Borrow 2048 Game Agent from <https://github.com/ucsdcsegaoclasses/expectimax>.
	
	Features:
	
	* Here computer Maximum player hopes to get as much score as possible
	* Supports 2 kinds of adversarial players:
		
		1. Minimum player, prevents maximum player to get higher score
		2. Random player, randomly choose game state, no bias

	To play the game:
	
	```
	python playgame.py
	```

* Expectimax Result:
![8](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/expect.mp4)

![9](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/expect.png)

* Minimax Result:
![10](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/minimax.mp4)

![9](/Users/z.mq/Desktop/hw code/CSE 257/Proj/PLot/Minimax.png)
It is obvious that the score for Minimax is much lower than Expectimax

## Markov Decision Process

This part and following reinforcement learning both should be applied to specific problems. 
To test policy iteration and value iteration of a simple 2-state problem,run:

```
python MDP.py
```

Final result:

~~~python
-----Policy Iteration-------
Best policy
s1 a12
s2 None
-----Value Iteration-------
Optimal value
s1 -9.999995180417654
s2 10.0
~~~

## Reinforcement Learning

First part is policy evaluation methods and Q-learning. 

* To test the Monte Carlo Policy Evaluation, run:

```
python3 eValuation.py -t 1
```
Result:

```
s1 -25.05088787927581
s2 10.0
```

* To test the Temporal Difference Policy Evaluation, run:

```
python3 eValuation.py -t 2
```
Result:

```
s1 -29.550134983176296
s2 10.0
```

* To test the Tebular Q learning, run:

```
python3 eValuation.py -t 3
```
Result:

```
s1 -9.875051276658827
s2 10.0
```
Nearly corresponds to the result of MDP

# Overview & Background
### Related surveys
Deep reinforcement learning in power distribution systems: Overview, challenges, and opportunities. [[paper](https://ieeexplore.ieee.org/abstract/document/9372283)][2021]
A Survey on Physics Informed Reinforcement Learning: Review and Open Problems. [[paper](https://arxiv.org/abs/2309.01909)][2023]


# Network reconfiguration and restoration
### Network restoration

- Service Restoration Using Deep Reinforcement Learning and Dynamic Microgrid Formation in Distribution Networks. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10158030)][2023][igder2023service][DQN]
- Distribution Service Restoration With Renewable Energy Sources: A Review. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9858638)][2022]
- Deep Reinforcement Learning From Demonstrations to Assist Service Restoration in Islanded Microgrids. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9705112)][2022][DDPG]
- Hierarchical Combination of Deep Reinforcement Learning and Quadratic Programming for Distribution System Restoration. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10051723)][2023][hosseini2023hierarchical][DQN]
- Hybrid imitation learning for real-time service restoration in resilient distribution systems. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9424985)][2021][zhang2021hybrid][DQN]
- Curriculum-based reinforcement learning for distribution system critical load restoration. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9903581)][2022][zhang2022curriculum][PPO]
- Resilient Operation of Distribution Grids Using Deep Reinforcement Learning. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9445634)][2021][DDPG]
### Network reconfiguration

- Distribution system resilience under asynchronous information using deep reinforcement learning. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9345996)][2021]


# Crew dispatch

- Toward a synthetic model for distribution system restoration and crew dispatch. [2019]
- Power distribution system outage management with co-optimization of repairs, reconﬁguration, and DG dispatch. [2018]
- Dynamic restoration of large-scale distribution network contingencies: Crew dispatch assessment. [2007]
- Dynamic restoration of active distribution networks by coordinated repair crew dispatch and cold load pickup. [2023]
- A multiobjective algorithm to determine patrol sequences for out-of-service nodes in power distribution feeders. [2021]
- Post-storm repair crew dispatch for distribution grid restoration using stochastic Monte Carlo tree search and deep neural networks. [2023]
# Volt-VAR control

- A graph policy network approach for Volt-Var Control in power distribution systems. [[paper](https://www.sciencedirect.com/science/article/pii/S0306261922008479)][2022][lee2022graph]
   - We propose a framework that combines RL with graph neural networks and study the benefits and limitations of graph-based policy in the [VVC](https://www.sciencedirect.com/topics/engineering/volt-var-control) setting.
- Model-augmented safe reinforcement learning for Volt-VAR control in power distribution networks. [[paper](https://www.sciencedirect.com/science/article/pii/S0306261922002148)][2022][gao2022model]
   - To improve the algorithm’s performance when learning from limited data, we propose a novel mutual information [regularization](https://www.sciencedirect.com/topics/engineering/regularization) neural network for the safety layer.
- Two-Stage Volt/Var Control in Active Distribution Networks With Multi-Agent Deep Reinforcement Learning Method. [[paper](https://ieeexplore.ieee.org/abstract/document/9328796)][2021][sun2021two]
   - We proposes a two-stage deep reinforcement learning (DRL)-based real-time VVC method to mitigate fast voltage violation while minimizing the network power loss.
   - The real-time VVC problem is formulated and solved using a multi-agent deep deterministic policy gradient (MADDPG) method, which features offline centralized training and online decentralized application.
- Online Multi-Agent Reinforcement Learning for Decentralized Inverter-Based Volt-VAR Control. [[paper](https://ieeexplore.ieee.org/abstract/document/9356806)][2021][liu2021online]
   - In this framework, the VVC problem is formulated as a constrained Markov game and we propose a novel multi-agent constrained soft actor-critic (MACSAC) reinforcement learning algorithm. MACSAC is used to train the control agents online, so the accurate ADN model is no longer needed.
- Data-Driven Affinely Adjustable Robust Volt/VAr Control. [[paper](https://ieeexplore.ieee.org/abstract/document/10108906)][2023]
   -  To achieve a fast and accurate estimation of voltage sensitivities, we propose a data-driven method based on deep neural network (DNN), together with a rule-based bus-selection process using the bidirectional search method. Our method only uses the operating statuses of selected buses as inputs to DNN, thus significantly improving the training efficiency and reducing information redundancy. Finally, a distributed consensus-based solution, based on the alternating direction method of multipliers (ADMM), for the AARVVC is applied to decide the inverter’s reactive power adjustment rule with respect to its active power. Only limited information exchange is required between each local agent and the central agent to obtain the slope of the reactive power adjustment rule, and there is no need for the central agent to solve any (sub)optimization problems.
- Consensus Multi-Agent Reinforcement Learning for Volt-VAR Control in Power Distribution Networks. [[paper](https://ieeexplore.ieee.org/abstract/document/9353702)][2021][gao2021consensus]
   - we propose consensus multi-agent deep reinforcement learning algorithm to solve the VVC problem, which determines the operation schedules for voltage regulators, on-load tap changers, and capacitors. The VVC problem is formulated as a networked multi-agent Markov decision process, which is solved using the maximum entropy reinforcement learning framework and a novel communication-efficient consensus strategy. The proposed algorithm allows individual agents to learn a group control policy using local rewards.
- Robust Data-Driven and Fully Distributed Volt/VAR Control for Active Distribution Networks With Multiple Virtual Power Plants. [[paper](https://ieeexplore.ieee.org/abstract/document/9754708)][2022][li2022robust]
   - The proposed scheme combines the advantages of a robust regression based feedback optimization algorithm and a revised alternating direction multiplier method (ADMM).
   - learning assisted optimization
   - Firstly, we use the recently proposed robust hierarchical-optimization recursive least squares (HRLS) regression method [30] to approximate the real-time system input-output (or controlstate) response based on partial observation of ADNs. Then, the solution of the robust regression is embedded in the iterative data-driven optimization scheme to form a uniﬁed real-time VVC framework, where feedback control is combined with the ADMM framework to implement fully distributed optimization.
   - A tuned ADMM method incorporated gradient projection algorithm is proposed for the data-driven optimization paradigm.


# Robust Optimization
Data-driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable Reformulations. [[paper](https://arxiv.org/abs/1505.05116)][2015][Robust Optimization]
# Learning-assisted Optimization

Learning to Branch in Mixed Integer Programming. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10080)][2016]
Learning to Optimize. [[paper](https://arxiv.org/abs/1606.01885)][2016]

- Li et al. formulate this as a reinforcement learning problem, in which any optimization algorithm can be represented as a policy

Learning Combinatorial Optimization Algorithms over Graphs. [[paper](https://proceedings.neurips.cc/paper/2017/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html)][2017]

- Elias Khalil et al. propose a unique combination of reinforcement learning and graph embedding  to learn heuristic algorithms that exploit the structure for NP-hard combinatorial optimization problems. 

Exact Combinatorial Optimization with Graph Convolutional Neural Networks. [[paper](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html)][2019]
Solving Mixed Integer Programs Using Neural Networks. [[paper](https://arxiv.org/abs/2012.13349)][2020]

- We propose a new graph convolutional neural network model for learning branch-and-bound variable selection policies, which leverages the natural variable-constraint bipartite graph representation of mixed-integer linear programs.

Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems. [[paper](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2020.0976)][2020]

- propose a number of machine learning techniques to effectively extract information from previously solved instances in order to significantly improve the computational performance of MIP solvers when solving similar instances in the future. 

A Survey of Learning‑Based Intelligent Optimization Algorithms. [[paper](https://link.springer.com/article/10.1007/s11831-021-09562-1)][2021]
Machine learning for combinatorial optimization: A methodological tour d’horizon. [[paper](https://www.sciencedirect.com/science/article/pii/S0377221720306895)][2021]
Deep Policy Dynamic Programming for Vehicle Routing Problems. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-08011-1_14)][2022]

- DPDP prioritizes and restricts the DP state space using a policy derived from a deep neural network, which is trained to predict edges from example solutions.

DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html)][2022]
Learning to Branch with Tree MDPs. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html)][2022]
A survey for solving mixed integer programming via machine learning. [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222014035)][2023]
Mixed-Integer Optimization with Constraint Learning. [[paper](https://pubsonline.informs.org/doi/full/10.1287/opre.2021.0707)][2023]
Combinatorial Optimization and Reasoning with Graph Neural Networks. [[paper](https://www.jmlr.org/papers/v24/21-0449.html)][2023]

# Demand response
Hierarchical distributed multi-energy demand response for coordinated operation of building clusters. [[paper](https://www.sciencedirect.com/science/article/pii/S0306261921016068)][2022]

# Distribution system optimal power ﬂow

- Compact Optimization Learning for AC Optimal Power Flow. [[paper](https://arxiv.org/abs/2301.08840)][2023][E2E]
- End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch[[paper](https://arxiv.org/abs/2304.11726)][2023][E2E]
- Learning to Optimize: Accelerating Optimal Power Flow via Data-driven Constraint Screening. [[paper](https://arxiv.org/abs/2312.07276)][2023][Learning-assist Optimization]
- Optimal Power Flow Based on Physical-Model-Integrated Neural Network with Worth-Learning Data Generation. [[paper](https://arxiv.org/abs/2301.03766)][physics-informed]
- Learning Regionally Decentralized AC Optimal Power Flows With ADMM. [[paper](https://ieeexplore.ieee.org/abstract/document/10057067)][2023][Learning-assisted Optimization]
- Learning-Aided Asynchronous ADMM for Optimal Power Flow. [[paper](https://ieeexplore.ieee.org/abstract/document/9573286)][2021][Learning-assisted Optimization]

# Physics-informed learning

- Data-driven optimal power ﬂow: A physics-informed machine learning approach. [2020]
- Applications of Physics-Informed Neural Networks in Power Systems - A Review. [2022]
- DeepOPF: A Feasibility-Optimized Deep Neural Network Approach for AC Optimal Power Flow Problems. [2022]
- Fast Inverter Control by Learning the OPF Mapping Using Sensitivity-Informed Gaussian Processes. [2022]
- A convex neural network solver for DCOPF with generalization guarantees. [2021]
- High-ﬁdelity machine learning approximations of large-scale optimal power ﬂow. [2020]
- Learning to solve AC optimal power ﬂow by differentiating through holomorphic embeddings. [2020]
- Physics-informed neural networks for AC optimal power ﬂow. [2022]
- Physics-Informed Graphical Representation-Enabled Deep Reinforcement Learning for Robust Distribution System Voltage Control. [2023]
- Physics-Informed Multi-Agent deep reinforcement learning enabled distributed voltage control for active distribution network using PV inverters. [2024]
- Physical-assisted multi-agent graph reinforcement learning enabled fast voltage regulation for PV-rich active distribution network. [2023]
- Constraint learning-based optimal power dispatch for active distribution networks with extremely imbalanced data. [2023]



# Physics-informed RL

- A Survey on Physics Informed Reinforcement Learning: Review and Open Problems. [2023]

# Robust learning against adversarial attacks

- Two-stage Deep Reinforcement Learning for Inverter-based Volt-VAR Control in Active Distribution Network.
- Improving Robustness of Reinforcement Learning for Power System Control with Adversarial Training.




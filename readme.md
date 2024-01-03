# Overview & Background
## Related surveys

- Deep reinforcement learning in power distribution systems: Overview, challenges, and opportunities. [[paper](https://ieeexplore.ieee.org/abstract/document/9372283)][2021]
- A Survey on Physics Informed Reinforcement Learning: Review and Open Problems. [[paper](https://arxiv.org/abs/2309.01909)][2023]
- A Survey on Physics Informed Reinforcement Learning: Review and Open Problems. [[paper](https://arxiv.org/abs/2309.01909)][2023]
- A survey for solving mixed integer programming via machine learning. [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222014035)][2023]

# Summary of Use cases
## Network reconfiguration and restoration

- Service Restoration Using Deep Reinforcement Learning and Dynamic Microgrid Formation in Distribution Networks. [[paper](https://ieeexplore.ieee.org/abstract/document/10158030)][2023][igder2023service][DQN]
- Distribution Service Restoration With Renewable Energy Sources: A Review. [[paper](https://ieeexplore.ieee.org/abstract/document/9858638)][2022]
- Deep Reinforcement Learning From Demonstrations to Assist Service Restoration in Islanded Microgrids. [[paper](https://ieeexplore.ieee.org/abstract/document/9705112)][2022][DDPG]
- Hierarchical Combination of Deep Reinforcement Learning and Quadratic Programming for Distribution System Restoration. [[paper](https://ieeexplore.ieee.org/abstract/document/10051723)][2023][hosseini2023hierarchical][DQN]
- Hybrid imitation learning for real-time service restoration in resilient distribution systems. [[paper](https://ieeexplore.ieee.org/abstract/document/9424985)][2021][zhang2021hybrid][DQN]
- Curriculum-based reinforcement learning for distribution system critical load restoration. [[paper](https://ieeexplore.ieee.org/abstract/document/9903581)][2022][zhang2022curriculum][PPO]
- Resilient Operation of Distribution Grids Using Deep Reinforcement Learning. [[paper](https://ieeexplore.ieee.org/abstract/document/9445634)][2021][DDPG]
- Distribution system resilience under asynchronous information using deep reinforcement learning. [[paper](https://ieeexplore.ieee.org/abstract/document/9345996)][2021]
## Crew dispatch

- Toward a synthetic model for distribution system restoration and crew dispatch. [[paper](https://ieeexplore.ieee.org/abstract/document/8587140)][2019]
- Power distribution system outage management with co-optimization of repairs, reconﬁguration, and DG dispatch. [[paper](https://ieeexplore.ieee.org/abstract/document/7812566)][2018]
- Dynamic restoration of large-scale distribution network contingencies: Crew dispatch assessment. [[paper](https://ieeexplore.ieee.org/abstract/document/4538529)][2007]
- Dynamic restoration of active distribution networks by coordinated repair crew dispatch and cold load pickup. [[paper](https://ieeexplore.ieee.org/abstract/document/10234113)][2023]
- A multiobjective algorithm to determine patrol sequences for out-of-service nodes in power distribution feeders. [[paper](https://www.sciencedirect.com/science/article/pii/S0378779621001796)][2021]
- Post-storm repair crew dispatch for distribution grid restoration using stochastic Monte Carlo tree search and deep neural networks. [[paper](https://www.sciencedirect.com/science/article/pii/S0142061522004847)][2023]
## Volt-VAR control

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
- Consensus Multi-Agent Reinforcement Learning for Volt-VAR Control in Power Distribution Networks. [[paper](https://ieeexplore.ieee.org/abstract/document/9353702)][2021][gao2021consensus]
   - we propose consensus multi-agent deep reinforcement learning algorithm to solve the VVC problem, which determines the operation schedules for voltage regulators, on-load tap changers, and capacitors. 
- Robust Data-Driven and Fully Distributed Volt/VAR Control for Active Distribution Networks With Multiple Virtual Power Plants. [[paper](https://ieeexplore.ieee.org/abstract/document/9754708)][2022][li2022robust][learning assisted optimization]
   - The proposed scheme combines the advantages of a robust regression based feedback optimization algorithm and a revised alternating direction multiplier method (ADMM).
   - A tuned ADMM method incorporated gradient projection algorithm is proposed for the data-driven optimization paradigm.

## Demand response
Hierarchical distributed multi-energy demand response for coordinated operation of building clusters. [[paper](https://www.sciencedirect.com/science/article/pii/S0306261921016068)][2022]

## Distribution system optimal power ﬂow

- Compact Optimization Learning for AC Optimal Power Flow. [[paper](https://arxiv.org/abs/2301.08840)][2023][E2E]
- End-to-End Feasible Optimization Proxies for Large-Scale Economic Dispatch[[paper](https://arxiv.org/abs/2304.11726)][2023][E2E]
- Learning to Optimize: Accelerating Optimal Power Flow via Data-driven Constraint Screening. [[paper](https://arxiv.org/abs/2312.07276)][2023][Learning-assist Optimization]
- Optimal Power Flow Based on Physical-Model-Integrated Neural Network with Worth-Learning Data Generation. [[paper](https://arxiv.org/abs/2301.03766)][physics-informed]
- Learning Regionally Decentralized AC Optimal Power Flows With ADMM. [[paper](https://ieeexplore.ieee.org/abstract/document/10057067)][2023][Learning-assisted Optimization]
- Learning-Aided Asynchronous ADMM for Optimal Power Flow. [[paper](https://ieeexplore.ieee.org/abstract/document/9573286)][2021][Learning-assisted Optimization]

# Summary of Algorithms
## Learning-assisted Optimization Theory

- Learning to Branch in Mixed Integer Programming. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10080)][2016]
- Learning to Optimize. [[paper](https://arxiv.org/abs/1606.01885)][2016]
   - Li et al. formulate this as a reinforcement learning problem, in which any optimization algorithm can be represented as a policy
- Learning Combinatorial Optimization Algorithms over Graphs. [[paper](https://proceedings.neurips.cc/paper/2017/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html)][2017]
   - Elias Khalil et al. propose a unique combination of reinforcement learning and graph embedding  to learn heuristic algorithms that exploit the structure for NP-hard combinatorial optimization problems. 
- Exact Combinatorial Optimization with Graph Convolutional Neural Networks. [[paper](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html)][2019]
- Solving Mixed Integer Programs Using Neural Networks. [[paper](https://arxiv.org/abs/2012.13349)][2020]
   - We propose a new graph convolutional neural network model for learning branch-and-bound variable selection policies, which leverages the natural variable-constraint bipartite graph representation of mixed-integer linear programs.
- Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems. [[paper](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2020.0976)][2020]
   - propose a number of machine learning techniques to effectively extract information from previously solved instances in order to significantly improve the computational performance of MIP solvers when solving similar instances in the future. 
- A Survey of Learning‑Based Intelligent Optimization Algorithms. [[paper](https://link.springer.com/article/10.1007/s11831-021-09562-1)][2021]
- Machine learning for combinatorial optimization: A methodological tour d’horizon. [[paper](https://www.sciencedirect.com/science/article/pii/S0377221720306895)][2021]
- Deep Policy Dynamic Programming for Vehicle Routing Problems. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-08011-1_14)][2022]
   - DPDP prioritizes and restricts the DP state space using a policy derived from a deep neural network, which is trained to predict edges from example solutions.
- DIMES: A Differentiable Meta Solver for Combinatorial Optimization Problems. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a3a7387e49f4de290c23beea2dfcdc75-Abstract-Conference.html)][2022]
- Learning to Branch with Tree MDPs. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html)][2022]
- Mixed-Integer Optimization with Constraint Learning. [[paper](https://pubsonline.informs.org/doi/full/10.1287/opre.2021.0707)][2023]
- Combinatorial Optimization and Reasoning with Graph Neural Networks. [[paper](https://www.jmlr.org/papers/v24/21-0449.html)][2023]

## Physics-informed learning

- Data-driven optimal power ﬂow: A physics-informed machine learning approach. [[paper](https://ieeexplore.ieee.org/abstract/document/9115822)][2020]
- Applications of Physics-Informed Neural Networks in Power Systems - A Review. [[paper](https://ieeexplore.ieee.org/abstract/document/9743327/)][2022]
- DeepOPF: A Feasibility-Optimized Deep Neural Network Approach for AC Optimal Power Flow Problems. [[paper](https://ieeexplore.ieee.org/abstract/document/9894104/)][2022]
- Fast Inverter Control by Learning the OPF Mapping Using Sensitivity-Informed Gaussian Processes. [[paper](https://ieeexplore.ieee.org/abstract/document/9905715/)][2022]
- A convex neural network solver for DCOPF with generalization guarantees. [[paper](https://ieeexplore.ieee.org/abstract/document/9599403/)][2021]
- High-ﬁdelity machine learning approximations of large-scale optimal power ﬂow. [[paper](https://arxiv.org/abs/2006.16356)][2020]
- Learning to solve AC optimal power ﬂow by differentiating through holomorphic embeddings. [[paper](https://arxiv.org/abs/2012.09622)][2020]
- Physics-informed neural networks for AC optimal power ﬂow. [[paper](https://www.sciencedirect.com/science/article/pii/S0378779622005636)][2022]
- Physics-Informed Graphical Representation-Enabled Deep Reinforcement Learning for Robust Distribution System Voltage Control. [[paper](https://ieeexplore.ieee.org/abstract/document/10113230/)][2023]
- Physics-Informed Multi-Agent deep reinforcement learning enabled distributed voltage control for active distribution network using PV inverters. [[paper](https://www.sciencedirect.com/science/article/pii/S0142061523006981)][2024]
- Physical-assisted multi-agent graph reinforcement learning enabled fast voltage regulation for PV-rich active distribution network. [[paper](https://www.sciencedirect.com/science/article/pii/S0306261923011078)][2023]
- Constraint learning-based optimal power dispatch for active distribution networks with extremely imbalanced data. [[paper](https://ieeexplore.ieee.org/abstract/document/10375977/)][2023]



# Robust learning against adversarial attacks

- Two-stage Deep Reinforcement Learning for Inverter-based Volt-VAR Control in Active Distribution Network. [[paper](https://ieeexplore.ieee.org/abstract/document/9274529/)][2020]
- Improving Robustness of Reinforcement Learning for Power System Control with Adversarial Training. [[paper](https://arxiv.org/abs/2110.08956)][2021]

# Robust Optimization
Data-driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable Reformulations. [[paper](https://arxiv.org/abs/1505.05116)][2015][Robust Optimization]


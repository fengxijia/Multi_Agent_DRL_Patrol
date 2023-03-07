# MARL_Patrol

The multi-robot patrolling problem (MRPP) is a surveillance task that employs multiple robots to visit every point of interest (POI) as frequently as possible, which requires cooperative decision-making between agents to optimize the group performance. MRPP can be useful for many intriguing applications such as environmental monitoring, searching for objects, and detecting anomalies. 
<!--The patrolling strategies fall into two major categories: deterministic approaches and non-deterministic approaches. The former have guaranteed optimality based on a precomputed deterministic patrol path, but would suffer from being centralized and lack of robustness. Although the latter, which include potential fields, auction-based methods, and swarm intelligence algorithms, are generally more adaptive to changes in the system, they tend to give local-optimal solutions as they greedily optimize their local objectives. Reinforcement learning (RL) is a promising adaptive approach which optimizes a long-term objective and has a natural balance between exploration and exploitation. However, the current RL frameworks for MRPP in literature have poor scalability due to curse of dimensionality. Therefore, -->
This project simulates the multi-agent patrolling problems. We provide a simulation environment in Python, trained the agents using reinforcement learning to patrol every target places as frequently as possible. Our network model design <!--improves the scalability of the current RL frameworks by ultilizing pointer network, which--> can be trained under different environmental settings like different number of agents and targets.

Regarding the decision timing of the agents, we provide two versions of codes, one is to simulate the situation where agents make decisions at every simulation time step, the other is to simulate agents making decisions when there are on the targets.

The current RL algorithm implemented is Proximal Policy Optimization (PPO).
<!--The challenge lies in finding effective representations for the state space and the action space. This project will explore how to refine the reward shaping as well as how to utilize inter-agent communication for better cooperation. Moreover, the proposed approach will be compared with other existing benchmark approaches-->

## Installation

### Prerequisites

Dependencies you need to install.

```
gym==0.23.1
imageio==2.9.0
matplotlib==3.4.3
networkx==2.6.3
numpy==1.20.3
scikit_learn==1.1.1
torch==1.11.0
```

### Installing

There are two ways to install virtual environments. One is from yml file, the other is from requirements.txt.

* From yml file

```
conda env create -n NEW_NAME -f patrol.yml
```
You can name the virtual environment by replacing NEW_NAME with any name you want.

* From requirements.txt

Run the requirements.txt file inside the code folder to install dependencies.
```
pip install -r requirements.txt
```

## Train the model

Training from scratch:

```
python single_proc_ppo_patrol.py
```

Training from weights of default model (suffix: v2, when training is over without interruption, the best model will be named v2):

```
python single_proc_ppo_patrol.py --load 
```

Loaded from certain model:

```
python single_proc_ppo_patrol.py --load --suffix YOUR_VERSION_NUMBER
```

## Running the tests

Display simulations while testing:

```
python single_proc_ppo_patrol.py --test --render --suffix TRAINING_MODEL_VERSION_NUMBER
```

<!--## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 
-->
## Authors

<!--* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)-->
* **Xijia Feng**
* **Jiawei Cao**
* **Yiding Ma**

<!--See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.-->

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

<!--## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc-->

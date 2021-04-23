## Deep Reinforcement Learning Nano Degree
### Project 1 - Navigation

This repository contains the solution for the Project 1 of the Deep Reinforcement Learning Nano Degree from Udacity. The goal of this project is to train an reinforcement learning agent to navigate and collect bananas through Deep Q-Learning.

The Banana environment consists in a large square world, where we aim at collecting yellow bananas, and avoid blue banana. The agent nagivates through the world by moving forward, left, right or backwards. Each yellow bana collects gives a +1 reward, while a blue banana gives a -1 reward. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Place the file in the `drlnd-p1` GitHub repository, in the `navigation/banana` folder, and unzip (or decompress) the file.

2. Install the dependencies
```
cd drlnd-p1/python
pip install .
```

### Reproduce
Open and execute the `navigation/Report.ipynb` notebook.

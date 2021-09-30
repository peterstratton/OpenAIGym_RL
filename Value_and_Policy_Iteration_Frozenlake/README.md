# Value_and_Policy_Iteration_FrozenLake

This repository contains implementations of value iteration and policy iteration that are ran in the  FrozenLake Open AI gym environment: https://gym.openai.com/envs/FrozenLake8x8-v0/. Gym environments are test problems that all contain a general interface. This provides a space to run and test a variety of different types of RL algorithms. For more info on gym environments, check here: https://gym.openai.com/docs/. 

## Installation
First, install Python. To install python, check here: https://www.python.org/downloads/.

Second, install the required packages. Commands to install them are below:
```
pip install numpy
pip install gym
```

Third, clone the repo using the command:
```
git clone https://github.com/peterstratton/Value_and_Policy_Iteration_FrozenLake.git
```

## Run
To run the code, simply cd into the cloned repo and run:
```
python main.py
```
This will run both policy iteration and value iteration and display their results.

## Output 
The output of the program displays the percentage of successful and unsuccesful completions of the environment and the mean total reward accumulated over a single run for the policies obtained through value and policy iteration. Before running the program, the user has the option to render the environment, which will print the environment to the console window each time an action is taken.

![](https://github.com/peterstratton/Value_and_Policy_Iteration_FrozenLake/blob/main/results_pictures/rendered_env_frozenlake.png)
![](https://github.com/peterstratton/Value_and_Policy_Iteration_FrozenLake/blob/main/results_pictures/vp_run_stats.png)  





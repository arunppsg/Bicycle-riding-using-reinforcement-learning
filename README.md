# Learning to ride a bicycle using reinforcement learning and shaping

This repository is a work in progress implementation of: 
[Learning to Drive a Bicycle Using Reinforcement Learning and Shaping](https://www.researchgate.net/publication/221346431_Learning_to_Drive_a_Bicycle_Using_Reinforcement_Learning_and_Shaping) I am not affiliated with the authors of the paper. 

## code organization
The file main.py contains all the code. The Bicycle class describes the environment. The agent sends the action to the environment through the method Bicycle.take_action and the method returns next state, reward and is_end. 

## Progress: 

- [x] Implement Bicycle Environment
- [x] Implement Model in PyTorch
- [x] Run episodes 
- [ ] Obtain Convergence

The work is in progress to obtain convergence.

## References:
[Learning to ride bicycle using reinforcement learning and shaping](https://www.researchgate.net/publication/221346431_Learning_to_Drive_a_Bicycle_Using_Reinforcement_Learning_and_Shaping) 
[Bicycle environment](https://github.com/amarack/python-rl) 
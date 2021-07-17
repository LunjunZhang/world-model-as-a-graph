# World Model as a Graph

This is the code accompanying the paper: **World Model as a Graph: 
Learning Latent Landmarks for Planning** (ICML 2021 Long Presentation). 

By [Lunjun Zhang](http://www.cs.toronto.edu/~lunjun/), [Ge Yang](https://scholar.google.com/citations?user=vaQcF6kAAAAJ), 
[Bradly Stadie](https://bstadie.github.io/).

A link to our paper can be found on [arXiv](https://arxiv.org/abs/2011.12491).

Videos / blog can be found on our [website](https://sites.google.com/view/latent-landmarks/).

## Overview

![image info](./figures/l3p.png)

Model-based RL agents today plan using step-by-step virtual rollouts in a learned dynamics model. 
This type of world model quickly diverges from reality as the planning horizon increases. 
Humans, however, can do planning by **analyzing the structure of a problem in the large** 
and **decomposing it into simpler sub-problems**. 
How can we teach learning agents to do something similar? 

We enhance RL agents’ ability to do temporally extended reasoning by learning a graph-structured world model 
composed of sparse, multi-step transitions. 
We show how to **learn both the nodes and the edges** on the graph together with a goal-conditioned policy, 
and how to better leverage **temporal abstraction** in online planning.

## Instructions

Requirements: 
```
python==3.7.4
numpy==1.19.5
torch==1.5.1+cu101
tensorflow==1.13.1
gym==0.13.1
mpi4py==3.0.3
mujoco_py==2.0.2.13
pandas==1.1.1
```

The implementation of 
<img src="https://latex.codecogs.com/gif.latex?L^{3}P " /> 
 is provided in the `rl` folder.

The training scripts are provided in `scripts` folder. 

## Citations

Please cite our paper as:

```
@inproceedings{zhang2021worldmodel,
  title={World model as a graph: Learning latent landmarks for planning},
  author={Zhang, Lunjun and Yang, Ge and Stadie, Bradly C},
  booktitle={International Conference on Machine Learning},
  pages={12611--12620},
  year={2021},
  organization={PMLR}
}
```

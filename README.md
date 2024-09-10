# IR2 Multi-Robot RL Exploration

[IROS 2024] Public code and model for [IR2: Implicit Rendezvous for Robotic Exploration Teams under Sparse Intermittent Connectivity](INSERT_LINK_HERE).

We present IR2, a deep reinforcement learning approach to information sharing for multi-robot exporation under communication constraints. Leveraging attention-based neural networks and hierarchical graph formulation, robots can effectively balance the longer-term trade-offs between disconnecting for solo exploration and reconnecting for information sharing in large-scale, complex environments. 


## GIF Demo

This demonstration showcases 4 robots exploring in an unknown `Complex` map under line-of-sight signal strength communication constraints. The top gif illustrates the global map and robot positions assuming no communication constraints. Conversely, the bottom 4 gifs illustrates the individual robots' map and position beliefs subjected to communication constraints. 

If this GIF is taking too long to load, you may view the demonstration [here](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/master/gifs/combined_gif_demo.gif).
<p align="center">
  <img src="gifs/combined_gif_demo.gif" style="width: 750px; height: auto;" />
</p>





## Instructions


#### Dependencies
This repository was tested using the following dependencies. Newer version of these packages may work as well. 
* `python == 3.8`
* `pytorch == 1.10.0`
* `ray == 1.10.0`
* `scikit-image == 0.19.3`
* `scikit-learn == 1.2.1`
* `scipy == 1.10.0`
* `matplotlib == 3.6.3`
* `tensorboard == 2.8.0`


#### Training
1. Set training parameters in `parameters.py`.
2. Run python `driver.py`.


#### Evaluation
1. Set inference parameters in `test_parameters.py`.
2. Run `test_driver.py`.



## Code Structure

* `parameter.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the local network.
* `multi_robot_worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `graph_generator.py` Generate and update the collision-free graph.
* `graph.py` Graph definition and utilities.
* `node.py` Initialize and update nodes in the coliision-free graph.
* `sensor.py` Simulate the sensor model of Lidar.
* `robot.py` Acts as a replay buffer.
* `ss_realistic_model.py` Realistic signal strength communication model.
* `/model` Trained model.
* `/DungeonMaps` Maps of training environments.



## References
If you intend to use our work in your research, please cite the following publication:
```
@INPROCEEDINGS{derek2024IR2,
  author={Derek, MS Tan and Ma, Yixiao and Liang, Jingsong and Cao, Yuhong and Sartoretti, Guillaume},
  booktitle={2024 IEEE International Conference on Intelligent Robots and Systems (IROS)}, 
  title={IR2: Implicit Rendezvous for Robotic Exploration Teams under Sparse Intermittent Connectivity}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}}
```


## Authors
[Derek Ming Siang Tan](https://github.com/derektan95)\
[Yixiao Ma](https://yixiao-m.github.io)\
[Jingsong Liang](https://github.com/JingsongLiang)\
[Yi Cheng Chng](https://github.com/chngyicheng)\
[Yuhong Cao](https://github.com/caoyuhong001)\
[Guillaume Sartoretti](https://github.com/gsartoretti)

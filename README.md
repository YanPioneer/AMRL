# The code of AMRL. You can follow the steps to make it work.

#### Dependencies

- gym==0.14.0
- matplotlib==3.5.1
- mujoco-py==1.50.1.68
- numpy==1.21.5
- protobuf==3.19.4
- tensorboard==1.14.0
- tensorboardX==2.5.1
- torch==1.2.0

### Usage

All codes are tested under **PyTorch 1.2.0** and **Python 3.7.11** on **Ubuntu 20.04** and **Windows 10**.

### Training

- #### step 1: train meta reinforcement learning

​	`./01_MetaReinforcementLearning.py`

- #### step 2:Build buffer pool of source domain data and target domain data

​	`./04_ant_goal_buffer.py`

- #### step 3:Use active query to fine tune the model of specific tasks

​	`./02_ant_goal_AL.py`

### Testing

​	`./03_MRL_Test.py`

import os

import numpy as np

from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from configs.default import default_config
from rlkit.core import logger


def get_action():
    action_list = []
    num_split = 8
    for i in range(0, 30000):
        np.random.seed(1 + i)
        action_random = np.array(np.double(np.random.uniform(-1.0, 1.0, size=(num_split,))))
        action_list.append(action_random)
    return action_list


def get_extra_data_to_save(replay_buffer):
    """
    Save things that shouldn't be saved every snapshot but rather
    overwritten every time.
    :param epoch:
    :return:
    """
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%  replay buffer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    data_to_save = dict()
    data_to_save['replay_buffer'] = replay_buffer
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%% data ok %%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    return data_to_save


def init_env(variant, action):
    
    paths = []
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    print('tasks:', tasks)
    train_tasks = list(tasks[:20])
    ant_buffer = MultiTaskReplayBuffer(
            1000000,
            env,
            train_tasks,
        )
    for j in range(0, 15):
        
        env.reset_task(j)
        print('*'*30)
        print(j)
        print('*'*30)

        for i in range(0, 10):
            observations = []
            rewards = []
            terminals = []
            actions = []
            agent_infos = []
            env_infos = []
            o = env.reset()
            next_o = None
            path_length = 0
            while path_length < 200:
                a = action[i*200+path_length+j*2000]
                agent_info = {}
                next_o, r, d, env_info = env.step(a)
                # update the agent's current context
                observations.append(o)
                rewards.append(r)
                terminals.append(d)
                actions.append(a)
                agent_infos.append(agent_info)
                path_length += 1
                o = next_o
                env_infos.append(env_info)

            actions = np.array(actions)
            if len(actions.shape) == 1:
                actions = np.expand_dims(actions, 1)
            observations = np.array(observations)
            if len(observations.shape) == 1:
                observations = np.expand_dims(observations, 1)
                next_o = np.array([next_o])
            next_observations = np.vstack(
                (
                    observations[1:, :],
                    np.expand_dims(next_o, 0)
                )
            )  # 将两个数组按垂直方向叠加
            weights = np.ones(np.array(rewards).shape)  # 适应性权重
            path = dict(
                observations=observations,
                actions=actions,
                rewards=np.array(rewards).reshape(-1, 1),
                weights=np.array(weights).reshape(-1, 1),
                next_observations=next_observations,
                terminals=np.array(terminals).reshape(-1, 1),
                agent_infos=agent_infos,
                env_infos=env_infos,
            )
            paths.append(path)
        ant_buffer.add_paths(j, paths)
    logger.save_extra_data(get_extra_data_to_save(ant_buffer), path='ant_goal_20_reply_buffer')

if __name__ == "__main__":
    print('start')
    variants = default_config
    actions = get_action()
    init_env(variants, actions)

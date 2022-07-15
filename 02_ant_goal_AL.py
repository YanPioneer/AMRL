import pickle
import numpy as np
from numpy import matrix
import sklearn.metrics
from cvxopt import matrix, solvers
import pandas as pd
from scipy.stats import spearmanr
import os
import pathlib
import numpy as np
import click
import json

import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


if __name__ == "__main__":
    ######################################################################################################################
    ##                                          1、the path of the model
    ######################################################################################################################
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    training_model_dir = os.path.join(father_path,
                                        'output',
                                        'ant-goal',
                                        '2021_12_21_10_22_11',  # meta-rl
                                      )
    ######################################################################################################################
    ##                                          2、load meta training model
    ######################################################################################################################
    variant = default_config

    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    ### instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params'][
        'use_next_obs_in_context'] else obs_dim + action_dim + reward_dim  # (o,a,o',r)
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    print("obs_dim", obs_dim)
    print("action_dim:", action_dim)
    print("latent_dim:", latent_dim)
    print("obs_dim + action_dim + latent_dim:", obs_dim + action_dim + latent_dim)
    print("obs_dim + latent_dim:", obs_dim + latent_dim)

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,  # s,a,z
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,  # s,a,z
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,  # s,z
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    if training_model_dir is not None:
        path = training_model_dir
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id,
                                      base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    ######################################################################################################################
    ###                                         3、Active query
    ######################################################################################################################

    trained_task = list(range(0, 15, 1))
    for i in trained_task:
        training_model_dir = os.path.join('/buffer-path')
        data = pickle.load(open(os.path.join(training_model_dir), 'rb'))
        algorithm.replay_buffer.task_buffers[i] = data["replay_buffer"]
        algorithm.enc_replay_buffer.task_buffers[i] = data["replay_buffer"]

    # SU
    su_initial_size = 800

    for key in trained_task:
        batch_data = algorithm.replay_buffer.task_buffers[key].random_batch_del(su_initial_size)
        algorithm.replay_buffer_SU.add_path(key, batch_data)

    TU_father_path = 'XXXX'
    TU_data_dir = os.path.join(TU_father_path, 'buffer')

    # TU
    for k in range(15, 16):
        TU_data = pickle.load(open(os.path.join(TU_data_dir, 'ant_goal_differ_buffer' + str(k) + '.pkl'), 'rb'))
        algorithm.replay_buffer_TU.task_buffers[k] = TU_data["replay_buffer"]
        # re-train meta-rl model
        algorithm.mmd_meta_testing(k, trained_task, 5, 40, nQ=120)


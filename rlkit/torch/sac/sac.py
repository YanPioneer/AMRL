import heapq
import math
import random
import time

import sklearn.metrics
from cvxopt import matrix, solvers
import sklearn
from collections import OrderedDict

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

from rlkit.core import logger, eval_util
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        if 'weights' in batch.keys():  #????????
            w = batch['weights'][None, ...]
            return [o, a, r, w, no, t]
        else:
            return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_sac_meta(self, indices, source_idx):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        print(self.batch_size)
        TL_batch_size = math.floor(self.batch_size * 0.9)
        batches_TL = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=TL_batch_size)) for idx in
                   indices]
        unpacked_TL = [self.unpack_batch(batch) for batch in batches_TL]  # 变成[[o, a, r, no, t],...]
        # 2、采集SL数据
        # neighbor_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        neighbor_indices = source_idx
        for ind in indices:
            if ind in neighbor_indices:
                neighbor_indices.remove(ind)
        SL_batch_size = self.batch_size - TL_batch_size
        SL_batch_size_per_task = math.ceil(SL_batch_size / len(neighbor_indices))
        SL_batch_size_total = 0
        batches_SL_temp = []
        for ind in neighbor_indices:
            if SL_batch_size_total < SL_batch_size:
                if SL_batch_size - SL_batch_size_total < SL_batch_size_per_task:
                    SL_batch_size_per_task = SL_batch_size - SL_batch_size_total
                batches_SL_temp.append(ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(ind, batch_size=SL_batch_size_per_task)))
                SL_batch_size_total = SL_batch_size_total + SL_batch_size_per_task
        batches_SL = [batches_SL_temp[0]]
        for ind in range(1, len(batches_SL_temp)):
            for key in batches_SL_temp[ind].keys():
                batches_SL[0][key] = torch.cat([batches_SL_temp[ind][key], batches_SL[0][key]], dim=0)
        unpacked_SL = [self.unpack_batch(batch2) for batch2 in batches_SL]  # 变成[[o, a, r, no, t],...]
        unpacked = [[]]
        for ind in range(len(unpacked_TL[0])):
            unpacked[0].append(torch.cat([unpacked_TL[0][ind], unpacked_SL[0][ind]], dim=1))
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]  # 变成[ [[o], [a], [r], [no], [t]] , ... ]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]  # 变成[[o,...], [a,...], [r,...], [no,...], [t,...]]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = []
        for idx in indices:
            batch_dict = self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size,
                                                             sequence=self.recurrent)
            del batch_dict['weights']
            batches.append(ptu.np_to_pytorch_batch(batch_dict))
        # batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _do_training_meta(self, indices, source_idx):
        """

        :param indices: 随机挑选的、进行 meta training 的任务列表，选择的meta任务数量为meta_batch=16
        :return:
        """
        mb_size = self.embedding_mini_batch_size  # 32
        num_updates = self.embedding_batch_size // mb_size  # 32 // 32 = 1

        # sample context batch
        context_batch = self.sample_context(indices)
        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size,
                      :]  # context.shape:  torch.Size([16, 32, 15])
            self._take_step_meta_testing(indices, context, source_idx)
            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, weights, next_obs, terms = self.sample_sac(indices)  #??????

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        weights_flat = weights.view(self.batch_size * num_tasks, -1)  #????
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        # vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_loss = self.vf_criterion((weights_flat ** (1 / 2)) * v_pred, (weights_flat ** (1 / 2)) * v_target.detach())  # ????
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        # mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        # std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        mean_reg_loss = (self.policy_mean_reg_weight * weights_flat * (policy_mean ** 2)).mean()
        std_reg_loss = (self.policy_std_reg_weight * weights_flat * (policy_log_std ** 2)).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def _take_step_meta_testing(self, indices, context, source_idx):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, weights, next_obs, terms = self.sample_sac_meta(indices, source_idx)  #??????

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        weights_flat = weights.view(self.batch_size * num_tasks, -1)  #????
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        # vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_loss = self.vf_criterion((weights_flat ** (1 / 2)) * v_pred, (weights_flat ** (1 / 2)) * v_target.detach())  # ????
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        # mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        # std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        mean_reg_loss = (self.policy_mean_reg_weight * weights_flat * (policy_mean ** 2)).mean()
        std_reg_loss = (self.policy_std_reg_weight * weights_flat * (policy_log_std ** 2)).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot

    def mmd_meta_testing(self, task_idx, source_idx, iter, num_iter_step, nQ):
        """
        与 task_idx 上进行 基于主动学习的迁移
        :param task_idx: 目标域任务
        :param iter:更新轮数
        :param num_iter_step:每一轮走的步数
        :return:

        使用分层采样
        """
        s_memory = []
        for i in range(5):
            s_memory.append([])

        for i in range(15):
            index = list(range(i*800, i*800+800))
            # index = np.array(index)
            sample_ = random.sample(index, 40)
            sample_.sort(reverse=True)
            print(sample_)
            for j in range(5):
                for k in range(j*8, j*8+8):
                    sample_[k] -= 8*i*j
                s_memory[j] += sample_[j*8:j*8+8]

        self.replay_buffer.task_buffers[task_idx].clear()
        self.enc_replay_buffer.task_buffers[task_idx].clear()
        self.task_idx = task_idx
        self.env.reset_task(task_idx)
        self.agent.clear_z()
        num_transitions = 0
        num_trajs = 0

        o = self.env.reset()
        next_o = None
        for epoch in range(iter):
            print("meta testing epoch:", epoch)
            path_length = 0
            observations = []
            actions = []
            rewards = []
            terminals = []
            agent_infos = []
            env_infos = []
            while path_length < num_iter_step:
                a, agent_info = self.agent.get_action(o)
                next_o, r, d, env_info = self.env.step(a)
                self.agent.update_context([o, a, r, next_o, d, env_info])
                self.agent.infer_posterior(self.agent.context)
                # self.policy.sample_z()

                observations.append(o)
                rewards.append(r)
                terminals.append(d)
                actions.append(a)
                agent_infos.append(agent_info)
                path_length += 1
                o = next_o
                env_infos.append(env_info)
                if d:  # done
                    break

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
            weights = np.ones(np.array(rewards).shape)

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

            path['context'] = self.sampler.policy.z.detach().cpu().numpy()
            self.replay_buffer.add_paths(self.task_idx, [path])
            self.enc_replay_buffer.add_paths(self.task_idx, [path])


            SU = np.array(self.replay_buffer_SU.get_all_data(source_idx))
            TU = np.array(self.replay_buffer_TU.get_all_data([task_idx]))
            SL = np.array(self.replay_buffer.get_all_data(source_idx))
            TL = np.array(self.replay_buffer.get_all_data([task_idx]))
            Q_index = s_memory[epoch]
            Q_index = np.array(Q_index)
            print('Q_index:', Q_index)

            # 根据 Q_index 向 SL 添加数据
            temp_ind = 0
            print('任务：', source_idx)
            for ind in source_idx:

                print("Q_index:", Q_index)
                total_cur = temp_ind + self.replay_buffer_SU.task_buffers[ind]._size
                print("total_cur:", total_cur)
                indices = Q_index[np.where(Q_index < (total_cur))]
                if len(indices) != 0:
                    indices = indices - temp_ind
                    add_data = self.replay_buffer_SU.task_buffers[ind].sample_data(indices)    # 取出SU元素
                    self.replay_buffer_SU.task_buffers[ind].delete_buffer(indices)     # 删除SU元素
                    print("查询后 replay buffer SU:", self.replay_buffer_SU.task_buffers[ind]._size)
                    self.replay_buffer.add_path(ind, add_data)
                    print("查询后 replay buffer SL:", self.replay_buffer.task_buffers[ind]._size)
                indices_ind = np.where(Q_index < total_cur)
                Q_index = np.delete(Q_index, indices_ind, axis=0)
                temp_ind = total_cur

            print("%%%%%%%%%%%%%%%%%%% meta testing %%%%%%%%%%%%%%%%%%")
            self._do_training_meta([task_idx], source_idx)

        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_o = None
        for step in range(self.sampler.max_path_length - iter * num_iter_step):
            a, agent_info = self.agent.get_action(o)
            next_o, r, d, env_info = self.env.step(a)


            self.agent.update_context([o, a, r, next_o, d, env_info])
            self.agent.infer_posterior(self.agent.context)

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            o = next_o
            env_infos.append(env_info)
            if d:  # done
                break

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
        )

        weights = np.ones(np.array(rewards).shape)

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

        path['context'] = self.sampler.policy.z.detach().cpu().numpy()

        self.replay_buffer.add_paths(self.task_idx, [path])

        print("%%%%%%%%%%%%%% 结束 meta testing，保存 meta test 模型参数 %%%%%%%%%%%%%%%%%")
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)


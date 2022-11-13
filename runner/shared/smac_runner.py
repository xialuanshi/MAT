import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.base_runner import Runner
from torch.distributions import Categorical


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run2(self):
        for episode in range(1):
            self.eval(episode)

    def sample_mat_happo_actions(self, m_action_logits, h_action_logits, deterministic=False):
        actions, m_action_log_probs, h_action_log_probs = [], [], []
        for m_distri, h_distri in zip(m_action_logits, h_action_logits):
            two_logist = self.mat_weight * m_distri.logits + self.happo_weight * h_distri.logits
            distri = Categorical(logits=two_logist)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            actions.append(_t2n(action))
            m_action_log_probs.append(_t2n(m_distri.log_prob(action)))
            h_action_log_probs.append(_t2n(h_distri.log_prob(action)))
        actions = np.array(np.split(np.array(actions), self.n_rollout_threads))
        print("action is iiiiiiiii ",actions)
        """
        two:
        action is iiiiiiiii  [[[3 2]
              [5 1]
              [2 4]]
            
             [[4 4]
              [2 3]
              [2 3]]]

        """

        m_action_log_probs = np.array(np.split(np.array(m_action_log_probs), self.n_rollout_threads))
        h_action_log_probs = np.array(np.split(np.array(h_action_log_probs), self.n_rollout_threads))
        return actions, m_action_log_probs, h_action_log_probs

    def sample_mat_mappo_actions(self, m_action_logits, map_action_logits, deterministic=False):
        actions, m_action_log_probs, map_action_log_probs = [], [], []
        for i in range(len(m_action_logits)):
            m_distri = m_action_logits[i]
            two_logist = m_distri.logits * self.mat_weight + self.mappo_weight * map_action_logits.logits[i]
            distri = Categorical(logits=two_logist)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            actions.append(_t2n(action))
            m_action_log_probs.append(_t2n(m_distri.log_prob(action)))
        map_action_log_probs = _t2n(map_action_logits.log_prob(actions))
        map_action_log_probs = np.array(map_action_log_probs).reshape(len(map_action_log_probs), 1)
        actions = np.array(np.split(np.array(actions), self.n_rollout_threads))
        m_action_log_probs = np.array(np.split(np.array(m_action_log_probs), self.n_rollout_threads))
        map_action_log_probs = np.array(np.split(np.array(map_action_log_probs), self.n_rollout_threads))
        return actions, m_action_log_probs, map_action_log_probs

    def sample_three_actions(self, m_action_logits, h_action_logits, map_action_logits, deterministic=False):
        # map_action_logits 中 probs 的shape应该是[self.n_rollout_threads, self.num_agents, act_space]
        actions, m_action_log_probs, h_action_log_probs, map_action_log_probs = [], [], [], []
        for i in range(len(m_action_logits)):
            m_distri = m_action_logits[i]
            h_distri = h_action_logits[i]
            three_probs = m_distri.probs * self.mat_weight + h_distri.probs * self.happo_weight + \
                           self.mappo_weight * map_action_logits.probs[i*4:(i+1)*4]
            distri = Categorical(probs=three_probs)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            actions.append(_t2n(action))
            m_action_log_probs.append(_t2n(m_distri.log_prob(action)))
            h_action_log_probs.append(_t2n(h_distri.log_prob(action)))
        print(torch.tensor(actions))
        cat_action = torch.tensor(actions).reshape(self.n_rollout_threads * self.num_agents, 1).to(self.device)
        map_action_log_probs = _t2n(map_action_logits.log_prob(cat_action))
        print("action is ", actions)
        print("m_action_log_probs is ", m_action_log_probs)
        print("map_action_log_probs is ", map_action_log_probs)
        '''
        mappo actions shape  
        tensor([[ 2],
        [ 3],
        [ 8],
        [ 2],
        [11],
        [ 8]])
        
        2 n_rollout_threads env
        actions>>>>>>>>>>>>>>>>>>>>>>>>>>> tensor([[ 4],
        [ 4],
        [ 4],
        [ 7],
        [12],
        [ 9],
        [ 0],
        [11],
        [ 1],
        [ 0],
        [ 5],
        [ 5]])

        '''
        map_action_log_probs = np.array(map_action_log_probs).reshape(self.num_agents, self.n_rollout_threads)
        actions = np.array(np.split(np.array(actions), self.n_rollout_threads, axis=1))
        m_action_log_probs = np.array(np.split(np.array(m_action_log_probs), self.n_rollout_threads, axis=1))
        h_action_log_probs = np.array(np.split(np.array(h_action_log_probs), self.n_rollout_threads, axis=1))
        map_action_log_probs = np.array(np.split(np.array(map_action_log_probs), self.n_rollout_threads, axis=1))
        return actions, m_action_log_probs, h_action_log_probs, map_action_log_probs

    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        agent_nums = 5
        max_steps = self.episode_length + 100
        for episode in range(episodes):
            self.warmup()
            per_thread_agent = {i: max_steps for i in range(agent_nums)}
            threads_done_step = {}
            step_done_agent ={i:per_thread_agent for i in range(self.n_rollout_threads)}
            thread_final_reward = { i:0 for i in range(self.n_rollout_threads)}
            if self.use_linear_lr_decay:
                self.mat_trainer.policy.lr_decay(episode, episodes)

                for agent_id in range(self.num_agents):
                    self.happo_trainer[agent_id].policy.lr_decay(episode, episodes)

                self.mappo_trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                m_values, m_actions, m_action_log_probs, m_rnn_states, m_rnn_states_critic, m_action_logits = self.collect(
                    step)

                if self.happo_weight != 0:
                    h_values, h_actions, h_action_log_probs, h_rnn_states, h_rnn_states_critic, \
                            h_action_logits = self.happo_collect(step)

                if self.mappo_weight != 0:
                    map_values, map_actions, map_action_log_probs, map_rnn_states, map_rnn_states_critic, \
                            map_action_logits = self.mappo_collect(step)

                if self.mappo_weight != 0 and self.happo_weight == 0:
                    actions, m_action_log_probs, h_action_log_probs = self.sample_mat_mappo_actions(m_action_logits,
                                                                                                    map_action_logits)
                elif self.mappo_weight == 0 and self.happo_weight != 0:
                    actions, m_action_log_probs, map_action_log_probs = self.sample_mat_happo_actions(m_action_logits,
                                                                                                      h_action_logits)
                elif self.mappo_weight != 0 and self.happo_weight != 0:
                    actions, m_action_log_probs, h_action_log_probs, map_action_log_probs = self.sample_three_actions(
                        m_action_logits, h_action_logits, map_action_logits)
                else:
                    actions = m_actions

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions, is_thread_dones = self.envs.step(actions)

                # 记录每个进程结束的步长
                if is_thread_dones:
                    for x in is_thread_dones:
                        if x not in threads_done_step:
                            threads_done_step[x] = step
                            # 正的最大值，付的最小值, 也可以环境返回
                            thread_final_reward[x] = max(rewards[x]) if max(rewards[x]) >= 20 else min(rewards[x])
                for done_thread in range(self.n_rollout_threads):
                    for done_agent in range(agent_nums):
                        if dones[done_thread][done_agent]:
                            if step_done_agent[done_thread][done_agent] >= self.episode_length:
                                step_done_agent[done_thread][done_agent] = step

                m_data = obs, share_obs, rewards, dones, infos, available_actions, \
                         m_values, actions, m_action_log_probs, m_rnn_states, m_rnn_states_critic
                self.mat_insert(m_data, threads_done_step, step_done_agent, thread_final_reward)

                if self.happo_weight != 0:
                    h_data = obs, share_obs, rewards, dones, infos, available_actions, \
                             h_values, actions, h_action_log_probs, h_rnn_states, h_rnn_states_critic
                    self.happo_insert(h_data, threads_done_step)

                if self.mappo_weight != 0:
                    map_date = obs, share_obs, rewards, dones, infos, available_actions, \
                               map_values, actions, map_action_log_probs, map_rnn_states, map_rnn_states_critic
                    self.mappo_insert(map_date, threads_done_step)

                # 所有环境都结束，跳出循环
                if len(is_thread_dones) == self.n_rollout_threads:
                    break


            # compute return and update network
            self.compute()
            mat_train_infos, happo_train_infos, mappo_train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.map_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(
                    incre_battles_game) > 0 else 0.0
                print("incre win rate is {}.".format(incre_win_rate))
                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won

                mat_train_infos['dead_ratio'] = 1 - self.mat_buffer.active_masks.sum() / reduce(lambda x, y: x * y,
                                                                                                list(
                                                                                                    self.mat_buffer.active_masks.shape))
                if self.happo_weight != 0:
                    for agent_id in range(self.num_agents):
                        happo_train_infos[agent_id]['dead_ratio'] = 1 - self.happo_buffer[agent_id].active_masks.sum() / (
                                self.num_agents * reduce(lambda x, y: x * y,
                                                         list(self.happo_buffer[agent_id].active_masks.shape)))
                if self.mappo_weight != 0:
                    mappo_train_infos['dead_ratio'] = 1 - self.mappo_buffer.active_masks.sum() / reduce(lambda x, y: x * y,
                                                                                                        list(
                                                                                                            self.mappo_buffer.active_masks.shape))

                self.log_train(mat_train_infos, happo_train_infos, mappo_train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.mat_buffer.reset(obs, share_obs, available_actions)  # 每一局进行重置

        # self.mat_buffer.share_obs[0] = share_obs.copy()
        # self.mat_buffer.obs[0] = obs.copy()
        # self.mat_buffer.available_actions[0] = available_actions.copy()

        # happo
        if self.happo_weight != 0:
            for agent_id in range(self.num_agents):
                self.happo_buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
                self.happo_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
                self.happo_buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

        # mappo
        if self.mappo_weight != 0:
            self.mappo_buffer.reset(obs, share_obs, available_actions)
            # self.mappo_buffer.share_obs[0] = share_obs.copy()
            # self.mappo_buffer.obs[0] = obs.copy()
            # self.mappo_buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.mat_trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic, action_logits \
            = self.mat_trainer.policy.get_actions(np.concatenate(self.mat_buffer.share_obs[step]),
                                                  np.concatenate(self.mat_buffer.obs[step]),
                                                  np.concatenate(self.mat_buffer.rnn_states[step]),
                                                  np.concatenate(self.mat_buffer.rnn_states_critic[step]),
                                                  np.concatenate(self.mat_buffer.masks[step]),
                                                  np.concatenate(self.mat_buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_logits

    @torch.no_grad()
    def happo_collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        action_logits = []
        for agent_id in range(self.num_agents):
            self.happo_trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, action_logit \
                = self.happo_trainer[agent_id].policy.get_actions(self.happo_buffer[agent_id].share_obs[step],
                                                                  self.happo_buffer[agent_id].obs[step],
                                                                  self.happo_buffer[agent_id].rnn_states[step],
                                                                  self.happo_buffer[agent_id].rnn_states_critic[step],
                                                                  self.happo_buffer[agent_id].masks[step],
                                                                  self.happo_buffer[agent_id].available_actions[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
            action_logits.append(action_logit)
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_logits

    @torch.no_grad()
    def mappo_collect(self, step):
        self.mappo_trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic, action_logits \
            = self.mappo_trainer.policy.get_actions(np.concatenate(self.mappo_buffer.share_obs[step]),
                                                    np.concatenate(self.mappo_buffer.obs[step]),
                                                    np.concatenate(self.mappo_buffer.rnn_states[step]),
                                                    np.concatenate(self.mappo_buffer.rnn_states_critic[step]),
                                                    np.concatenate(self.mappo_buffer.masks[step]),
                                                    np.concatenate(self.mappo_buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_logits

    def mat_insert(self, data, threads_done_step, step_done_agent, thread_final_reward):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)  # np.array.all()是与操作，所有元素为True，输出为True

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.mat_buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.mat_buffer.threads_done_step = threads_done_step
        self.step_done_agent = step_done_agent
        self.thread_final_reward = thread_final_reward

        self.mat_buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                               actions, action_log_probs, values, rewards, masks, bad_masks, active_masks,
                               available_actions)

    def happo_insert(self, data, threads_done_step):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.happo_buffer[0].rnn_states_critic.shape[2:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.happo_buffer[agent_id].threads_done_step = threads_done_step
            self.happo_buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                               rnn_states_critic[:, agent_id], actions[:, agent_id],
                                               action_log_probs[:, agent_id],
                                               values[:, agent_id], rewards[:, agent_id], masks[:, agent_id],
                                               bad_masks[:, agent_id],
                                               active_masks[:, agent_id], available_actions[:, agent_id])

    def mappo_insert(self, data, threads_done_step):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.mappo_buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            share_obs = obs
        self.mappo_buffer.threads_done_step = threads_done_step
        self.mappo_buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                                 actions, action_log_probs, values, rewards, masks, bad_masks, active_masks,
                                 available_actions)

    def log_train(self, mat_train_infos, happo_train_infos, mappo_train_infos, total_num_steps):
        mat_train_infos["average_step_rewards"] = np.mean(self.mat_buffer.rewards)
        for k, v in mat_train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                alg_k = "mat_" + k
                self.writter.add_scalars(alg_k, {alg_k: v}, total_num_steps)

        if self.happo_weight != 0:
            for agent_id in range(self.num_agents):
                happo_train_infos[agent_id]["average_step_rewards"] = np.mean(self.happo_buffer[agent_id].rewards)
                for k, v in happo_train_infos[agent_id].items():
                    agent_k = "agent%i/" % agent_id + k
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        if self.happo_weight != 0:
            mappo_train_infos["average_step_rewards"] = np.mean(self.mappo_buffer.rewards)
            for k, v in mappo_train_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    alg_k = "mappo_" + k
                    self.writter.add_scalars(alg_k, {alg_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        m_eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                     dtype=np.float32)
        h_eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                     dtype=np.float32)
        map_eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.mat_trainer.prep_rollout()
            m_actions, m_eval_rnn_states, m_action_logits = \
                self.mat_trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(m_eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)

            # eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            m_eval_rnn_states = np.array(np.split(_t2n(m_eval_rnn_states), self.n_eval_rollout_threads))

            if self.happo_weight != 0:
                h_action_logits = []
                for agent_id in range(self.num_agents):
                    self.happo_trainer[agent_id].prep_rollout()
                    _, temp_rnn_state, h_action_logit = \
                        self.happo_trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                                h_eval_rnn_states[:, agent_id],
                                                                eval_masks[:, agent_id],
                                                                eval_available_actions[:, agent_id],
                                                                deterministic=True)
                    h_eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    h_action_logits.append(h_action_logit)

            if self.mappo_weight != 0:
                self.mappo_trainer.prep_rollout()
                _, map_eval_rnn_states, map_action_logits = \
                    self.mappo_trainer.policy.act(np.concatenate(eval_obs),
                                                  np.concatenate(map_eval_rnn_states),
                                                  np.concatenate(eval_masks),
                                                  np.concatenate(eval_available_actions),
                                                  deterministic=True)
                map_eval_rnn_states = np.array(np.split(_t2n(map_eval_rnn_states), self.n_eval_rollout_threads))

            if self.mappo_weight != 0 and self.happo_weight == 0:
                eval_actions, _, _ = self.sample_mat_mappo_actions(m_action_logits, map_action_logits, deterministic=True)
            elif self.mappo_weight == 0 and self.happo_weight != 0:
                eval_actions, _, _ = self.sample_mat_happo_actions(m_action_logits, h_action_logits, deterministic=True)
            elif self.mappo_weight != 0 and self.happo_weight != 0:
                eval_actions, _, _, _ = self.sample_three_actions(m_action_logits, h_action_logits, map_action_logits)
            else:
                eval_actions = m_actions
            #
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            m_eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            h_eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                  self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                # self.eval_envs.save_replay()
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break

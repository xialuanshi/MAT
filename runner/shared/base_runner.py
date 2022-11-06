import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from mat.utils.shared_buffer import SharedReplayBuffer

# HAPPO
from mat.utils.separated_buffer import SeparatedReplayBuffer

# MAPPO
from mat.utils.shared_buffer_mappo import SharedReplayBuffer as MAPPO_SharedReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):
        self.mat_weight = 1
        self.happo_weight = 0
        self.mappo_weight = 0
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

            # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # HAPPO
        self.use_single_network = self.all_args.use_single_network

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":

        from mat.algorithms.mat.mat_trainer import MATTrainer as MAT_TrainAlgo
        from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as MAT_Policy

        if self.happo_weight != 0:
            from mat.algorithms.mat.happo_trainer import HAPPO as HATRPO_TrainAlgo
            from mat.algorithms.mat.happo_policy import HAPPO_Policy

        if self.mappo_weight != 0:
            from mat.algorithms.mat.mappo_trainer import MAPPO as MAPPO_TrainAlgo
            from mat.algorithms.mat.mappo.mappo_policy import MAPPO_Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else \
            self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        # policy network
        self.mat_policy = MAT_Policy(self.all_args,
                                     self.envs.observation_space[0],
                                     share_observation_space,
                                     self.envs.action_space[0],
                                     self.num_agents,
                                     device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        self.mat_trainer = MAT_TrainAlgo(self.all_args, self.mat_policy, self.num_agents, device=self.device)

        # buffer
        self.mat_buffer = SharedReplayBuffer(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space[0],
                                             share_observation_space,
                                             self.envs.action_space[0],
                                             self.all_args.env_name)

        # HAPPO
        if self.happo_weight != 0:
            self.happo_policy = []
            for agent_id in range(self.num_agents):
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                # policy network
                po = HAPPO_Policy(self.all_args,
                                  self.envs.observation_space[agent_id],
                                  share_observation_space,
                                  self.envs.action_space[agent_id],
                                  device=self.device)
                self.happo_policy.append(po)

            self.happo_trainer = []
            self.happo_buffer = []
            for agent_id in range(self.num_agents):
                # algorithm
                tr = HATRPO_TrainAlgo(self.all_args, self.happo_policy[agent_id], device=self.device)
                # buffer
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                bu = SeparatedReplayBuffer(self.all_args,
                                           self.envs.observation_space[agent_id],
                                           share_observation_space,
                                           self.envs.action_space[agent_id])
                self.happo_buffer.append(bu)
                self.happo_trainer.append(tr)

        # MAPPO
        if self.mappo_weight != 0:
            self.mappo_policy = MAPPO_Policy(self.all_args, self.envs.observation_space[0], share_observation_space,
                                             self.envs.action_space[0], device=self.device)
            self.mappo_trainer = MAPPO_TrainAlgo(self.all_args, self.mappo_policy, device=self.device)
            self.mappo_buffer = MAPPO_SharedReplayBuffer(self.all_args,
                                                         self.num_agents,
                                                         self.envs.observation_space[0],
                                                         share_observation_space,
                                                         self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.mat_trainer.prep_rollout()
        # 由于单个进程结束后，会继续插入最后时刻的obs,因此从该进程环境结束后，后面的数据是一样的
        next_values = self.mat_trainer.policy.get_values(np.concatenate(self.mat_buffer.share_obs[-1]),
                                                         np.concatenate(self.mat_buffer.obs[-1]),
                                                         np.concatenate(self.mat_buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.mat_buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.mat_buffer.compute_returns(next_values, self.mat_trainer.value_normalizer)

        # HAPPO
        if self.happo_weight != 0:
            for agent_id in range(self.num_agents):
                self.happo_trainer[agent_id].prep_rollout()
                next_value = self.happo_trainer[agent_id].policy.get_values(self.happo_buffer[agent_id].share_obs[-1],
                                                                            self.happo_buffer[agent_id].rnn_states_critic[
                                                                                -1],
                                                                            self.happo_buffer[agent_id].masks[-1])
                next_value = _t2n(next_value)
                self.happo_buffer[agent_id].compute_returns(next_value, self.happo_trainer[agent_id].value_normalizer)

        # MAPPO
        if self.mappo_weight != 0:
            self.mappo_trainer.prep_rollout()
            next_values = self.mappo_trainer.policy.get_values(np.concatenate(self.mappo_buffer.share_obs[-1]),
                                                               np.concatenate(self.mappo_buffer.rnn_states_critic[-1]),
                                                               np.concatenate(self.mappo_buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            self.mappo_buffer.compute_returns(next_values, self.mappo_trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.mat_trainer.prep_training()
        mat_train_infos = self.mat_trainer.train(self.mat_buffer)
        #self.mat_buffer.after_update()

        # HAPPO
        happo_train_infos = []
        # random update order
        if self.happo_weight != 0:
            action_dim = self.happo_buffer[0].actions.shape[-1]
            factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

            for agent_id in torch.randperm(self.num_agents):
                self.happo_trainer[agent_id].prep_training()
                self.happo_buffer[agent_id].update_factor(factor)
                available_actions = None if self.happo_buffer[agent_id].available_actions is None \
                    else self.happo_buffer[agent_id].available_actions[:-1].reshape(-1, *self.happo_buffer[
                                                                                             agent_id].available_actions.shape[
                                                                                         2:])

                old_actions_logprob, _ = self.happo_trainer[agent_id].policy.actor.evaluate_actions(
                    self.happo_buffer[agent_id].obs[:-1].reshape(-1, *self.happo_buffer[agent_id].obs.shape[2:]),
                    self.happo_buffer[agent_id].rnn_states[0:1].reshape(-1,
                                                                        *self.happo_buffer[agent_id].rnn_states.shape[2:]),
                    self.happo_buffer[agent_id].actions.reshape(-1, *self.happo_buffer[agent_id].actions.shape[2:]),
                    self.happo_buffer[agent_id].masks[:-1].reshape(-1, *self.happo_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.happo_buffer[agent_id].active_masks[:-1].reshape(-1,
                                                                          *self.happo_buffer[agent_id].active_masks.shape[
                                                                           2:]))
                train_info = self.happo_trainer[agent_id].train(self.happo_buffer[agent_id])

                new_actions_logprob, _ = self.happo_trainer[agent_id].policy.actor.evaluate_actions(
                    self.happo_buffer[agent_id].obs[:-1].reshape(-1, *self.happo_buffer[agent_id].obs.shape[2:]),
                    self.happo_buffer[agent_id].rnn_states[0:1].reshape(-1,
                                                                        *self.happo_buffer[agent_id].rnn_states.shape[2:]),
                    self.happo_buffer[agent_id].actions.reshape(-1, *self.happo_buffer[agent_id].actions.shape[2:]),
                    self.happo_buffer[agent_id].masks[:-1].reshape(-1, *self.happo_buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.happo_buffer[agent_id].active_masks[:-1].reshape(-1,
                                                                          *self.happo_buffer[agent_id].active_masks.shape[
                                                                           2:]))

                factor = factor * _t2n(
                    torch.prod(torch.exp(new_actions_logprob - old_actions_logprob), dim=-1).reshape(self.episode_length,
                                                                                                     self.n_rollout_threads,
                                                                                                     1))
                happo_train_infos.append(train_info)
                #self.happo_buffer[agent_id].after_update()

        # MAPPO
        mappo_train_infos = []
        if self.mappo_weight != 0:
            self.mappo_trainer.prep_training()
            mappo_train_infos = self.mappo_trainer.train(self.mappo_buffer)
            #self.mappo_buffer.after_update()

        return mat_train_infos, happo_train_infos, mappo_train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.mat_policy.save(self.save_dir, episode)

        if self.happo_weight != 0:
            for agent_id in range(self.num_agents):
                if self.use_single_network:
                    policy_model = self.happo_trainer[agent_id].policy.model
                    torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
                else:
                    policy_actor = self.happo_trainer[agent_id].policy.actor
                    torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                    policy_critic = self.happo_trainer[agent_id].policy.critic
                    torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
        # mappo
        if self.mappo_weight != 0:
            policy_actor = self.mappo_trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_mappo.pt")
            policy_critic = self.mappo_trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_mappo.pt")
            if self.mappo_trainer._use_valuenorm:
                policy_vnorm = self.mappo_trainer.value_normalizer
                torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_mappo.pt")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.mat_policy.restore(model_dir)

        # HAPPO
        if self.happo_weight != 0:
            for agent_id in range(self.num_agents):
                if self.use_single_network:
                    policy_model_state_dict = torch.load(str(model_dir) + '/model_agent' + str(agent_id) + '.pt')
                    self.happo_policy[agent_id].model.load_state_dict(policy_model_state_dict)
                else:
                    policy_actor_state_dict = torch.load(str(model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                    self.happo_policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                    policy_critic_state_dict = torch.load(str(model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                    self.happo_policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

        # mappo
        if self.mappo_weight != 0:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_mappo.pt')
            self.mappo_policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_mappo.pt')
                self.mappo_policy.critic.load_state_dict(policy_critic_state_dict)
                if self.mappo_trainer._use_valuenorm:
                    policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm_mappo.pt')
                    self.mappo_trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def log_train(self, mat_train_infos, happo_train_infos, mappo_train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in mat_train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                alg_k = "mat_" + k
                self.writter.add_scalars(alg_k, {alg_k: v}, total_num_steps)

        if self.happo_weight != 0:
            for agent_id in range(self.num_agents):
                for k, v in happo_train_infos[agent_id].items():
                    agent_k = "agent%i/" % agent_id + k
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        if self.mappo_weight != 0:
            for k, v in mappo_train_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    alg_k = "mappo_" + k
                    self.writter.add_scalars(alg_k, {alg_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

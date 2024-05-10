import numpy as np
import torch
import os
from networks import EFFDQN_Diabetes_Real_Net, DQN_Diabetes_Net, effd3qn_split, EFFDQN_meal_Net, effd3qn2
from torch.optim.lr_scheduler import ExponentialLR

class DQN(object):
    def __init__(self, args, env):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.n_actions = env.num_actions

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.target_update_freq = args.TARGET_UPDATE

        self.policy_net = DQN_Diabetes_Net(self.n_actions, state_size, args.state_embedding_size, args.n_hidden).to(
            args.device)
        self.policy_net.train()

        self.target_net = DQN_Diabetes_Net(self.n_actions, state_size, args.state_embedding_size, args.n_hidden).to(
            args.device)
        self.update_target_net()
        self.target_net.train()  # eval()?
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.device = args.device
        self.counter = 0

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, eps):
        if np.random.uniform() > eps:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return np.random.randint(self.n_actions)

    def learn(self, mem):
        states, actions, rewards, next_states, not_done = mem.sample(self.batch_size)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)  # 从动作值分布中提取出对应于实际选择的动作的动作值

        # Compute max_a(Q(s_{t+1}, a)) for all next states.
        next_q_values = self.target_net(next_states).max(dim=1)[0].detach().reshape(-1, 1)  # self.target_net => DQN

        # Compute the expected Q values
        expected_q_values = rewards + (not_done * self.gamma * next_q_values)

        # Compute loss
        loss = self.criterion(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪。这样可以避免梯度爆炸问题，提高模型的训练稳定性。
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating target net once every episode
        if self.counter % self.target_update_freq == 0:
            self.update_target_net()

        self.counter += 1

    def save(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy.pth'))

    def train(self):
        self.policy_net.train()

    def eval(self):
        self.policy_net.eval()

class EFFDQN(object):
    def __init__(self, args, env):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.n_actions = env.num_actions

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.target_update_freq = args.TARGET_UPDATE

        self.policy_net = EFFDQN_Diabetes_Real_Net(self.n_actions, action_size, state_size,
                                                   args.action_embedding_size, args.state_embedding_size, args.n_hidden
                                                   ).to(args.device)
        self.policy_net.train()

        self.target_net = EFFDQN_Diabetes_Real_Net(self.n_actions, action_size, state_size,
                                                   args.action_embedding_size, args.state_embedding_size, args.n_hidden
                                                   ).to(args.device)
        self.update_target_net()
        self.target_net.train()  # eval()?
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        if args.scheduler:
            # 添加学习率调度器
            self.scheduler = ExponentialLR(self.optimizer, gamma=args.lr_decay)
        self.criterion = torch.nn.MSELoss()
        self.device = args.device
        self.counter = 0
        self.PER = False
        if args.PER:
            self.PER = True

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, eps):
        if np.random.uniform() > eps:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return np.random.randint(self.n_actions)

    def learn(self, mem):
        if self.PER:
            tree_idxs, states, actions, rewards, next_states, not_done, weights = mem.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, not_done = mem.sample(self.batch_size)
        states = states.squeeze()
        next_states = next_states.squeeze()
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)  # 从动作值分布中提取出对应于实际选择的动作的动作值

        # Compute max_a(Q(s_{t+1}, a)) for all next states.
        next_q_values = self.target_net(next_states).max(dim=1)[0].detach().reshape(-1, 1)  # self.target_net => DQN

        # Compute the expected Q values
        expected_q_values = rewards + (not_done * self.gamma * next_q_values)

        # Compute loss
        if self.PER:
            TD_errors = (q_values - expected_q_values).detach().cpu().squeeze().numpy()  # PER
            mem.update_priorities(tree_idxs, TD_errors)  # PER
            loss = torch.mean(weights * (expected_q_values - q_values) ** 2)
        else:
            loss = self.criterion(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪。这样可以避免梯度爆炸问题，提高模型的训练稳定性。
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating target net once every episode
        if self.counter % self.target_update_freq == 0:
            self.update_target_net()

        self.counter += 1

    def save(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy.pth'))

    def train(self):
        self.policy_net.train()

    def eval(self):
        self.policy_net.eval()

class Multistep(object):

    def __init__(self, args, env):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.n_actions = env.num_actions

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.target_update_freq = args.TARGET_UPDATE

        if args.split:
            self.policy_net = effd3qn_split(self.n_actions, action_size, state_size,
                                            args.action_embedding_size, args.state_embedding_size, args.n_hidden,
                                            args.num_hidden, args.H, args.drop_prob).to(args.device)
            self.target_net = effd3qn_split(self.n_actions, action_size, state_size,
                                            args.action_embedding_size, args.state_embedding_size, args.n_hidden,
                                            args.num_hidden, args.H, args.drop_prob).to(args.device)
        else:
            self.policy_net = EFFDQN_Diabetes_Real_Net(self.n_actions, action_size, state_size,
                                                       args.action_embedding_size, args.state_embedding_size, args.n_hidden
                                                       ).to(args.device)

            self.target_net = EFFDQN_Diabetes_Real_Net(self.n_actions, action_size, state_size,
                                                       args.action_embedding_size, args.state_embedding_size, args.n_hidden
                                                       ).to(args.device)

            # self.policy_net = effd3qn2(self.n_actions, action_size, state_size,
            #                            args.action_embedding_size, args.state_embedding_size,
            #                            args.n_hidden, args.num_hidden, args.H, args.drop_prob).to(args.device)
            #
            # self.target_net = effd3qn2(self.n_actions, action_size, state_size,
            #                            args.action_embedding_size, args.state_embedding_size,
            #                            args.n_hidden, args.num_hidden, args.H, args.drop_prob).to(args.device)
        self.policy_net.train()
        self.update_target_net()
        self.target_net.train()  # eval()?
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        if args.scheduler:
            # 添加学习率调度器
            self.scheduler = ExponentialLR(self.optimizer, gamma=args.lr_decay)
        self.criterion = torch.nn.MSELoss()
        self.device = args.device
        self.counter = 0
        self.PER = False
        if args.PER:
            self.PER = True

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, eps):
        if np.random.uniform() > eps:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return np.random.randint(self.n_actions)

    def learn(self, mem):
        # Sample transitions
        if self.PER:
            tree_idxs, states, actions, rewards, next_states, not_done, weights = mem.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, not_done = mem.sample(self.batch_size)
        states = states.squeeze()
        next_states = next_states.squeeze()
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)

        # print(states)
        # print(next_states)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)  # 从动作值分布中提取出对应于实际选择的动作的动作值

        # --DQN--
        # Compute V(s_{t+1}) for all next states.
        next_q_values = self.target_net(next_states).max(dim=1)[0].detach().reshape(-1, 1)  # self.target_net => DQN

        # --DDQN--
        # Compute action primes - plug in next states to main model
        # Get target_outputs - plug in next states to target model
        # prime_actions = self.policy_net(next_states).argmax(dim=1).long().reshape(-1, 1).to(self.device)
        # next_q_values = self.target_net(next_states).gather(1, prime_actions)
        # print('next_state_values.shape', next_state_values.shape)

        # Compute the expected Q values
        # expected_q_values = rewards + (not_done * self.gamma * next_state_values)
        expected_q_values = rewards + not_done * (self.gamma ** mem.n) * next_q_values

        # Compute loss
        if self.PER:
            TD_errors = (q_values - expected_q_values).detach().cpu().squeeze().numpy()  # PER
            mem.update_priorities(tree_idxs, TD_errors)  # PER
            loss = torch.mean(weights * (expected_q_values - q_values) ** 2)
        else:
            loss = torch.mean((expected_q_values - q_values) ** 2)
        # loss = self.criterion(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪。这样可以避免梯度爆炸问题，提高模型的训练稳定性。
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # Updating target net once every episode
        if self.counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.counter += 1

    def save(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy.pth'))

    def train(self):
        self.policy_net.train()

    def eval(self):
        self.policy_net.eval()

class EFFDQN_meal(object):
    def __init__(self, args, env):
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.n_actions = env.num_actions

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.target_update_freq = args.TARGET_UPDATE

        self.policy_net = EFFDQN_meal_Net(self.n_actions, action_size, state_size,
                                                   args.action_embedding_size, args.state_embedding_size, args.n_hidden
                                                   ).to(args.device)
        self.policy_net.train()

        self.target_net = EFFDQN_meal_Net(self.n_actions, action_size, state_size,
                                                   args.action_embedding_size, args.state_embedding_size, args.n_hidden
                                                   ).to(args.device)
        self.update_target_net()
        self.target_net.train()  # eval()?
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        if args.scheduler:
            # 添加学习率调度器
            self.scheduler = ExponentialLR(self.optimizer, gamma=args.lr_decay)
        self.criterion = torch.nn.MSELoss()
        self.device = args.device
        self.counter = 0

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, eps):
        if np.random.uniform() > eps:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return np.random.randint(self.n_actions)

    def learn(self, mem):
        states, actions, rewards, next_states, not_done = mem.sample(self.batch_size)

        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)  # 从动作值分布中提取出对应于实际选择的动作的动作值

        # Compute max_a(Q(s_{t+1}, a)) for all next states.
        next_q_values = self.target_net(next_states).max(dim=1)[0].detach().reshape(-1, 1)  # self.target_net => DQN

        # Compute the expected Q values
        expected_q_values = rewards + (not_done * self.gamma * next_q_values)

        # Compute loss
        loss = self.criterion(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪。这样可以避免梯度爆炸问题，提高模型的训练稳定性。
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Updating target net once every episode
        if self.counter % self.target_update_freq == 0:
            self.update_target_net()

        self.counter += 1

    def save(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy.pth'))

    def train(self):
        self.policy_net.train()

    def eval(self):
        self.policy_net.eval()
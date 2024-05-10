import torch
import torch.nn as nn
import torch.nn.functional as F
class DQN_Diabetes_Net(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__ ( self, n_actions, state_size, state_embedding_size, n_hidden):
		super (DQN_Diabetes_Net, self).__init__ ()
		self.state_embedder = nn.Linear (state_size, state_embedding_size)
		self.obs_layer = nn.Linear (state_embedding_size, n_hidden)
		self.obs_layerout = nn.Linear (n_hidden, n_actions)

	def forward ( self, observation ):
		state, _ = torch.split(observation, 1, dim=1)
		state = self.state_embedder (state)
		state = F.relu (self.obs_layer (state))
		return self.obs_layerout (state)

class EFFDQN_Diabetes_Real_Net(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__(self, n_actions, action_size, state_size, action_embedding_size, state_embedding_size, n_hidden):
		super(EFFDQN_Diabetes_Real_Net, self).__init__()
		self.action_embedder = nn.Linear(action_size, action_embedding_size)
		self.state_embedder = nn.Linear(state_size, state_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear(state_embedding_size + action_embedding_size, n_hidden)
		self.obs_layerout = nn.Linear(n_hidden, n_actions)

	def forward(self, observation):
		state, eff_action = torch.split(observation, 1, dim=1)
		state_embedding = self.state_embedder(state)
		action_embedding = self.action_embedder(eff_action)
		observation = torch.cat((state_embedding, action_embedding), dim=1)
		h = F.relu(self.obs_layer(observation))
		return self.obs_layerout(h)

class EFFDQN_meal_Net(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__(self, n_actions, action_size, state_size, action_embedding_size, state_embedding_size, n_hidden, meal_size=1, meal_embedding_size=16):
		super(EFFDQN_meal_Net, self).__init__()
		self.action_embedder = nn.Linear(action_size, action_embedding_size)
		self.state_embedder = nn.Linear(state_size, state_embedding_size)
		self.meal_embedder = nn.Linear(meal_size, meal_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear(state_embedding_size + action_embedding_size + meal_embedding_size, n_hidden)
		self.obs_layerout = nn.Linear(n_hidden, n_actions)

	def forward(self, observation):
		state, eff_action, meal = torch.split(observation, 1, dim=1)
		state_embedding = self.state_embedder(state)
		action_embedding = self.action_embedder(eff_action)
		meal_embedding = self.meal_embedder(meal)
		observation = torch.cat((state_embedding, action_embedding, meal_embedding), dim=1)
		h = F.relu(self.obs_layer(observation))
		return self.obs_layerout(h)

class hidden_linear_layer(torch.nn.Module):
	#                  256,  128,   0
	def __init__(self, D_in, D_out, drop_prob):
		super(hidden_linear_layer, self).__init__()

		self.linear = nn.Linear(D_in, D_out).float()
		self.dropout = nn.Dropout(p=drop_prob)
		self.activation = nn.ELU()
		# self.batch_norm = nn.BatchNorm1d(num_features=D_out  # Expected more than 1 value per channel when training, got input size torch.Size([1, 128])

	def forward(self, x):
		# result = self.dropout(self.activation(self.batch_norm(self.linear(x))))
		result = self.dropout(self.activation(self.linear(x)))
		return result

class effd3qn_split(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__(self, n_actions, action_size, state_size, action_embedding_size, state_embedding_size, n_hidden, num_hidden, H, drop_prob):
		#                  6,         1,            1,              16,                 16,                256,        2,    128,    0
		super(effd3qn_split, self).__init__()
		self.action_embedder = nn.Linear(action_size, action_embedding_size)
		self.prolongedness_embedder = nn.Linear(action_size, action_embedding_size)
		self.state_embedder = nn.Linear(state_size, state_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear(state_embedding_size + 2*action_embedding_size, n_hidden)  # new state

		# self.obs_layerout = nn.Linear(n_hidden, n_actions)

		self.input_layer = hidden_linear_layer(n_hidden, H, drop_prob)
		self.value_layer = nn.Linear(H, 1)
		self.advantage_layer = nn.Linear(H, n_actions)
		self.layers = nn.ModuleList([self.input_layer])

		if num_hidden > 1:
			self.layers.extend([hidden_linear_layer(H, H, drop_prob) for i in range(num_hidden - 1)])

	def forward(self, observation):
		# print(observation)
		# print(observation.shape)
		state, last_action, last_prolongedness = torch.split(observation, 1, dim=1)
		state_embedding = self.state_embedder(state)
		action_embedding = self.action_embedder(last_action)
		prolongedness_embedding = self.prolongedness_embedder(last_prolongedness)
		observation = torch.cat((state_embedding, action_embedding, prolongedness_embedding), dim=1)
		h = F.relu(self.obs_layer(observation))  # new state(x)/ ELU/ BN(×)
		# print(observation.shape)  # (1, 256)

		pre_output = nn.Sequential(*self.layers).forward(h)
		# pre_output = self.layers[0].forward(observation)
		# print(pre_output.shape)
		# pre_output = self.layers[1].forward(observation)
		value = self.value_layer(pre_output)
		advantage = self.advantage_layer(pre_output)
		advantage_diff = advantage - advantage.mean(1, keepdim=True)
		y_pred = value + advantage_diff

		return y_pred

class effd3qn2(nn.Module):
	# Very simple fully connected network with relu activations
	def __init__(self, n_actions, action_size, state_size, action_embedding_size, state_embedding_size, n_hidden, num_hidden, H, drop_prob):
        #                  6,         1,            1,              16,                 16,                256,        2,    128,    0
		super(effd3qn2, self).__init__()
		self.action_embedder = nn.Linear(action_size, action_embedding_size)
		self.state_embedder = nn.Linear(state_size, state_embedding_size)
		self.num_actions = n_actions
		self.obs_layer = nn.Linear(state_embedding_size + action_embedding_size, n_hidden)  # new state

		# self.obs_layerout = nn.Linear(n_hidden, n_actions)

		self.input_layer = hidden_linear_layer(n_hidden, H, drop_prob)
		self.value_layer = nn.Linear(H, 1)
		self.advantage_layer = nn.Linear(H, n_actions)
		self.layers = nn.ModuleList([self.input_layer])

		if num_hidden > 1:
			self.layers.extend([hidden_linear_layer(H, H, drop_prob) for i in range(num_hidden - 1)])

	def forward(self, observation):
		state, eff_action = torch.split(observation, 1, dim=1)
		state_embedding = self.state_embedder(state)
		action_embedding = self.action_embedder(eff_action)
		observation = torch.cat((state_embedding, action_embedding), dim=1)
		h = F.relu(self.obs_layer(observation))  # new state(x)/ ELU/ BN?
		# print(observation.shape)  # (1, 256)

		pre_output = nn.Sequential(*self.layers).forward(h)
		# pre_output = self.layers[0].forward(observation)
		# print(pre_output.shape)
		# pre_output = self.layers[1].forward(observation)
		value = self.value_layer(pre_output)
		advantage = self.advantage_layer(pre_output)
		advantage_diff = advantage - advantage.mean(1, keepdim=True)
		y_pred = value + advantage_diff

		return y_pred
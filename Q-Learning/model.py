import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class QNetwork(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size):
        """
            Creates the Nueral Network that will be used in the 
            Q-learning algorithm.

            Parameters:
                obs_size    (int) : the input size of the network
                act_size    (int) : the output size of the network
                hidden_size (int) : the number of nodes in the hidden layer 
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_size)
        
    def forward(self, x):
        """
            Implementation of the feedforward algorithm
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def act(self, state):
        """
            selects the action using epsilon-greedy policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.max(1)[1].item()
        return action
        
class QLearningAgent():
    def __init__(self, obs_size, act_size, hidden_size, gamma, epsilon, lr):
        """
            Creates the class that trains the network on the enviroment using Q-learning
        """
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        
        self.q_net = QNetwork(obs_size, act_size, hidden_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
    def select_action(self, state):
        """
            selects the action using epsilon-greedy policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        if random.random() > self.epsilon:
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.act_size)
        return action
        
    def update_q_net(self, state, action, reward, next_state, done):
        """
            updates the network using the Q learning update equation
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action = torch.tensor([[action]])
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)
        
        q_value = self.q_net(state).gather(1, action)
        next_q_value = self.q_net(next_state).max(1)[0].unsqueeze(1)
        expected_q_value = reward + (1 - done) * self.gamma * next_q_value
        
        loss = F.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self, env, num_episodes):
        """
            training loop of algorithm
        """
        for i_episode in range(1, num_episodes+1):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update_q_net(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
            if i_episode % 10 == 0:
                print("Episode {}/{}: Total Reward = {}".format(i_episode, num_episodes, total_reward))

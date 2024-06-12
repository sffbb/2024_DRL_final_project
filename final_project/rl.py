import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
from env import RobotArmEnv
from gym import spaces
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
try :
    from parameter import goal , obstacles_in_main
    print("from parameter !!!")#,goal , obstacles_in_main)
    env = RobotArmEnv(goal,obstacles_in_main[0])
    #h, l = env.calculate_action_space()
except:
    print("Not from parameter >_<")

    env = RobotArmEnv()




noise_std = 0.0001
class ActorCritic(nn.Module):
    #def __init__(self, state_dim, action_dim, hidden_size=64,epsilon=0.1, noise_std=noise_std)
    def __init__(self, state_dim, action_dim, hidden_size=64,epsilon=0.2, noise_std=noise_std):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        ).to(device) # 移動到指定的設備

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device) # 移動到指定的設備
        self.epsilon = epsilon
        self.noise_std = noise_std
    def forward(self):
        raise NotImplementedError

    def act(self, state, action_low, action_high):
        with torch.no_grad():
            state = state.to(device)
            action = self.actor(state)

            # Ensure action is between -1 and 1 before scaling
            action = torch.clamp(action, -1, 1)

            # Convert action_low and action_high to torch tensors
            action_low = torch.tensor(action_low, device=device)
            action_high = torch.tensor(action_high, device=device)

            # Scale action to be within action_low and action_high
            scaled_action = action_low + (action + 1) * 0.5 * (action_high - action_low)

            # Ensure the scaled action is within the specified range
            scaled_action = torch.clamp(scaled_action, action_low, action_high)
            # ε-greedy exploration
            if random.random() < self.epsilon:
                # Randomly choose an action
                action = torch.tensor(np.random.uniform(action_low.cpu().numpy(), action_high.cpu().numpy()), dtype=torch.float32, device=device)
            else:
                # Use the actor network to select an action
                action = scaled_action

            # Add Gaussian noise to the selected action
            noise = torch.tensor(np.random.normal(0, self.noise_std, size=action.shape), dtype=torch.float32,
                                 device=device)
            action = action + noise
            action = torch.clamp(action, action_low, action_high)
        return action.cpu().numpy()

    def evaluate(self, state, action):
        state = state.to(device)  # 將 state 移動到指定的設備
        action = action.to(device)  # 將 action 移動到指定的設備
        action_mean = self.actor(state)
        critic_value = self.critic(state)
        action_std = torch.ones_like(action_mean, device=device) * 0.5  # 將 action_std 移動到指定的設備
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, critic_value

    def get_value(self, state):
        with torch.no_grad():
            state = state.to(device)# 移動到指定的設備
            value = self.critic(state)
        return value.item()

class PPO:
    def __init__(self, state_dim, action_dim,has,las, lr=3e-4, gamma=0.99, K_epoch=10, epsilon_clip=1e-5):
        self.policy = ActorCritic(state_dim, action_dim)
        self.policy.to(device) # 移動到指定的設備
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.K_epoch = K_epoch
        self.epsilon_clip = epsilon_clip
        self.noise_std = noise_std
        self.has = has  # New parameter to store high action space
        self.las = las  # New parameter to store low action space

        # Set the action space using has and las
        self.action_space = spaces.Box(low=np.array(las, dtype='float32'),
                                       high=np.array(has, dtype='float32'),
                                       shape=(5,), dtype=np.float32)
    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(self.K_epoch):
            states = states.clone().detach().requires_grad_(True).to(device)  # 移動到指定的設備
            actions = actions.clone().detach().requires_grad_(True).to(device)  # 移動到指定的設備
            log_probs = log_probs.clone().detach().requires_grad_(True).to(device)  # 移動到指定的設備
            returns = returns.clone().detach().requires_grad_(True).to(device)  # 移動到指定的設備
            advantages = advantages.clone().detach().requires_grad_(True).to(device)  # 移動到指定的設備

            new_log_probs, entropies, values = self.policy.evaluate(states, actions)

            ratio = torch.exp(new_log_probs - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) # clip
            advantages = torch.unsqueeze(advantages, dim=1)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = entropies.mean()

            loss = policy_loss + value_loss - 0.01 * entropy_loss
            self.optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            self.optimizer.step()

    def get_action(self, state):
        action_low = self.las
        action_high = self.has
        state = torch.tensor(state, dtype=torch.float32).to(device)  # 移動到指定的設備
        action = self.policy.act(state, action_low, action_high)
        return action


class EpsilonGreedyActorCritic(ActorCritic):
    def __init__(self, state_dim, action_dim, hidden_size=64, epsilon=0.1):
        super(EpsilonGreedyActorCritic, self).__init__(state_dim, action_dim, hidden_size)
        self.epsilon = epsilon

    def act(self, state, action_low, action_high):
        if random.random() < self.epsilon:
            # Randomly choose an action
            action = np.random.uniform(action_low, action_high)
        else:
            # Use the actor network to select an action
            action = super().act(state, action_low, action_high)
        return action

class NoisyActorCritic(ActorCritic):
    def __init__(self, state_dim, action_dim, hidden_size=64, noise_std=0.1):
        super(NoisyActorCritic, self).__init__(state_dim, action_dim, hidden_size)
        self.noise_std = noise_std

    def act(self, state, action_low, action_high):
        action = super().act(state, action_low, action_high)
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        action = action + noise
        return np.clip(action, action_low, action_high)
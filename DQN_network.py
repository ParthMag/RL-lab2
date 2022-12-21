import numpy as np
import gym
import torch
import torch.nn as nn
from collections import deque

class ExperienceReplayBuffer(object):
    def __init__(self, max_len=1000):
        self.buffer = deque(maxlen=max_len)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def __len__(self):
        return len(self.buffer)
    
    def sample_batch(self, n):
        if n > len(self.buffer):
            print("Error! Asked to retrive too many elements from the buffer")
        
        indices = np.random.choice(len(self.buffer), n, replace=False)
        batch = [self.buffer[i] for i in indices]

        return zip(*batch)
    

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            )
    def forward(self, x):
        return self.network(x)
    
    
env = gym.make('LunarLander-v2')
import torch
from numpy import random

class RandomChannelFlip(object):
    def __init__(self, keys, prob=0.5, channel_axis=0):
        self.channel_axis = channel_axis
        self.keys = keys
        self.prob = prob
        
    def __call__(self, data):
        apply = random.random() <= self.prob

        if apply:
            for key in self.keys:
                if key in data:
                    data[key] = torch.flip(data[key], dims=[0])
            
        return data
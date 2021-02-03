#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, embed_size):
        '''
        @param embed_size: Embedding size of word
        '''
        super(Highway, self).__init__()
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        

    def forward(self,conv_out):
        projection = F.relu(self.projection(conv_out))
        gate = torch.sigmoid(self.gate(conv_out))
        highway = torch.mul(projection, gate) + torch.mul(conv_out, 1-gate)
        return highway
### END YOUR CODE 


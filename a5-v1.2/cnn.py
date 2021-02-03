#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, embed_size = 50, m_word = 21, k = 5, f = None):

        '''
        @param embedded_size: embedding size of char
        @param k : kernel size
        @f : number of filters, should be the embedding size of word
        '''

        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_size,
                                    out_channels=f,
                                    kernel_size=k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, reshaped):
        '''
        @param reshaped (Tensor): Tensor of char-level embedding with shape (max_sentence_length, 
                                    batch_size, e_char, max_word_length), where e_char = embed_size of char, 
        '''
        conv = self.conv1d(reshaped)
        conv_out = self.maxpool(F.relu(conv))

        return torch.squeeze(conv_out, -1)


### END YOUR CODE


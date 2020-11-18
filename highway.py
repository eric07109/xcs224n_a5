#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

class Highway(nn.Module):
    def __init__(self, word_embedding_size, dropout_rate = 0.2):
        self.word_embedding_size = word_embedding_size
        self.projection_weights = nn.Linear(self.word_embedding_size, self.word_embedding_size, bias=True)
        self.gate_weights = nn.Linear(self.word_embedding_size, self.word_embedding_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, convolution_outputs):
        # batch operation
        x_proj = nn.ReLU(self.projection_weights(convolution_outputs))
        x_gate = nn.Sigmoid(self.gate_weights(convolution_outputs))
        x_highway = (x_gate * x_proj) + ((1 - x_gate) * x_proj)
        highway_outputs.append(x_highway)


### END YOUR CODE 


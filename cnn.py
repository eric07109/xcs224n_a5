#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

class CNN(nn.Module):
    def __init__(self, feature_number, max_word_length, kernel_size = 5,):
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels = max_word_length, out_channels = feature_number, kernel_size = kernel_size, bias = True)

    def forward(self, char_embeddings):
        x_conv = self.conv1d(char_embeddings)
        x_conv_out = nn.MaxPool1d(nn.ReLU(x_conv))
        return x_conv_out
### END YOUR CODE


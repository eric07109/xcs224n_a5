#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
sanity_check.py: sanity checks for assignment 5
Usage:
    sanity_check.py 1a
    sanity_check.py 1b
    sanity_check.py 1f
    sanity_check.py 2a
    sanity_check.py 2b
    sanity_check.py 2c
    sanity_check.py 2d
"""
import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.utils
from docopt import docopt

from char_decoder import CharDecoder
from nmt_model import NMT
from utils import pad_sents_char
from vocab import Vocab, VocabEntry

# ----------
# CONSTANTS
# ----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 3
DROPOUT_RATE = 0.0


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1c_sanity_check():
    """ Sanity check for to_input_tensor_char() function.
    """
    print("-" * 80)
    print("Running Sanity Check for Question 1c")
    print("-" * 80)
    vocab = VocabEntry()

    print("Running test on a list of sentences")
    sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'],
                 ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
    sent_padded = vocab.to_input_tensor_char(sentences, torch.device)

question_1c_sanity_check()
# **Tutorial Highlights**
#
# -  Handle loading and preprocessing of `Cornell Movie-Dialogs
#    Corpus <https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html>`__
#    dataset
# -  Implement a sequence-to-sequence model with `Luong attention
#    mechanism(s) <https://arxiv.org/abs/1508.04025>`__
# -  Jointly train encoder and decoder models using mini-batches
# -  Implement greedy-search decoding module
# -  Interact with trained chatbot
# Dataset: 
# https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip


import torch
import csv 
import random
import re
import os 
import unicodedata
import codecs
import itertools
import math
import json
from torch.jit import script, trace
from torch import nn
from torch import optim
from torch.nn import functional as F 
from io import open


device = ("cuda"
	if torch.cuda.is_available()
	else "cpu"
)
corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)

def printlines(file, n=10):
	with open(file, "rb") as datafile:
		lines = datafile.readlines()

	for line in lines[:n]:
		print(line)

printlines(os.path.join(corpus, "utterances.jsonl"))
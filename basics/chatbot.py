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


def load_convo(file_name):
	lines = {}
	conversations = {}

	with open(file_name, "r", encoding="iso-8859-1") as f:
		for line in file:
			line_json = json.loads(line)
			line_obj = {}
			line_obj["lineID"] = line_json["id"]
			line_obj["characterID"] = line_json["speaker"]
			line_obj["text"] = line_json["text"]
			lines[line_obj["lineID"]] = line_obj

			if line_json["conversation_id"] not in conversations:
				conv_obj = {}
				conv_obj["conversation_id"] = line_json["conversation_id"]
				conv_obj["movieID"] = line_json["meta"]["movie_id"]
				conv_obj["lines"] = [line_obj]

			else:
				conv_obj = conversations[line_json["conversation_id"]]
				conv_obj["lines"].insert(0, line_obj)

			conversations[conv_obj["conversationID"]] = conv_obj

	return lines, conversations
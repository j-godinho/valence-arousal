import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

import argparse

import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import gc
import math

import nltk
from nltk.tokenize import word_tokenize


try: 
	import matplotlib.pyplot as plt
except:
	print("Failed to import matplotlib")
	

def load_sentences_data(args):
	print("[LOADING SENTENCES DATA]")

	#load facebook dataset	
	if("facebook" in args.sentences):	
		dataset = pd.read_csv(args.sentences)
		sentences = np.array(dataset["Anonymized Message"])
		arousals = np.array(dataset["Arousal_mean"]).reshape(-1, 1)
		valences = np.array(dataset["Valence_mean"]).reshape(-1, 1)
	else:
		dataset = pd.read_csv(args.sentences, sep='\t')
		sentences = np.array(dataset["sentence"])
		arousals = np.array(dataset["Arousal"]).reshape(-1, 1)
		valences = np.array(dataset["Valence"]).reshape(-1, 1)

	# Normalization
	scalerV = MinMaxScaler(feature_range=(np.min(valences), np.max(valences)))
	scalerA = MinMaxScaler(feature_range=(np.min(arousals), np.max(arousals)))
	
	return sentences, valences, arousals, scalerV, scalerA

def load_words_data(args, scalerV, scalerA):
	dataset = pd.read_csv(args.words, sep='\t')
	words = np.asarray(dataset["Description"])
	valences = np.asarray(dataset["Valence_value"]).reshape(-1, 1)
	arousals = np.asarray(dataset["Arousal_value"]).reshape(-1, 1)

	valences = scalerV.fit_transform(valences)
	arousals = scalerA.fit_transform(arousals)

	words = {words[i]: [valences[i],arousals[i]] for i in range(len(words))}

	return words

def train_avaliate_model(sentences, s_valences, s_arousals, words):

	mean_val = np.mean(s_valences)
	mean_arousal = np.mean(s_arousals)

	v_pred = []
	a_pred = []
	for sentence in sentences:
		score_v = []
		score_a = []
		for word in nltk.word_tokenize(sentence):
			if(word in words.keys()):
				score_v.append(words[word][0][0])
				score_a.append(words[word][1][0])

		if(len(score_v) == 0):
			v_pred.append(mean_val)
			a_pred.append(mean_arousal)
		else:
			v_pred.append(sum(score_v) / len(score_v))
			a_pred.append(sum(score_a) / len(score_a))

	v_pred = np.asarray(v_pred).reshape(-1, 1)
	a_pred = np.asarray(a_pred).reshape(-1, 1)

	calculate_metrics(s_valences, s_arousals, v_pred, a_pred)


def calculate_metrics(y_val, y_arou, pred_val, pred_arou):
	valence_pearson = pearsonr(pred_val, y_val)[0]
	arousal_pearson = pearsonr(pred_arou, y_arou)[0]
	valence_mae = mean_absolute_error(pred_val, y_val)
	arousal_mae = mean_absolute_error(pred_arou, y_arou)
	valence_mse = mean_squared_error(pred_val, y_val)
	arousal_mse = mean_squared_error(pred_arou, y_arou)

	print("Valence Pearson: {} | Arousal Pearson: {} | Valence MAE: {} | Arousal MAE: {} | Valence MSE: {} | Arousal MSE: {}".format(	valence_pearson,
																											 							arousal_pearson, 
																											 							valence_mae, 
																											 							arousal_mae, 
																											 							valence_mse, 
																											 							arousal_mse))

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--sentences", help="path to sentences dataset file", type=str, required=True)
	parser.add_argument("--words", help="path to words dataset file", type=str, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	sentences, sentences_valences, sentences_arousals, scalerV, scalerA = load_sentences_data(args)
	words = load_words_data(args, scalerV, scalerA)
	train_avaliate_model(sentences, sentences_valences, sentences_arousals, words)
main()

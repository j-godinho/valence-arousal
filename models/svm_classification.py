import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

import argparse
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize

from gensim.models.wrappers import FastText

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr

try: 
	import matplotlib.pyplot as plt
except:
	print("Failed to import matplotlib")

def load_data(args):
	if("facebook" in args.data):	
		dataset = pd.read_csv(args.data)
		sentences = np.array(dataset["Anonymized Message"])
		arousals = np.array(dataset["Arousal_mean"]).reshape(-1, 1)
		valences = np.array(dataset["Valence_mean"]).reshape(-1, 1)
	else:
		dataset = pd.read_csv(args.data, sep='\\t')
		sentences = np.array(dataset["sentence"])
		arousals = np.array(dataset["Arousal"]).reshape(-1, 1)
		valences = np.array(dataset["Valence"]).reshape(-1, 1)

	scaler = MinMaxScaler(feature_range=(-1, 1))

	if(args.dimension == "valence"):
		valences = scaler.fit_transform(valences)
		return sentences, valences, scaler	
	else:
		arousals = scaler.fit_transform(arousals)
		return sentences, arousals, scaler

def encode_data(args, sentences):
	# Words list
	words = set()	
	for sentence in sentences:
		for word in nltk.word_tokenize(sentence):
			words.add(word)

	# Words dictionary
	words_dict = {w: i for i, w in enumerate(words)}		
	
	X = []
	if(args.type == "bag"):
		X = bag_of_words(X, sentences, words, words_dict)
	else:
		embeddings, emb_dim = load_embeddings(args)
		X = average_embeddings(X, sentences, words, embeddings, emb_dim)

	return X

def bag_of_words(X, sentences, words, words_dict):
	X = np.zeros((len(sentences), len(words)))
	for i in range(len(sentences)):
		for word in nltk.word_tokenize(sentences[i]):
			X[i][words_dict[word]] += 1
	return X

def average_embeddings(X, sentences, words, embeddings, emb_dim):
	X = np.zeros((len(sentences), emb_dim))
	for i in range(len(sentences)):
		count = 0
		for word in nltk.word_tokenize(sentences[i]):
			try:
				embedding_vector = embeddings[word.lower()]
				X[i] = np.add(X[i], embedding_vector)
				count += 1
			except:
				print("Not found embedding for: <{0}>".format(word))
		if(count != 0):
			X[i] /= count
	return X 

def build_model(x_train, x_test, y_train, y_test, scaler):
	model = SVR(kernel='rbf', C=1e3, gamma=0.1)

	predicted = model.fit(x_train, y_train.ravel()).predict(x_test)
	
	## Remove normalization
	predicted = predicted.reshape(-1, 1)
	predicted = scaler.inverse_transform(predicted) 
	
	y_test = y_test[:,0].reshape(-1, 1)
	y_test = scaler.inverse_transform(y_test)
	
	pearson = pearsonr(predicted, y_test)[0]
	mse = mean_squared_error(predicted, y_test)
	mae = mean_absolute_error(predicted, y_test)

	print("Pearson: {} | MAE: {} | MSE: {}".format(pearson, mae, mse))

	return pearson, mse, mae

def k_fold(args, X, Y, maximum):
	pearson = []
	mse = []
	mae = []

	kfold = KFold(n_splits=args.k)
	for train, test in kfold.split(X):
		x_train = X[train]
		x_test = X[test]
		y_train = Y[train]
		y_test = Y[test]
	
		p, s, a = build_model(x_train, x_test, y_train, y_test, maximum)

		pearson.append(p)
		mse.append(s)
		mae.append(a)
	
	print("[MEAN] {} - Pearson:{} | MAE:{} | MSE:{}".format(args,
															np.mean(pearson), 
															np.mean(mae),
															np.mean(mse)))

def load_embeddings(args):	
	if(args.fasttext):
		embeddings_dict = FastText.load_fasttext_format(args.fasttext) 
	elif(args.emb):
		embeddings_dict = np.load(args.emb).item()
	else:
		print("Error - No embeddings specified")

	return embeddings_dict, len(embeddings_dict['the'])

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", help="path to dataset file", type=str, required=True)
	parser.add_argument("--dimension", help="test <valence> or <arousal>?", type=str, required=True)
	parser.add_argument("--type", help="<bag> or <average>", type=str, required=True)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=False)
	parser.add_argument("--fasttext", help="use fasttext embeddings?", type=str, required=False)
	parser.add_argument("--k", help="number of foldings", type=int, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	sentences, Y, scaler = load_data(args)
	X = encode_data(args, sentences)
	k_fold(args, X, Y, scaler)

main()

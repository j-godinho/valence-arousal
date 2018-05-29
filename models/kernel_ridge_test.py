import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

from sklearn.feature_extraction.text import CountVectorizer
import argparse
import pandas as pd

from nltk.tokenize import word_tokenize
from gensim.models import FastText
import nltk

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import pearsonr

try: 
	import matplotlib.pyplot as plt
except:
	print("Failed to import matplotlib")	

def handle_generalization(args, words_train, valences_train, arousals_train, words_dict):
	dataset = pd.read_csv(args.anew, sep=',')
	words = np.asarray(dataset["Description"])

	valences = []
	arousals = []

	to_remove = []
	for w in words:
		for i in range(len(words_train)):
			if(w == words_train[i]):
				to_remove.append(i)
				valences.append(valences_train[i])
				arousals.append(arousals_train[i])
				continue

	words_train = np.delete(words_train, to_remove)
	valences_train = np.delete(valences_train, to_remove).reshape(-1, 1)
	arousals_train = np.delete(arousals_train, to_remove).reshape(-1, 1)

	valences = np.asarray(valences).reshape(-1, 1)
	arousals = np.asarray(arousals).reshape(-1, 1)

	print(len(words_train), len(valences_train), len(arousals_train), len(words), len(valences), len(arousals))

	# Encode words to integers
	x_train = np.array([words_dict[word] for word in words_train])
	x_test = np.array([words_dict[word] for word in words])

	return x_train, x_test, valences_train, valences, arousals_train, arousals

def load_data(args):
	dataset = pd.read_csv(args.wordratings, sep='\t')

	words = np.array(dataset["Description"])

	valences = np.array(dataset["Valence_value"]).reshape(-1,1)
	arousals = np.array(dataset["Arousal_value"]).reshape(-1,1)

	Y = np.concatenate((valences, arousals), axis=1)
	
	# Normalization
	scaler = MinMaxScaler(feature_range=(-1, 1))
	Y = scaler.fit_transform(Y)

	return words, Y, scaler

def encode_data(args, words):
	embeddings, emb_dim = load_embeddings(args)

	X = np.zeros((len(words) , emb_dim))
	for i, word in enumerate(words):
		try:
			embedding_vector = embeddings[word]
			X[i] = embedding_vector
		except:
			print("Not found embedding for: <{0}>".format(word))
	return X

def build_model(x_train, x_test, y_train, y_test, scaler):

	model = KernelRidge()

	model = model.fit(x_train, y_train)
	predicted = model.predict(x_test)

	# Results
	get_results(predicted, y_test, scaler)


def get_results(predicted, y_test, scaler):
	# Remove Normalization
	predicted = scaler.inverse_transform(predicted)
	y_test = scaler.inverse_transform(y_test)

	y_valence_test = y_test[:,0]
	y_arousal_test = y_test[:,1]

	test_valence_predict = predicted[:,0]
	test_arousal_predict = predicted[:,1]

	# Compute pearson correlation
	print("Pearson Correlation (Valence):", pearsonr(test_valence_predict, y_valence_test))
	print("Pearson Correlation (Arousal):", pearsonr(test_arousal_predict, y_arousal_test))
	print("Mean Absolute Error (Valence):",	mean_absolute_error(test_valence_predict, y_valence_test))
	print("Mean Absolute Error (Arousal):",	mean_absolute_error(test_arousal_predict, y_arousal_test))
	print("Mean Squared Error (Valence):", 	mean_squared_error(test_valence_predict, y_valence_test))
	print("Mean Squared Error (Arousal):",	mean_squared_error(test_arousal_predict, y_arousal_test))


def load_embeddings(args):	
	if(args.fasttext):
		embeddings_dict = FastText.load_fasttext_format(args.fasttext) 
	elif(args.emb):
		embeddings_dict = np.load(args.emb).item()
	else:
		print("Error - No embeddings specified")

	return embeddings_dict, 300

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--wordratings", help="path to dataset file", type=str, required=True)
	parser.add_argument("--fasttext", help="use fasttext embeddings?", type=str, required=False)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=False)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	words, Y, scaler = load_data(args)

	x_train, x_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test = handle_generalization(args, words, Y, words_dict)
	X = encode_data(args, words)
	build_model(x_train, x_test, y_train, y_test)

main()

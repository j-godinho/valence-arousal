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

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Flatten, Dropout, TimeDistributed, Bidirectional, InputLayer, Input, GRU, concatenate, GlobalMaxPooling1D
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping

import gc
import math

from gensim.models import FastText

import nltk
from nltk.tokenize import word_tokenize


try: 
	import matplotlib.pyplot as plt
except:
	print("Failed to import matplotlib")
	
from custom_layers.GlobalMaxPooling1DMasked import *
from custom_layers.Attention import *


def load_embeddings(args):	
	if(args.fasttext):
		embeddings_dict = FastText.load_fasttext_format(args.fasttext) 
	elif(args.emb):
		embeddings_dict = np.load(args.emb).item()
	else:
		print("Error - No embeddings specified")

	return embeddings_dict, len(embeddings_dict['the'])

def load_data(args):
	print("[LOADING DATA]")

	#load facebook dataset	
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
	
	# Normalization
	scalerV = MinMaxScaler(feature_range=(-1, 1))
	scalerA = MinMaxScaler(feature_range=(-1, 1))
	
	valences = scalerV.fit_transform(valences)
	arousals = scalerA.fit_transform(arousals)
	
	words = set()	
	for sentence in sentences:
		for word in nltk.word_tokenize(sentence):
			words.add(word)
	
	words_dict = {w: i for i, w in enumerate(words, start=1)}		
	tokenized_sentences = [[words_dict[word] for word in nltk.word_tokenize(sentence)] for sentence in sentences]
	vocab_size = len(words)	
	encoded_docs = sequence.pad_sequences(tokenized_sentences)

	return encoded_docs, np.concatenate((valences, arousals),axis=1), vocab_size, encoded_docs.shape[1], words, scalerV, scalerA

def get_word_classification(path):
	word_model = keras.models.load_model(path)
	initial_layer = word_model.get_layer("initial_layer")
	return initial_layer


def k_fold(args, embeddings, emb_dim, vocab_size, max_len, words, X, Y, scalerV, scalerA):
	valence_cor = []
	arousal_cor = []
	valence_mse = []
	arousal_mse = []
	valence_mae = []
	arousal_mae = [] 

	print("[K FOLD]")
	
	i = 0
	kfold = KFold(n_splits=args.k, random_state=2)
	for train, test in kfold.split(X):
		x_train = X[train]
		x_test = X[test]
		y_valence_train = Y[train][:,0]
		y_valence_test = Y[test][:,0]
		y_arousal_train = Y[train][:,1]
		y_arousal_test =  Y[test][:,1]

		model = build_model(args, embeddings, emb_dim, vocab_size, max_len, words)
		v_pearson, a_pearson, v_mse, a_mse, v_mae, a_mae = train_predict_model(model, x_train, x_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test, scalerV, scalerA)

		valence_cor.append(v_pearson)
		arousal_cor.append(a_pearson)
		valence_mse.append(v_mse)
		arousal_mse.append(a_mse)
		valence_mae.append(v_mae)
		arousal_mae.append(a_mae)

		print("Index:{0} - V_p:{1} | A_p:{2} | V_mae:{3} | A_mae:{4} | V_mse:{5} | A_mse:{6}".format(i, v_pearson, a_pearson, v_mae, a_mae, v_mse, a_mse))
		i = i + 1
	
	print("[MEAN]: args: {} | V_p:{} | A_p:{} | V_mae:{} | A_mae:{} | V_mse:{} | A_mse:{}".format(args,	
																								np.mean(valence_cor), 
																								np.mean(arousal_cor),
																								np.mean(valence_mae),
																								np.mean(arousal_mae),
																								np.mean(valence_mse), 
																								np.mean(arousal_mse)))
	


def build_model(args, embeddings, emb_dim, vocab_size, max_len, words):

	if(args.wordratings):
		dense_layer = get_word_classification(args.wordratings)
	else:
		dense_layer = Dense(120, activation="tanh")

	# Embedding layer
	#embeddings_matrix = np.zeros((vocab_size+1, emb_dim))
	embeddings_matrix = np.random.rand(vocab_size + 1, emb_dim)
	embeddings_matrix[0] *= 0

	for index, word in enumerate(words, start=1):
		try:
			embedding_vector = embeddings[word]
			embeddings_matrix[index] = embedding_vector
		except:
			print("Not found embedding for: <{0}>".format(word))

	input_layer = Input(shape=(max_len,))

	embedding_layer = Embedding(embeddings_matrix.shape[0], 
								embeddings_matrix.shape[1], 
								weights = [embeddings_matrix],
								mask_zero=True,
								trainable=False)(input_layer)

	layer1 = TimeDistributed(dense_layer)(embedding_layer)

	# Recurrent layer. Either LSTM or GRU
	if(args.rnn == "LSTM"):
		rnn_layer = LSTM(units=64, return_sequences=(args.attention or args.maxpooling))
	else:
		rnn_layer = GRU(units=64, return_sequences=(args.attention or args.maxpooling))

	rnn = Bidirectional(rnn_layer)(layer1)

	# Max Pooling and attention
	if(args.maxpooling and args.attention):
		max_pooling = GlobalMaxPooling1DMasked()(rnn)
		attention = Attention()(rnn)
		connection = concatenate([max_pooling, attention])
	elif(args.maxpooling):
		max_pooling = GlobalMaxPooling1DMasked()
		connection = max_pooling(rnn)
	elif(args.attention):
		attention = Attention()
		connection = attention(rnn)
	else:
		connection = rnn

	valence_output = Dense(1, activation="tanh", name="valence_output")(connection)
	arousal_output = Dense(1, activation="tanh", name="arousal_output")(connection)

	# Build Model
	model = Model(inputs=[input_layer], outputs=[valence_output, arousal_output])
	return model


def train_predict_model(model, x_train, x_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test, scalerV, scalerA):

	earlyStopping = EarlyStopping(patience=1)

	adamOpt = keras.optimizers.Adam(lr=0.001, amsgrad=True)

	# Compilation
	model.compile(loss={"valence_output" : "mean_squared_error", "arousal_output" : "mean_squared_error"}, optimizer=adamOpt)

	print("{},{},{},{},{}".format(x_train.shape, y_valence_train.shape, y_arousal_train.shape, x_test.shape, y_valence_test.shape, y_arousal_test.shape))

	# Training
	history = model.fit( x_train, 
						{"valence_output": y_valence_train, "arousal_output": y_arousal_train}, 
						validation_data=(x_test, {"valence_output": y_valence_test, "arousal_output": y_arousal_test}), 
						batch_size=20, 
						epochs=100,
						callbacks = [earlyStopping])
	
	# Predictions
	test_predict = model.predict(x_test)

	test_valence_predict = test_predict[0].reshape(-1,1)
	test_arousal_predict = test_predict[1].reshape(-1,1)
	y_valence_test = y_valence_test.reshape(-1, 1)
	y_arousal_test = y_arousal_test.reshape(-1, 1)

	# Remove normalization
	test_valence_predict = scalerV.inverse_transform(test_valence_predict)
	test_arousal_predict = scalerA.inverse_transform(test_arousal_predict)
	y_valence_test = scalerV.inverse_transform(y_valence_test)
	y_arousal_test = scalerA.inverse_transform(y_arousal_test)

	# Compute metrics
	valence_pearson = pearsonr(test_valence_predict, y_valence_test)[0]
	arousal_pearson = pearsonr(test_arousal_predict, y_arousal_test)[0]
	valence_mae = mean_absolute_error(test_valence_predict, y_valence_test)
	arousal_mae = mean_absolute_error(test_arousal_predict, y_arousal_test)
	valence_mse = mean_squared_error(test_valence_predict, y_valence_test)
	arousal_mse = mean_squared_error(test_arousal_predict, y_arousal_test)
	
	return valence_pearson, arousal_pearson, valence_mse, arousal_mse, valence_mae, arousal_mae

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", help="path to dataset file", type=str, required=True)
	parser.add_argument("--fasttext", help="use fasttext embeddings?", type=str, required=False)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=False)
	parser.add_argument("--wordratings", help="path to word ratings model", type=str, required=False)
	parser.add_argument("--k", help="number of foldings", type=int, required=True)
	parser.add_argument("--attention", help="use attention layer", action="store_true")
	parser.add_argument("--maxpooling", help="use MaxPooling layer", action="store_true")
	parser.add_argument("--rnn", help="type of recurrent layer <LSTM>|<GRU>", type=str, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	X, Y, vocab_size, max_len, words, scalerV, scalerA = load_data(args)
	embeddings, emb_dim = load_embeddings(args)
	k_fold(args, embeddings, emb_dim, vocab_size, max_len, words, X, Y, scalerV, scalerA)

main()

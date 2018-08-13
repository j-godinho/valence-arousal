import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
from keras import backend as K

import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, InputLayer, Input
from keras import optimizers
from keras.callbacks import EarlyStopping

#from gensim.models import FastText

try: 
	import matplotlib.pyplot as plt
except:
	print("Failed to import matplotlib")

def load_embeddings(args):	
	if(args.fasttext):
		embeddings_dict = FastText.load_fasttext_format(args.fasttext) 
	elif(args.emb):
		embeddings_dict = np.load(args.emb).item()
	else:
		print("Error - No embeddings specified")

	return embeddings_dict, 300


def handle_generalization(args, words_train, valences_train, arousals_train, words_dict):
	dataset = pd.read_csv(args.anew, sep=',')
	words = np.asarray(dataset["Description"])

	valences = []
	arousals = []

	to_remove = []
	for w in words:
		for i in range(len(words_train)):
			#print(w, words_train[i])
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


def get_word_classification(args, embeddings, emb_dim):
	dataset = pd.read_csv(args.wordratings, sep='\t')
		
	words = np.array(dataset["Description"])
	valences = np.array(dataset["Valence_value"]).reshape(-1,1)
	arousals = np.array(dataset["Arousal_value"]).reshape(-1,1)

	words_dict = {w: i for i, w in enumerate(words)}		
	 
	# Normalize to same values
	scalerV = MinMaxScaler(feature_range=(0, 1))
	valences = scalerV.fit_transform(valences)
	
	scalerA = MinMaxScaler(feature_range=(0, 1))
	arousals = scalerA.fit_transform(arousals)

	x_train, x_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test = handle_generalization(args, words, valences, arousals, words_dict)
	
	# Build embeddings layer
	embeddings_matrix = np.zeros((len(words), emb_dim))

	for index, word in enumerate(words):
		try:
			embedding_vector = embeddings[word]
			embeddings_matrix[index] = embedding_vector
		except:
			print("Not found embedding for: <{0}>".format(word))
		
	input_layer = Input(shape=(1,))

	print("Matrix: {0}, x_train: {1}, x_test: {2}, y_valence_train: {3} , y_valence_test: {4}, y_arousal_train: {5}, y_arousal_test: {6}.".format(
			embeddings_matrix.shape, 
			x_train.shape,
			x_test.shape, 
			y_valence_train.shape,
			y_valence_test.shape,
			y_arousal_train.shape,
			y_arousal_test.shape))

	embedding_layer = Embedding(embeddings_matrix.shape[0], 
								embeddings_matrix.shape[1], 
								weights = [embeddings_matrix],
								trainable=False)(input_layer)

	flatten = Flatten()(embedding_layer)
	dense_layer = Dense(120, name="initial_layer", activation="relu")
	connection_dense = dense_layer(flatten)
	valence_output = Dense(1, activation="sigmoid", name="valence_output")(connection_dense)
	arousal_output = Dense(1, activation="sigmoid", name="arousal_output")(connection_dense)

	model = Model(inputs=[input_layer], outputs=[valence_output, arousal_output])
	
	adamOpt = keras.optimizers.Adam(lr=0.001, amsgrad=True)
	earlyStopping = EarlyStopping(patience=1)

	# Compilation
	model.compile(loss={"valence_output" : "mean_squared_error", "arousal_output" : "mean_squared_error"}, optimizer=adamOpt)

	# Training
	history = model.fit(	x_train, 
							{"valence_output": y_valence_train, "arousal_output": y_arousal_train}, 
							validation_data=(x_test, {"valence_output": y_valence_test, "arousal_output": y_arousal_test}), 
							batch_size=5, 
							epochs=100,
							callbacks = [earlyStopping])

	# Evaluation
	test_predict = np.array(model.predict(x_test))

	test_valence_predict = test_predict[0]
	test_arousal_predict = test_predict[1]

	# Undo normalization
	test_valence_predict = scalerV.inverse_transform(test_valence_predict)
	test_arousal_predict = scalerA.inverse_transform(test_arousal_predict)
	
	y_valence_test = scalerV.inverse_transform(y_valence_test)
	y_arousal_test = scalerA.inverse_transform(y_arousal_test)

	# Finish undo normalization

	for i in range(len(test_valence_predict)):
		print("{}: {}-{} | {}-{}".format(words[x_test[i]], test_valence_predict[i], y_valence_test[i], test_arousal_predict[i], y_arousal_test[i]))
	
	#for i in range(len(y_valence_test)):
	#	print("{}: {}-{}".format(words[x_test[i]], y_valence_test[i], y_arousal_test[i]))


	# Compute pearson correlation
	print("Pearson Correlation (Valence):", pearsonr(test_valence_predict, y_valence_test))
	print("Pearson Correlation (Arousal):", pearsonr(test_arousal_predict, y_arousal_test))
	print("Mean Absolute Error (Valence):",	mean_absolute_error(test_valence_predict, y_valence_test))
	print("Mean Absolute Error (Arousal):",	mean_absolute_error(test_arousal_predict, y_arousal_test))
	print("Mean Squared Error (Valence):", 	mean_squared_error(test_valence_predict, y_valence_test))
	print("Mean Squared Error (Arousal):",	mean_squared_error(test_arousal_predict, y_arousal_test))
	

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--fasttext", help="path to Fasttext binary file", type=str, required=False)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=True)
	parser.add_argument("--wordratings", help="path to word ratings file", type=str, required=True)
	parser.add_argument("--anew", help="path to generalization dataset file", type=str, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	embeddings, emb_dim = load_embeddings(args)
	get_word_classification(args, embeddings, emb_dim)
	
main()

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

from gensim.models.wrappers import FastText

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

	return embeddings_dict, len(embeddings_dict['the'])

def get_word_classification(args, embeddings, emb_dim):
	dataset = pd.read_csv(args.wordratings)
		
	words = np.array(dataset["Word"])
	arousals = np.array(dataset["A.Mean.Sum"]).reshape(-1,1)
	valences = np.array(dataset["V.Mean.Sum"]).reshape(-1,1)

	# Normalization arousals and valences to -1 - 1 range
	scalerA = MinMaxScaler(feature_range=(-1, 1))
	arousals = scalerA.fit_transform(arousals)

	scalerV = MinMaxScaler(feature_range=(-1, 1))
	valences = scalerV.fit_transform(valences)


	words_dict = {w: i for i, w in enumerate(words)}		
	X = np.array([words_dict[word] for word in words])

	x_train, x_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test = train_test_split(X, valences, arousals, test_size=0.1)

	# Build embeddings layer
	embeddings_matrix = np.zeros((len(words), emb_dim))

	for index, word in enumerate(words):
		try:
			embedding_vector = embeddings[word.lower()]
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
	valence_output = Dense(1, activation="tanh", name="valence_output")(connection_dense)
	arousal_output = Dense(1, activation="tanh", name="arousal_output")(connection_dense)

	model = Model(inputs=[input_layer], outputs=[valence_output, arousal_output])
	
	adamOpt = keras.optimizers.Adam(lr=0.001)
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

	# Compute pearson correlation
	print("Pearson Correlation (Valence):", pearsonr(test_valence_predict, y_valence_test))
	print("Pearson Correlation (Arousal):", pearsonr(test_arousal_predict, y_arousal_test))
	print("Mean Absolute Error (Valence):",	mean_absolute_error(test_valence_predict, y_valence_test))
	print("Mean Absolute Error (Arousal):",	mean_absolute_error(test_arousal_predict, y_arousal_test))
	print("Mean Squared Error (Valence):", 	mean_squared_error(test_valence_predict, y_valence_test))
	print("Mean Squared Error (Arousal):",	mean_squared_error(test_arousal_predict, y_arousal_test))
	
	model.save("saved_models/wordclassification.h5")

	

def plot_figure(test_valence_predict, y_valence_test, test_arousal_predict, y_arousal_test):
	fig = plt.figure(1)

	#valence
	val = fig.add_subplot(211)
	val.plot(test_valence_predict)
	val.plot(y_valence_test)
	val.set_ylabel('Valence')

	#arousal
	arousal = fig.add_subplot(212)
	arousal.plot(test_arousal_predict)
	arousal.plot(y_arousal_test)
	arousal.set_ylabel("Arousal")

	plt.show()

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--fasttext", help="path to Fasttext binary file", type=str, required=False)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=False)
	parser.add_argument("--wordratings", help="path to word ratings file", type=str, required=False)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	embeddings, emb_dim = load_embeddings(args)
	get_word_classification(args, embeddings, emb_dim)
	
main()

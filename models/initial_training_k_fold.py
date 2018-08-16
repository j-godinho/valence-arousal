import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
from keras import backend as K

import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

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

def load_data(args):
	dataset = pd.read_csv(args.wordratings, sep='\t')
	words = np.array(dataset["Description"])
	valences = np.array(dataset["Valence_value"]).reshape(-1,1)
	arousals = np.array(dataset["Arousal_value"]).reshape(-1,1)

	# Normalization arousals and valences to -1 - 1 range
	scalerA = MinMaxScaler(feature_range=(0, 1))
	arousals = scalerA.fit_transform(arousals)

	scalerV = MinMaxScaler(feature_range=(0, 1))
	valences = scalerV.fit_transform(valences)

	words_dict = {w: i for i, w in enumerate(words)}		
	X = np.array([words_dict[word] for word in words])

	return X, np.concatenate((valences, arousals), axis=1), words, scalerV, scalerA

def k_fold(args, embeddings, emb_dim, words, X, Y, scalerV, scalerA):
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

		model = build_model(args, embeddings, emb_dim, words)
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
	print("[VAR]: V_p:{} | A_p:{} | V_mae:{} | A_mae:{} | V_mse:{} | A_mse:{}".format(			np.var(valence_cor), 
																								np.var(arousal_cor),
																								np.var(valence_mae),
																								np.var(arousal_mae),
																								np.var(valence_mse), 
																								np.var(arousal_mse)))

def build_model(args, embeddings, emb_dim, words):
	# Build embeddings layer
	embeddings_matrix = np.zeros((len(words), emb_dim))

	for index, word in enumerate(words):
		try:
			embedding_vector = embeddings[word]
			embeddings_matrix[index] = embedding_vector
		except:
			print("Not found embedding for: <{0}>".format(word))
		
	input_layer = Input(shape=(1,))

	embedding_layer = Embedding(embeddings_matrix.shape[0], 
								embeddings_matrix.shape[1], 
								weights = [embeddings_matrix],
								trainable=False)(input_layer)

	flatten = Flatten()(embedding_layer)
	dense_layer = Dense(120, name="initial_layer", activation="relu")
	connection_dense = dense_layer(flatten)
	connection_dense = Dropout(0.2)(connection_dense)
	valence_output = Dense(1, activation="sigmoid", name="valence_output")(connection_dense)
	arousal_output = Dense(1, activation="sigmoid", name="arousal_output")(connection_dense)

	model = Model(inputs=[input_layer], outputs=[valence_output, arousal_output])
	return model

def train_predict_model(model, x_train, x_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test, scalerV, scalerA):
		
	adamOpt = keras.optimizers.Adam(lr=0.001, amsgrad=True)
	earlyStopping = EarlyStopping(patience=3)

	# Compilation
	model.compile(loss={"valence_output" : "mean_squared_error", "arousal_output" : "mean_squared_error"}, optimizer=adamOpt)

	# Training
	history = model.fit(	x_train, 
							{"valence_output": y_valence_train, "arousal_output": y_arousal_train}, 
							validation_data=(x_test, {"valence_output": y_valence_test, "arousal_output": y_arousal_test}), 
							#validation_split=0.1,
							batch_size=20, 
							epochs=10,
							#callbacks = [earlyStopping]
							)

	# Evaluation
	test_predict = np.array(model.predict(x_test))

	test_valence_predict = test_predict[0].reshape(-1,1)
	test_arousal_predict = test_predict[1].reshape(-1,1)
	y_valence_test = y_valence_test.reshape(-1, 1)
	y_arousal_test = y_arousal_test.reshape(-1, 1)

	# Undo normalization
	test_valence_predict = scalerV.inverse_transform(test_valence_predict)
	test_arousal_predict = scalerA.inverse_transform(test_arousal_predict)
	
	y_valence_test = scalerV.inverse_transform(y_valence_test)
	y_arousal_test = scalerA.inverse_transform(y_arousal_test)

	# Finish undo normalization

	# Compute pearson correlation
	pearson_valence = pearsonr(test_valence_predict, y_valence_test)[0]
	pearson_arousal = pearsonr(test_arousal_predict, y_arousal_test)[0]
	mae_valence = mean_absolute_error(test_valence_predict, y_valence_test)
	mae_arousal = mean_absolute_error(test_arousal_predict, y_arousal_test)
	mse_valence = mean_squared_error(test_valence_predict, y_valence_test)
	mse_arousal = mean_squared_error(test_arousal_predict, y_arousal_test)

	#print("Pearson Correlation (Valence):", pearson_valence)
	#print("Pearson Correlation (Arousal):", pearson_arousal)
	#print("Mean Absolute Error (Valence):",	mae_valence)
	#print("Mean Absolute Error (Arousal):",	mae_arousal)
	#print("Mean Squared Error (Valence):", 	mse_valence)
	#print("Mean Squared Error (Arousal):",	mse_arousal)

	return pearson_valence, pearson_arousal, mae_valence, mae_arousal, mse_valence, mse_arousal
	
	#model.save("saved_models/wordclassification.h5")


def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--fasttext", help="path to Fasttext binary file", type=str, required=False)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=False)
	parser.add_argument("--wordratings", help="path to word ratings file", type=str, required=True)
	parser.add_argument("--k", help="number of folds", type=int, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	X, Y, words, scalerV, scalerA = load_data(args)
	embeddings, emb_dim = load_embeddings(args)
	k_fold(args, embeddings, emb_dim, words, X, Y, scalerV, scalerA)

main()

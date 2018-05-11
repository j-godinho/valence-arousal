import numpy as np
import io
import argparse
import pandas as pd
import nltk
import os

def load_vectors(args):
	fin = io.open(args.emb, 'r', encoding='utf-8', newline='\n', errors='ignore')
	n, d = map(int, fin.readline().split())
	data = {}
	for l in fin:
		line = l.split()
		word, coefs = line[0], np.asarray(line[1:], dtype='float32')
		data[word] = coefs
	return data

def load_data(args):
	print("Loading Data")

	if("facebook" in args.data):	
		dataset = pd.read_csv(args.data)
		sentences = np.array(dataset["Anonymized Message"])
		data_type = 0
		name = "facebook.embs"
	elif("combination" in args.data):
		dataset = pd.read_csv(args.data, sep='\\t')
		sentences = np.array(dataset["sentence"])
		data_type = 0
		name = "combination.embs"
	elif("warriner" in args.data):
		data_type = 1
		name = "warriner.embs"

	words = set()	
	if(data_type == 0):
		for sentence in sentences:
			for word in nltk.word_tokenize(sentence):
				words.add(word)
		
	return words, name

def calculate_oov_words(args, embeddings, words, name):
	print("Populating matrix")

	f = open("../fasttext/temp.embs", "w")
	data = {}
	for index, word in enumerate(words, start=1):
		lower_word = word.lower()
		try:
			data[lower_word] = embeddings[lower_word]
		except:
			print("Adding to file: <{0}>".format(lower_word))
			f.write(lower_word + "\n")

	f.close()

	print("Executing line to predict OOV words")
	exec_line = "{0} print-word-vectors {1} < ../fasttext/temp.embs > ../fasttext/outputtemp.embs".format(args.fasttext, args.binary)
	os.system(exec_line)

	print("Merging files")
	fin = io.open("../fasttext/outputtemp.embs", 'r', encoding='utf-8', newline='\n', errors='ignore')
	for l in fin:
		line = l.split()
		word, coefs = line[0], np.asarray(line[1:], dtype='float32')
		data[word] = coefs
	fin.close()

	np.save("../fasttext/" + name, data)


def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--fasttext", help="fasttext instalation file", type=str, required=True)
	parser.add_argument("--data", help="data path", type=str, required=True)
	parser.add_argument("--binary", help="fasttext binary file", type=str, required=True)
	parser.add_argument("--emb", help="pre trained vector embedding file", type=str, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	words, name = load_data(args)
	embeddings = load_vectors(args)
	calculate_oov_words(args, embeddings, words, name)

main()
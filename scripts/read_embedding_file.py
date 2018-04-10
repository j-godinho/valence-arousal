import numpy as np
import argparse

def read_embedding(args):
	print("[Reading embeddings file]")
	embeddings_dict = {}
	with open(args.input, 'r') as emb_file:
	   header = emb_file.readline().split()
	   n_embeddings, emb_dim = int(header[0]), int(header[1])
	   for l in emb_file:
	       line = l.split()
	       word, coefs = line[0], np.asarray(line[1:(emb_dim + 1)], dtype='float32')
	       embeddings_dict[word] = coefs	

	print("[Saving file]")
	np.save(args.output, embeddings_dict)

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", help="path to embeddings file", type=str, required=True)
	parser.add_argument("--output", help="path to output file", type=str, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	read_embedding(args)

main()
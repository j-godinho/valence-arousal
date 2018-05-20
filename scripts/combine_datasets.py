import argparse
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def get_files():

	# ANEW
	dataset = pd.read_csv(args.anew, sep=",")
	words = dataset["Description"]
	valences = dataset["Valence Mean"]
	arousals = dataset["Arousal Mean"]

	anew = {words[i]: [valences[i], arousals[i]] for i in range(len(words))}
	
	# NRC
	dataset = pd.read_csv(args.nrc, sep="\t")
	words = dataset["word1"]
	valences = dataset["valence_value"]
	arousals = dataset["arousal_value"]

	scaler = MinMaxScaler(feature_range=(1, 9))
	valences = scaler.fit_transform(valences.reshape(-1, 1)).flatten()
	arousals = scaler.fit_transform(arousals.reshape(-1, 1)).flatten()

	nrc = {words[i]: [valences[i], arousals[i]] for i in range(len(words))}
	
	# Warriner
	dataset = pd.read_csv(args.warriner)
	words = dataset["Word"]
	arousals = dataset["A.Mean.Sum"]
	valences = dataset["V.Mean.Sum"]

	warriner = {words[i]: [valences[i], arousals[i]] for i in range(len(words))}
	
	if("fulfillment" in anew):
		print(anew['fulfillment'])
	if("fulfillment" in nrc):
		print(nrc['fulfillment'])
	if("fulfillment" in warriner):
		print(warriner['fulfillment'])

	return anew, nrc, warriner

def array_mean(array):
	new_array = np.asarray(array)
	return np.mean(new_array)

def combine_file(dic1, other1, other2, global_dict):
	for key in dic1:
		val = []
		aro = []
		val.append(dic1[key][0])
		aro.append(dic1[key][1])
		if(key in other1):
			val.append(other1[key][0])
			aro.append(other1[key][1])
		if(key in other2):
			val.append(other2[key][0])
			aro.append(other2[key][1])
		if(key not in global_dict):
			global_dict[key] = [array_mean(val), array_mean(aro)]

def merge_files(dic1, dic2, dic3):
	global_dict = {}

	combine_file(dic1, dic2, dic3, global_dict)
	combine_file(dic2, dic1, dic3, global_dict)
	combine_file(dic3, dic2, dic1, global_dict)

	if("fulfillment" in global_dict):
		print(global_dict['fulfillment'])
	write_dic_to_file(global_dict)

def write_dic_to_file(global_dict):
	f = open(args.output, 'w')
	f.write("Description\tValence_value\tArousal_value\n")
	for key in global_dict:
		f.write("{}\t{}\t{}\n".format(key, float(global_dict[key][0]), float(global_dict[key][1])))
	f.close()

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--anew", help="path to anew dataset file", type=str, required=True)
	parser.add_argument("--nrc", help="path to nrc dataset file", type=str, required=True)
	parser.add_argument("--warriner", help="path to warriner dataset file", type=str, required=True)
	parser.add_argument("--output", help="path to output dataset file", type=str, required=True)
	args = parser.parse_args()
	return args

def main():
	args = receive_arguments()
	f1, f2, f3 = get_files()
	merge_files(f1, f2, f3)

args = receive_arguments()
main()

import numpy as np
import argparse
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt 

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

def load_data(args):
	path = args.path


	with open(path+"/valence_cor", 'rb') as f:
		valence_cor = pickle.load(f)
	with open(path+"/arousal_cor", 'rb') as f:
		arousal_cor = pickle.load(f)
	with open(path+"/valence_mae", 'rb') as f:
		valence_mae = pickle.load(f)
	with open(path+"/arousal_mae", 'rb') as f:
		arousal_mae = pickle.load(f)
	with open(path+"/valence_mse", 'rb') as f:
		valence_mse = pickle.load(f)
	with open(path+"/arousal_mse", 'rb') as f:
		arousal_mse = pickle.load(f)

	#valences = [v_face, v_emo, v_anet]
	#arousals = [a_face, a_emo, a_anet]
	#data_to_plot = [v_face, a_face, v_emo, a_emo, v_anet, a_anet]
	#names_to_plot = ["Facebook", "EmoBank", "ANET"]

	plt.figure(1)

	plot_boxes2([valence_cor], [arousal_cor], ["correlations"], 211)
	
	#valences = [v_anew, v_warriner, v_nrc]
	#arousals = [a_anew, a_warriner, a_nrc]
	#names_to_plot = ["ANEW", "Warriner", "NRC"]
	
	#plot_boxes2(valences, arousals, names_to_plot, 212)

	#plt.show()


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)	

def plot_boxes(data, names):
	# Create a figure instance
	fig = plt.figure(1, figsize=(9, 6))

	# Create an axes instance
	ax = fig.add_subplot(111)

	# Create the boxplot
	bp = ax.boxplot(data)

	plt.show()

def plot_boxes2(valence, arousal, names, plot):
	plt.subplot(plot)

	v_color = '#D7191C'
	a_color = '#2C7BB6'

	bpl = plt.boxplot(valences, positions=np.array(range(len(valences)))*2.0-0.4, sym='', widths=0.6)
	bpr = plt.boxplot(arousals, positions=np.array(range(len(arousals)))*2.0+0.4, sym='', widths=0.6)
	set_box_color(bpl, v_color) # colors are from http://colorbrewer2.org/
	set_box_color(bpr, a_color)

	# draw temporary red and blue lines and use them to create a legend
	#plt.plot([], c=v_color, label='Valences')
	#plt.plot([], c=a_color, label='Arousals')
	#plt.legend()

	plt.xticks(range(0, len(names) * 2, 2), names)
	plt.xlim(-2, len(names)*2)
	#plt.ylim(0, 8)
	plt.tight_layout()
	#plt.savefig('boxcompare.png')
	

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def receive_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", help="path", type=str, required=True)
	args = parser.parse_args()
	return args


def main():
	args = receive_arguments()
	load_data(args)

main()
import numpy as np
import argparse
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt 

import seaborn as sns

def load_data(args):

	# Facebook
	facebook = pd.read_csv(args.facebook)
	v_face = np.asarray(facebook["Arousal_mean"])
	a_face = np.asarray(facebook["Valence_mean"])

	# EmoBank
	emobank = pd.read_csv(args.emobank, sep='\t')
	v_emo = np.asarray(emobank["Arousal"])
	a_emo = np.asarray(emobank["Valence"])

	# ANET
	anet = pd.read_csv(args.anet, sep='\t')
	v_anet = np.asarray(anet["AroMN"])
	a_anet = np.asarray(anet["PlMN"])

	# ANEW
	anew = pd.read_csv(args.anew, sep=',')
	v_anew = np.asarray(anew["Valence Mean"])
	a_anew = np.asarray(anew["Arousal Mean"])

	# Warriner
	warriner = pd.read_csv(args.warriner)
	v_warriner = np.asarray(warriner["A.Mean.Sum"])
	a_warriner = np.asarray(warriner["V.Mean.Sum"])

	# NRC
	nrc = pd.read_csv(args.nrc, sep='\t')
	v_nrc = np.asarray(nrc["valence_value"])
	a_nrc = np.asarray(nrc["arousal_value"])

	# Word Combination
	combination = pd.read_csv(args.combination, sep='\t')
	v_comb = np.asarray(combination["Valence_value"])
	a_comb = np.asarray(combination["Arousal_value"])

	valences = [v_face, v_emo, v_anet]
	arousals = [a_face, a_emo, a_anet]
	#data_to_plot = [v_face, a_face, v_emo, a_emo, v_anet, a_anet]
	names_to_plot = ["Facebook", "EmoBank", "ANET"]
	
	plt.figure(1)

	plot_boxes2(valences, arousals, names_to_plot, 211)
	
	valences = [v_anew, v_warriner, v_nrc, v_comb]
	arousals = [a_anew, a_warriner, a_nrc, a_comb]
	names_to_plot = ["ANEW", "Warriner", "NRC", "Combination"]
	
	plot_boxes2(valences, arousals, names_to_plot, 212)

	plt.show()

	plt.figure(2)

	plot_density(v_face, a_face, "Facebook", 231)
	plot_density(v_emo, a_emo, "EmoBank", 232)
	plot_density(v_anet, a_anet, "ANET", 233)
	plot_density(v_anew, a_anew, "ANEW", 234)
	plot_density(v_warriner, a_warriner, "Warriner", 235)
	plot_density(v_nrc, a_nrc, "NRC", 236)
	#plot_density(v_comb, a_comb, "Combination", 236)

	plt.show()


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

def plot_density(x, y, name, plot):
	plt.subplot(plot)
	plt.title(name)
	plt.tight_layout()
	sns.set_style("white")
	ax = sns.kdeplot(x, y, shade=True, shade_lowest=False)
	ax.set(xlabel='Valences', ylabel='Arousals')	

def plot_boxes(data, names):
	# Create a figure instance
	fig = plt.figure(1, figsize=(9, 6))

	# Create an axes instance
	ax = fig.add_subplot(111)

	# Create the boxplot
	bp = ax.boxplot(data)

	plt.show()

def plot_boxes2(valences, arousals, names, plot):
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
	parser.add_argument("--facebook", help="facebook dataset", type=str, required=True)
	parser.add_argument("--emobank", help="emobank dataset", type=str, required=True)
	parser.add_argument("--anet", help="anet dataset", type=str, required=True)
	parser.add_argument("--anew", help="anew dataset", type=str, required=True)
	parser.add_argument("--warriner", help="warriner dataset", type=str, required=True)
	parser.add_argument("--nrc", help="nrc dataset", type=str, required=True)
	parser.add_argument("--combination", help="combination dataset", type=str, required=True)
	args = parser.parse_args()
	return args


def main():
	args = receive_arguments()
	load_data(args)

main()
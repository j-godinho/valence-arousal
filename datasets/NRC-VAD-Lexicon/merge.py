import pandas as pd

dataset1 = pd.read_csv("a.scores", sep='\t')
dataset2 = pd.read_csv("v.scores", sep='\t')


left_merge = pd.merge(left=dataset1, right=dataset2, on="word1", how="inner")
#print(left_merge.head())
#print(left_merge)

left_merge.to_csv("comb.scores", sep='\t', index=False, float_format="%.3f")

##### Initial Training
python3 initial_training.py --wordratings ../datasets/Ratings_Warriner_et_al.csv --emb ../../embeddings_dict.npy 

##### Regular SVM

## SVM EmoBank k=10 valence bag 
python3 svm_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --dimension valence --type bag

## SVM EmoBank k=10 arousal bag 
python3 svm_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --dimension arousal --type bag

## SVM facebook k=10 valence bag 
python3 svm_classification.py --data ../datasets/facebook_posts.csv  --k 10 --dimension valence --type bag

## SVM facebook k=10 arousal bag 
python3 svm_classification.py --data ../datasets/facebook_posts.csv  --k 10 --dimension arousal --type bag

##### Avg Embedding SVM

## SVM EmoBank k=10 valence average 
python3 svm_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --dimension valence --type average --emb ../../embeddings_dict.npy

## SVM EmoBank k=10 arousal average 
python3 svm_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --dimension arousal --type average --emb ../../embeddings_dict.npy

## SVM facebook k=10 valence average 
python3 svm_classification.py --data ../datasets/facebook_posts.csv  --k 10 --dimension valence --type average --emb ../../embeddings_dict.npy

## SVM facebook k=10 arousal average 
python3 svm_classification.py --data ../datasets/facebook_posts.csv  --k 10 --dimension arousal --type average --emb ../../embeddings_dict.npy

##### Bi-LSTM

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn LSTM  --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn LSTM  --emb ../../embeddings_dict.npy
 

##### Bi-GRU

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn GRU --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn GRU  --emb ../../embeddings_dict.npy

##### Bi-LSTM + Attention

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn LSTM --attention --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn LSTM --attention --emb ../../embeddings_dict.npy

##### Bi-LSTM + Max Pooling

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn LSTM --maxpooling --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn LSTM --maxpooling --emb ../../embeddings_dict.npy

##### Bi-LSTM + Max Pooling + Attention

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn LSTM --attention --maxpooling --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn LSTM --attention --maxpooling --emb ../../embeddings_dict.npy

##### Bi-LSTM + Max Pooling + Attention + Initial Training

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn LSTM --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn LSTM --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --emb ../../embeddings_dict.npy

##### Bi-GRU + Max Pooling + Attention + Initial Training

## EmoBank
python3 quotation_classification.py --data ../datasets/EmoBank/combination.tsv --k 10 --rnn GRU --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --emb ../../embeddings_dict.npy

## Facebook
python3 quotation_classification.py --data ../datasets/facebook_posts.csv --k 10 --rnn GRU --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --emb ../../embeddings_dict.npy
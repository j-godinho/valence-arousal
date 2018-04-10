##### Regular SVM

## SVM EmoBank k=10 valence bag 
python3 svm_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --dimension valence --type bag

## SVM EmoBank k=10 arousal bag 
python3 svm_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --dimension arousal --type bag

## SVM facebook k=10 valence bag 
python3 svm_classification.py --data ../datasets/opinion_mining/facebook_posts.csv  --k 10 --dimension valence --type bag

## SVM facebook k=10 arousal bag 
python3 svm_classification.py --data ../datasets/opinion_mining/facebook_posts.csv  --k 10 --dimension arousal --type bag

##### Avg Embedding SVM

## SVM EmoBank k=10 valence average 
python3 svm_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --dimension valence --type average --fasttext ../embeddings/fasttext/wiki.en

## SVM EmoBank k=10 arousal average 
python3 svm_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --dimension arousal --type average --fasttext ../embeddings/fasttext/wiki.en

## SVM facebook k=10 valence average 
python3 svm_classification.py --data ../datasets/opinion_mining/facebook_posts.csv  --k 10 --dimension valence --type average --fasttext ../embeddings/fasttext/wiki.en

## SVM facebook k=10 arousal average 
python3 svm_classification.py --data ../datasets/opinion_mining/facebook_posts.csv  --k 10 --dimension arousal --type average --fasttext ../embeddings/fasttext/wiki.en

##### Bi-LSTM

## EmoBank
python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn LSTM  --fasttext ../embeddings/fasttext/wiki.en

## Facebook
python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn LSTM  --fasttext ../embeddings/fasttext/wiki.en
 

##### Bi-GRU

## EmoBank
python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn GRU --fasttext ../embeddings/fasttext/wiki.en

## Facebook
python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn GRU  --fasttext ../embeddings/fasttext/wiki.en

##### Bi-LSTM + Attention

## EmoBank
#python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn GRU --attention --fasttext ../embeddings/fasttext/wiki.en

## Facebook
#python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn GRU --attention --fasttext ../embeddings/fasttext/wiki.en

##### Bi-LSTM + Max Pooling

## EmoBank
#python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn GRU --maxpooling --fasttext ../embeddings/fasttext/wiki.en

## Facebook
#python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn GRU --maxpooling --fasttext ../embeddings/fasttext/wiki.en

##### Bi-LSTM + Max Pooling + Attention

## EmoBank
#python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn GRU --attention --maxpooling --fasttext ../embeddings/fasttext/wiki.en

## Facebook
#python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn GRU --attention --maxpooling --fasttext ../embeddings/fasttext/wiki.en

##### Bi-LSTM + Max Pooling + Attention + Initial Training

## EmoBank
python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn LSTM --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --fasttext ../embeddings/fasttext/wiki.en

## Facebook
python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn LSTM --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --fasttext ../embeddings/fasttext/wiki.en

##### Bi-GRU + Max Pooling + Attention + Initial Training

## EmoBank
python3 quotation_classification.py --data ../datasets/opinion_mining/EmoBank/combination.tsv --k 10 --rnn GRU --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --fasttext ../embeddings/fasttext/wiki.en

## Facebook
python3 quotation_classification.py --data ../datasets/opinion_mining/facebook_posts.csv --k 10 --rnn GRU --attention --maxpooling --wordratings ../models/saved_models/wordclassification.h5 --fasttext ../embeddings/fasttext/wiki.en
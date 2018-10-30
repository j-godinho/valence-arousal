# Emotion Classification
Deep learning for the classification of emotions in textual information regarding valence and arousal dimensions.

### Organization
The project is divided into 3 folders:
* datasets (Contains the used datasets for training and evaluation of models)
* models   (Contains all the models for each experiment)
* scripts  (Contains the necessary scripts to build the embeddings vectors)

### Models

* Average Baseline: Sentence score is given by the mean of word classification scores.
* SVM Classification: Support Vector Machine with Regression. Either uses bag-of-words or the average of embeddings vectors.
* Kernel Ridge K-fold: Baseline for the word classification model. Uses Kernel ridge with cross-validation.
* Initial Training K-fold: Word classification model using cross-validation.
* Quotation Classification: Main script to train and evaluate the sentence model.


### Usage

To represent each token, it uses pre-trained GloVe embeddings with Jaro Winkler's distance to calculate out-of-vocabulary tokens. To create these embeddings go to folder scripts/:
```
python3.5 build_glove_embeddings.py --glove ../../embeddings/glove/glove.840B.300d.txt --data ../datasets/facebook_posts.csv
```

To run the main script:

```
python3.5 quotation_classification.py --data ../datasets/facebook_posts.csv --rnn LSTM --attention --emb ../scripts/glove_embeddingsall_sentences.emb.npy --dropout 0.2 --k 10
```

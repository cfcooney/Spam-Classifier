"""
Name: Ciaran Cooney
Date: April 2017
Data preparation and Naive Bayes classifier for text data
for masters project as Dublin City University.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from textblob import blob
import sklearn.feature_extraction.text
from textblob import TextBlob
import nltk
from textblob.utils import strip_punc
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#####Import Data pickle#####
with open('training.pickle', 'r') as f:
     training = pickle.load(f)

labels = ['labels','text']
df = pd.DataFrame.from_records(training, columns=labels)

#####Split data into lemmas and save#####
#split = df.text.apply(split_into_tokens)
new_split = df.text.apply(split_into_lemmas)
with open('new_split.pickle', 'wb') as f:
    pickle.dump(new_split, f)

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(df['text'])
print len(bow_transformer.vocabulary_) #view number of words

text_bow = bow_transformer.transform(df['text'])
print('sparse matrix shape:', text_bow.shape)
print('number of non-zeros:', text_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * text_bow.nnz / (text_bow.shape[0] * text_bow.shape[1])))

text_tfidf = tfidf_transformer.transform(text_bow)
print(text_tfidf.shape)

#####Cassification#####
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

spam_detector = MultinomialNB().fit(text_tfidf, df['labels'])

all_predictions = spam_detector.predict(text_tfidf)
print('accuracy', np.mean(df['labels'] == all_predictions))
print('confusion matrix\n', confusion_matrix(df['labels'], all_predictions))

np.mean(df['labels'] == all_predictions)

#####Save classifier as pickle#####
with open('spam_detector.pickle', 'wb') as f:
    pickle.dump(spam_detector, f)

#####Display Classification Report#####
print(classification_report(df['labels'], all_predictions))
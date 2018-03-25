"""
Name: Ciaran Cooney
Date: April 2017
Data preparation and SVM classifier for text data
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

with open('training.pickle', 'r') as f:
     training = pickle.load(f)
with open('test.pickle', 'r') as f:
     test = pickle.load(f)
        
labels = ['labels','text']
df = pd.DataFrame.from_records(training, columns=labels)
labels = ['labels','text']
df_1 = pd.DataFrame.from_records(test, columns=labels)

df.text.apply(split_into_lemmas)
svm_split = df.text.apply(split_into_lemmas)

svm_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(df['text'])

svm_bow = svm_transformer.transform(df['text'])
print('sparse matrix shape:', svm_bow.shape)
print('number of non-zeros:', svm_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * svm_bow.nnz / (svm_bow.shape[0] * svm_bow.shape[1])))

tfidf_transformer = TfidfTransformer().fit(svm_bow)
text_tfidf = tfidf_transformer.transform(svm_bow)

#####Save tfidf data#####
with open('text_tfidf.pickle', 'wb') as f:
    pickle.dump(text_tfidf, f)

#####Fit a classifier#####
X = text_tfidf
y = df['labels']
clf = svm.SVC(kernel='rbf', C=0.5, gamma=0.1) #adjusted parameter values
clf.fit(X, y)
#####Save the classifier
with open('svm_detector.pickle', 'wb') as f:
    pickle.dump(clf, f)

 all_predictions = clf.predict(test_tfidf)
print('accuracy', np.mean(df_1['labels'] == all_predictions)) #df or df_1 depedning on training or test
print('confusion matrix\n',  confusion_matrix(df_1['labels'], all_predictions)) #all_predictions) df or df_1 depending on training or test
print('(row=expected, col=predicted)')

#####Compute ROC curve for classifier#####
n_classes = df_1.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(df_1['labels'], all_predictions)
    roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(df_1['labels'].ravel(), all_predictions.ravel())

#####Plot ROC curve#####
plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for SVM ')
plt.legend(loc="lower right")
plt.show()

#####Calculate true-positive and false-positive rates#####
CM = confusion_matrix(df_1['labels'], all_predictions)
x1 =float (CM[0][0])
x2 =float (CM[0][1])
y1 =float (CM[1][0])
y2 =float (CM[1][1])
TPR = y2/(y2+y1)
FPR = y1/(y1+y2)
print "TPR = ", TPR
print "FPR = ", FPR
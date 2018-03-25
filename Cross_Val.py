"""
Name: Ciaran Cooney
Date: April 2017
vaildating results from classification
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

#####Load data and construct dataframes#####
with open('test.pickle', 'r') as f:
     test = pickle.load(f)
        
with open('training.pickle', 'r') as f:
     training = pickle.load(f)

labels = ['labels','text']
df = pd.DataFrame.from_records(training, columns=labels)
        
labels = ['labels','text']
df_1 = pd.DataFrame.from_records(test, columns=labels)

df.text.apply(split_into_tokens)
df.text.apply(split_into_lemmas)
new_split = df.text.apply(split_into_lemmas)

#####Transform the data#####
text_bow = bow_transformer.transform(df_1['text'])
print('sparse matrix shape:', text_bow.shape)
print('number of non-zeros:', text_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * text_bow.nnz / (text_bow.shape[0] * text_bow.shape[1])))

tfidf_transformer = TfidfTransformer().fit(text_bow)
text_tfidf = tfidf_transformer.transform(text_bow)

#Save test set in tfidf format for use with SVM
import pickle
with open('test_tfidf.pickle', 'wb') as f:
    pickle.dump(text_tfidf, f)

#####Make predictions#####
all_predictions = spam_detector.predict(text_tfidf)
print('accuracy', np.mean(df_1['labels'] == all_predictions))
print('confusion matrix\n', confusion_matrix(df_1['labels'], all_predictions))
print('(row=expected, col=predicted)')
#code used to compute the ROC curve for the classification model.

n_classes = df_1.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(df_1['labels'], all_predictions)
    roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(df_1['labels'].ravel(), all_predictions.ravel())

plt.figure()
plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()

CM = confusion_matrix(df_1['labels'], all_predictions)
x1 =float (CM[0][0])
x2 =float (CM[0][1])
y1 =float (CM[1][0])
y2 =float (CM[1][1])
TPR = y2/(y2+y1)
FPR = y1/(y1+y2)
print "TPR = ", TPR
print "FPR = ", FPR





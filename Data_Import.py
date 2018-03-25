import sys, os
import numpy as np
import pickle

#####Import Data and Label spam or ham#####
path = ['enron1', 'enron2', 'enron3', 'enron4', 'enron5', 'enron6']

for f in path:
	d = os.listdir( path )
	ham_folder = os.path.join(path, "ham")
	spam_folder = os.path.join(path, "spam")

	for f in os.listdir(ham_folder):
	    full_path = os.path.join(ham_folder, f)
	    text = open(full_path).read()
	    label = 0 #set label for ham
	    emails.append((label,text))
    
	for f in os.listdir(spam_folder):
	    full_path = os.path.join(spam_folder, f)
	    text = open(full_path).read()
	    label = 1 #set label for spam
	    emails.append((label,text))

#####Create training and test sets#####
np.random.shuffle(emails)
training, test = emails[0:21386], emails[21387:30553]

with open('test.pickle', 'wb') as f:
    pickle.dump(test, f)
    
with open('training.pickle', 'wb') as f:
    pickle.dump(training, f)
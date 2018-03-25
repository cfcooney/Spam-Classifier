"""
Name: Ciaran Cooney
Date: April 2017
Visualisation of text data for 
masters project as Dublin City University.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from textblob import blob

#####Load training data pickle#####
with open('training.pickle', 'r') as f:
     training = pickle.load(f)

#####Visualise data with DataFrame#####
labels = ['labels','text']
df = pd.DataFrame.from_records(training, columns=labels)
df['length'] = df['text'].map(lambda text: len(text))
print df.head()
df.length.describe()

df1 = pd.DataFrame.from_records(training, columns=labels) #describe data by category
df1.groupby('labels').describe()

#####Plot the data in various forms#####
df.length.plot(bins=20, kind='hist', range= (0,30000)) #plot by length of emails
plt.show()

df.hist(column='length', by='labels', bins=20, range=(0,30000)) #plot by length of emails according to label
plt.show()

sns.distplot(df1.length, hist=False,) #distribution plot
plt.show()

sns.kdeplot(df.length, bw=0.1)
plt.show()

ax1 =sns.boxplot(
x="labels",
y="length",
data=df)
ax1.set_ylim(0, 4000) #box-plot
plt.show()

ax2 = sns.violinplot(
x="labels",
y="length",
data= df,
order=["0", "1"])
ax2.set_ylim(0, 75000) #violin plot
plt.show()

ax3=sns.stripplot(
x="labels", y="length",
data=df, jitter=True)
ax3.set_ylim(0, 20000) #strip plot
plt.show()

sns.lmplot(
"labels", "length", data=df, fit_reg=False, legend=False)
plt.show()











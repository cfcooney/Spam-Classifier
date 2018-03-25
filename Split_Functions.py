"""
Name: Ciaran Cooney
Date: April 2017
Functions for splitting text data
"""

from textblob import TextBlob

def split_into_tokens(text):
    message = unicode(text, 'utf8', errors='ignore')  # convert bytes into proper unicode
    return TextBlob(message).words

def split_into_lemmas(text):
    message = unicode(text, 'utf8', errors = 'ignore').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


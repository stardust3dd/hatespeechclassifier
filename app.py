import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
import pickle
from flask import Flask, jsonify, render_template, request

stop = stopwords.words('english')
ps = PorterStemmer() # Porter’s Stemmer algorithm

def preprocess(sentence):
  clean_sentence = re.sub(r"\s+", " ", sentence) # to remove long spaces
  clean_sentence = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)", "", clean_sentence) # to remove links that start with HTTP/HTTPS in the tweet
  clean_sentence = re.sub(r"[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)", "", clean_sentence) # to remove other url links
  clean_sentence = re.sub(r"@(\w+)", "", clean_sentence) # to remove mentions
  clean_sentence = re.sub(r"&amp;", "", clean_sentence) # to remove emojis
  clean_sentence = re.sub(r"[^\w\s]","",clean_sentence) # to remove punctuations
  clean_sentence = re.sub(r"\d", "", clean_sentence) # to remove digits
  clean_sentence = clean_sentence.lower() # to lower the characters in a text
  return clean_sentence

def tokenize(tweet):
  tokens = word_tokenize(tweet)
  tokens = [w for w in tokens if not w in stop] 
  tokens = [ps.stem(t) for t in tokens]
  return tokens

app= Flask(__name__)
server= app.server()
model= pickle.load(open('nbclf.pkl', 'rb'))
tfidff= pickle.load(open('tfidf.pkl', 'rb')) 

@app.route('/')
def home():
  return render_template('index.html')
  
@app.route('/classification', methods=['POST'])
def classification():

  tweet = [str(x) for x in request.form.values()]
  print(tweet)
  #tweets.append('rt look like tranny')
  twit_vector= tfidff.transform(tweet)
  cl= model.predict(twit_vector)
  if cl[0]:
    return render_template('index.html', classification_text='Entered tweet is classified as {}: Most likely not hate speech.'.format(cl))
  else:
    return render_template('index.html', classification_text='Entered tweet is classified as {}: Most likely hate speech.'.format(cl))  

if __name__ == "__main__":
  app.run(debug=True)
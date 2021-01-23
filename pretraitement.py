#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara Dunuan, Chinatsu Kuroiwa
Janvier 2021
pretraitement.py
"""
import re
import pandas as pd
from pandas.core.frame import DataFrame
import spacy
nlp = spacy.load('fr_core_news_md')
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

"""
Transformer le fichier TSV en pandas DataFrame
:param tsvfile:
:return df: 
"""
def tsvtopandas(tsvfile):
	df = pd.read_csv(tsvfile, sep='\t')
	return df

"""
Equilibrer les nombres d'opinion pour chaque classe de manière aléatoire
:param corpus,nb_sample:
:return corpus: 
"""
def randomize(corpus,nb_sample):
    corpus_mauvais = corpus[corpus['valeur']== 0 ]
    corpus_random_mauvais =  corpus_mauvais.sample(n=nb_sample, frac=None, replace=False,  weights=None, random_state=None, axis=0)
    corpus_moyen = corpus[corpus['valeur']== 1 ]
    corpus_random_moyen =  corpus_moyen.sample(n=nb_sample, frac=None, replace=False,  weights=None, random_state=None, axis=0)
    corpus_bien = corpus[corpus['valeur']== 2]
    corpus_random_bien =  corpus_bien.sample(n=nb_sample, frac=None, replace=False,  weights=None, random_state=None, axis=0)
    corpus = pd.concat([corpus_random_mauvais,corpus_random_moyen,corpus_random_bien],axis=0)
    return corpus

"""
Normalisation des mots
:param text:
:return text: 
"""
def normalize(text):
	text = text.strip()
	text = text.lower()
	text = re.sub('([^\w\s-])',' \\1 ', text)
	text = re.sub(' \'', '\'', text) # pour aujord'hui
	text = re.sub('\' hui', '\'hui', text) # pour aujord'hui
	text = re.sub('-t-',' ', text) #remplace '-t-' à ' '
	text = re.sub('([\w])-(je|tu|il|elle|nous|vous|ils|elles|le|la|les)','\\1 \\2', text)
	text = re.sub('\' ',' ', text)
	text = re.sub('[,":;!?.\`~@#$%&^*\(\)_+\{\}\[\]]','', text)# supprime les ponctuations
	text = re.sub('\s+', ' ', text)
	return text

"""
Tokenization des mots
:param text:
:return string: 
"""
def tokenize(text):
	tokens = text.split(' ')
	return ' '.join(tokens)

"""
Lemmatization des mots
:param text:
:return string: 
"""
def lemmatize(text):
	doc = nlp(text)
	return ' '.join([token.lemma_ for token in doc])

"""
Supprimmer les stopwords de nltk
:param texte:
:return string: 
"""
def nltkstopwords(texte):
	stopWords = set(stopwords.words('french'))
	tokens = texte.split(' ')
	return ' '.join([token for token in tokens if token not in stopWords])

"""
Supprimmer les stopwords dans list_stopwords.txt
:param texte:
:return string: 
"""
def sans_stopwords(texte):
	stopWords = []
	file = open("list_stopwords.txt", "r")
	for line in file:
		line = line.rstrip()
		stopWords.append(line)
	file.close

	tokens = texte.split(' ')
	return ' '.join([token for token in tokens if token not in stopWords])
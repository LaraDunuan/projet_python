#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara Dunuan, Chinatsu Kuroiwa
Janvier 2021
features.py
"""

import sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import spacy
nlp = spacy.load('fr_core_news_md')

"""
Vectorization avec TfidfVectorizer
:param opinions:
:return X: 
"""
def tfidfvectorize(opinions):
	tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
	X = tfidfconverter.fit_transform(opinions).toarray()
	return X

"""
Vectorization avec CountVectorizer
:param opinions:
:return X: 
"""
def countvectorize(opinions):
	vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
	X = vectorizer.fit_transform(opinions).toarray()
	return X

"""
Vectorization avec nombres de ADJ dans les opinions
:param opinions:
:return nb_adj: 
"""
def count_adj(opinions):
	list_nb_adj = []
	
	for opinion in opinions:
		cnt_adj = 0
		doc = nlp(opinion)
		for token in doc:
			if token.pos_ == 'ADJ':
				cnt_adj += 1
		list_nb_adj.append(cnt_adj)

	# créer l'array numpy
	nb_adj = np.array(list_nb_adj)
	return (nb_adj)
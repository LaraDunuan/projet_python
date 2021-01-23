#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara Dunuan, Chinatsu Kuroiwa
Janvier 2021
classify.py
"""

import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pretraitement
import features
import algorithmes

def main():
	parser = argparse.ArgumentParser(description="Classification des opinions")
	parser.add_argument("-v", "--verbose", help="verbose mode", action="store_true")
	parser.add_argument("file_in", help="le fichier tsv")
	args = parser.parse_args()

	# Transformer le fichier tsv au pandas dataframe
	print("Transformer le fichier tsv au pandas dataframe...")
	corpus = pretraitement.tsvtopandas(args.file_in)
	# Équilibrer les données
	# print("Équilibrer les données avec randomisation...")
	# corpus = pretraitement.randomize(corpus,309)

	'''
	Pré-traitement:
	Ici, on a défini trois façons de pré-traitement
	A : ​Normalisation, tokenization,lemmatisation, suppression des caractères inutiles tel que  des signes de ponctuations
	B : ​Comme A, avec suppression de stopwords NLTK
	C : Comme A, avec suppression de stopwords (list_stopwords.txt)
	'''

	# A : Normalisation, tokenization et lemmatisation
	print("Normalisation, tokenization et lemmatisation...")
	corpus['opinion'] = [pretraitement.normalize(texte) for texte in corpus['texte']]
	corpus['opinion'] = [pretraitement.tokenize(texte) for texte in corpus['opinion']]
	corpus['opinion'] = [pretraitement.lemmatize(texte) for texte in corpus['opinion']]

	# B : Avec suppression des stopwords NLTK
	# print("Suppression des stopwords avec nltkstopwords...")
	# corpus['opinion'] = [pretraitement.nltkstopwords(texte) for texte in corpus['opinion']]

	# C : Avec suppression des stopwords (list_stopwords.txt)
	# print("Suppression des stopwords avec notre list_stopwords.txt...")
	# corpus['opinion'] = [pretraitement.sans_stopwords(texte) for texte in corpus['opinion']]

	# Extraction des valeurs des notes pour chaque opinion
	print("Extraction des valeurs pour y...")
	y = [value for value in corpus['valeur']]

	'''
	Vectorization avec plusieurs features
	Features:
	A : TF-IDF, la base
	B : Bag of Words
	X : 4Features: Longueur des opinions, Nombres des mots, Longueur moyenne des mots, Nombres d'adjectif dans les opinions
	'''

	# A : TF-IDF, la base
	print("Extraction des features (TF-IDF)...")
	X = features.tfidfvectorize(corpus['opinion'])

	# B : Bag of Words
	# print("Extraction des features (Bag of Words)...")
	# X = features.countvectorize(corpus['opinion'])

	# C : Longueur des opinions
	# print("Extraction des features (Longueur des opinions)...")
	# corpus['character_cnt'] = corpus['opinion'].str.len()
	# D : Nombres des mots
	# print("Extraction des features (Nombres des mots)...")
	# corpus['word_cnt'] = corpus['opinion'].str.split().str.len()
	# E : Longueur moyenne des mots
	# print("Extraction des features (Longueur moyenne des mots)...")
	# corpus['characters_per_word'] = corpus['character_cnt']/corpus['word_cnt']
	# F : Nombres d'adjectif dans les opinions
	# print("Extraction des features (Nombres d'adjectif dans les opinions)...")
	# nb_adj= features.count_adj(corpus['opinion'])

	# Combiner les features C, D, E et F
	# print("Combinasion des 4 features...")
	# X = np.column_stack((corpus['character_cnt'].to_numpy(), corpus['word_cnt'].to_numpy(), corpus['characters_per_word'].to_numpy(), nb_adj)).reshape(-1,4)

	'''
	Séparation des données en train et test
	'''
	print("Séparation des données en train et test...")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

	'''
	Avec matplotlib, nous avons les représentations graphiques des données.
	Nous avons observé que la répartition des classes à prédire n'est pas homogène pour les données de test.
	Et nous avons utilisé le paramètre stratify pour essayer de résoudre ce problème.
	plt.hist(y_train, align="right", label="train") 
	plt.hist(y_test, align="left", label="test") 
	plt.title("répartition des classes") 
	plt.show()
	'''

	'''
	Entraînement avec plusieurs algorithmes, hyperparamètres
	A : LinearSVC
	B : GaussianNB
	C : ComplementNB
	'''
	print("Entraînement...")
	clf = LinearSVC()
	# clf = GaussianNB() 
	# clf = ComplementNB()

	algorithmes.train_with_algo(clf, X_train, X_test, y_train, y_test)

	# Cross validation de l'algorithme
	print("Cross validation...")
	print(cross_val_score(clf, X, y)) # uniquement accuracy

	# Optimisation des hyperparamètres
	print("Optimisation des hyperparamètres...")
	param_grid =  {'C': [0.1, 0.5, 1, 10], 'dual' : [True, False]} # LinearSVC
	# param_grid =  {'var_smoothing': [1e-09, 1e-07, 1e-05]} # GaussianNB 
	# param_grid =  {'alpha': [0.5, 1.0, 2.0]} # ComplementNB
	grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
	estimator = grid.fit(X, y)
	df = pd.DataFrame(estimator.cv_results_)
	print(df.sort_values('rank_test_score'))
	df.to_csv(r'./rank_test_score.txt', sep='\t', mode='a')

if __name__ == '__main__':
	main()
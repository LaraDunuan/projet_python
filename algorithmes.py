#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara Dunuan, Chinatsu Kuroiwa
Janvier 2021
algorithmes.py
"""

import sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB

"""
Entrainer avec l'algorithme passé en argument
:param clf, X_train, X_test, y_train, y_test:
:return: 
"""
def train_with_algo(clf, X_train, X_test, y_train, y_test):
	# algorithme = clf
	nom_algo = str(clf)
	# entraînement
	clf = clf.fit(X_train, y_train)
	# prédiction
	y_pred = clf.predict(X_test)
	# évaluation
	clf.score(X_test, y_test)
	# report
	print(f"Classification Report : {nom_algo}")
	printreport(y_test, y_pred)

"""
Afficher le rapport pour l'algorithme
:param y_test, y_pred:
:return: 
"""
def printreport(y_test, y_pred):
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred))
	print(accuracy_score(y_test, y_pred))
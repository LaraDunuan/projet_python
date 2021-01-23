#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara Dunuan, Chinatsu Kuroiwa
Janvier 2021
get_wordfreq.py
"""

import argparse
import pandas as pd
from pandas.core.frame import DataFrame

import pretraitement

def main():
	parser = argparse.ArgumentParser(description="Get word frequencies")
	parser.add_argument("-v", "--verbose", help="verbose mode", action="store_true")
	parser.add_argument("file_in", help="le fichier tsv")
	args = parser.parse_args()

	# transformer le fichier tsv au pandas dataframe
	corpus = pretraitement.tsvtopandas(args.file_in)

	# A : Normalisation, tokenization et lemmatisation
	corpus['opinion'] = [pretraitement.normalize(texte) for texte in corpus['texte']]
	corpus['opinion'] = [pretraitement.tokenize(texte) for texte in corpus['opinion']]
	corpus['opinion'] = [pretraitement.lemmatize(texte) for texte in corpus['opinion']]

	# créer un fichier avec tout les mots et leurs fréquences.
	# on utilise ce fichier pour créer notre liste de stopwords
	output_file = r"word_freq.txt"
	f = open(output_file, "w", encoding='utf-8')

	wordfreq = {}
	N = 0

	for texte in corpus['opinion']:
		tokens = texte.split(' ')
		for word in tokens:
			N += 1
			if wordfreq.get(word) == None:
				wordfreq[word] = 1
			else:
				wordfreq[word] += 1
	
	for word in sorted(wordfreq, key=wordfreq.get, reverse=True):
		output_string =  word + "\t" +  str(wordfreq[word]) + "\n"
		f.write(output_string)	

if __name__ == '__main__':
	main()
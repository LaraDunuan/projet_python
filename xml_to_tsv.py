#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lara Dunuan, Chinatsu Kuroiwa
Décembre 2020
xml_to_tsv.py
"""
import csv
from lxml import etree
import re
import argparse

"""
Transformer le fichier XML en TSV
:param infile(xml):
:return: 
"""
def xml2csv(in_file):
    input_file = in_file
    output_file = r"out_data.tsv"

    # créer CSV
    f = open(output_file, "w", encoding='utf-8')

    # header
    output_string = "doc_id\tvaleur\ttexte\n"
    f.write(output_string)

    # parse XML
    tree =etree.parse(input_file)

    # get root
    root = tree.getroot()

    text_list = []

    documents = tree.xpath('./DOCUMENT')

    if len(documents):
	    for doc in documents:
	        doc_id = str(doc.attrib)
	        doc_id = doc_id[10: -2]
	        note = doc.find("./EVALUATION/NOTE")
	        valeur = str(note.attrib.items()[0])
	        valeur = valeur[12: -2]
	        texte = (doc.find("./TEXTE")).text
	        texte = re.sub('\n+', '', texte)

	        text_list.append(doc_id)
	        text_list.append(valeur)
	        text_list.append(texte)

	        if len(text_list) == 3:
	        	output_string = text_list[0] + "\t" + text_list[1] + "\t" + text_list[2] + "\n"
	        	f.write(output_string)
        		text_list = []
    f.close

def main():
	parser = argparse.ArgumentParser(description="XML to TSV")
	parser.add_argument("-v", "--verbose", help="verbose mode", action="store_true")
	parser.add_argument("file_in", help="le fichier xml")
	args = parser.parse_args()

	xml2csv(args.file_in)
	
if __name__ == '__main__':
	main()
  
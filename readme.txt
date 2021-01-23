Pour notre projet il y a deux fichiers principaux à lancer pour des objectifs différents.  
Vous avez besoin de libraires suivants pour lancer notre scripte :  sklearn, pandas, numpy, matplotlib, etc.

1. xml_to_tsv.py 
	Ce fichier est à l’objectif de transformer les données originales (corpus_aVoiraLire.xml) XML aux données nécessaires TSV (out_data.tsv).
	
		$ python3 xml_to_tsv.py corpus_aVoiraLire.xml
 
 
2. classify.py
	Ce scripte est à l’objectif d’entraîner un modèle de classification sur les critiques de films, livres, spectacles et bandes dessinées. 
	Il contient 3 différents features et 3 types différents d’algorithmes avec leur hyperparamètres.
	Vous pouvez combiner les différents features et les algorithmes comme vous voulez en enlevant ou en mettant en commentaire les lignes correspondants. 
	Par défaut, les features et algorithmes suivants sont fixé. 
	
	Feature : TF-IDF
	Algorithme : LinearSVC() avec son hyperparamètre : {'C': [0.1, 0.5, 1, 10], 'dual' : [True, False]}
	
		$ python3 classify.py out_data.tsv
		* ce scripte va faire prétraitement et classification en même temps. Il donc prend du temps pour afficher les résultats.

Pour les autres scriptes qui sont dans le notre dossier lara_chinatsu , il y a des éxplications de tous les scriptes à la fin de page de notre docmentation. 

 
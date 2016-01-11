event_mining
============

Feature rich event detection in social feeds.

This software performs New Event Detection from stream of public tweets.
It works against a MongoDB tweet database created by the hydra mining engine (look mining_hydra).

Event mining engine consists of 4 main modules:
WD_PROC class handles database interactions, raw tweet processing, information extraction and filtering procedures.
3 clustering models perform geo-based, entity-based and ngram-based clustering.
CLUST_PROC class is conserned with processing clusters obtained from 3 clustering models, handles feature extraction.
Neural_network class is a multilayer perceptron that performs True/False event classification.

Detailed description of the workflow, as well as project presentation is available in report folder.
Example usage and sample output is located in the corresponding files in the root directory.

The system relies on pre-trained tf_idf and word2vec models to evaluate similarity: corresponding files should be put in the assets folder.
Since the word embeddings file is too large for github (420 mb), I made it available in my dropbox folder: 
https://www.dropbox.com/sh/oydaw851u1yadki/AAAo8H2pq4P5RKWu0UFIa1hma?dl=0

The assets folder also contains true and false event datasets I hand-labeled for learning the model. 
The datasets are not too big, but I encourage everyone interested in this topic to use them freely.
I can also provide a sample of my 10-million tweet database (2GB) or the whole dump (40GB) on request.

Complete source code is availible in tweet_proc.py.
A more organized version of the same code, together with example usage and output is present in Event_mining.ipynb.
It also includes the code I used to hand-label clusters and learn the classification model.
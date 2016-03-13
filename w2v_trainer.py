# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
from datetime import datetime
from utils import progress
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, filename, logging=False):
        self.filename = filename
	self.logging = logging
        self.count = 0
    def __iter__(self):
        for line in open(self.filename, 'r'):
            if self.logging:
                progress(self.count, 0, skip=1000, mode = 2)
            self.count += 1
            yield unicode(line).split()
    def count_samples(self):
        cnt = 0
        for line in open(self.filename, 'r'):
            cnt += 1
        return cnt

load = 1
epochs = 15
workers = 8
samples = None
model_filename = 'w2v_model_v1'

if not load:
	model = Word2Vec(size = 256, min_count=10, workers=workers, sg=1, hs=1)
	ms = MySentences('/ssd/datasets/tw_ht_corpus.txt')
	model.build_vocab(ms)
else:
	model = Word2Vec.load(model_filename)

print '\nVocab size:', len(model.vocab), '\n'
model.workers = workers

for e in range(epochs):
    print '----------***---------- Epoch', e, '----------***----------'
	   2016-03-13 01:55:56,953    
    generator = MySentences('/ssd/datasets/tw_ht_corpus.txt')
    if not samples:
	samples = generator.count_samples()
    
    st = datetime.now()
    model.train(generator, total_examples=samples)
    en = datetime.now()
    model.save(model_filename)
    print 'epoch finished in', str(en - st)
    print 'model persisted to', model_filename 

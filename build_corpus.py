import re
from utils import progress
from processor import MDB, text_process

f = open('tw_ht_corpus.txt', 'a')
p = MDB('tweets', 1, port = 27017)
cols = p.client['tweets'].collection_names()
cols.remove('SPB')
cols.remove('Moscow')

i = 0
container = []
b_size = 1000

counts = []
for c in cols:
    ml = p.client['tweets'][c].find()
    counts.append(ml.count())
    
total = sum(counts)
print 'total:', total, 'documents'

for c in cols:
    ml = p.client['tweets'][c].find()
    for t in ml:
        try:
            dt = text_process(t)
            progress(i, total)
            if dt:
                f.write(dt + '\n')
        except Exception as e:
            print e
        finally:
            i += 1

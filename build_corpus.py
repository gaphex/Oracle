import re
from utils import progress
from text_processor import MDB, text_process

f = open('assets/tw_ht_corpus_2.txt', 'a')
p = MDB('tweets')
cols = p.client['tweets'].collection_names()
cols.remove('SPB')
cols.remove('EKB')
cols.remove('Moscow')
print cols
i = 0

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
            dt = text_process(t)[0]
            progress(i, total)
            if dt:
                f.write(dt + '\n')
        except Exception as e:
            print e
        finally:
            i += 1

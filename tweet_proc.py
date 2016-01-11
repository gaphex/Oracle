#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages")


import re
import sys
import time
import nltk
import math
import pickle
import string
import random
import _sqlite3
import operator
import itertools
import datetime
import numpy as np
import collections
import fastcluster
import CMUTweetTagger
from emoji import emoji
import matplotlib.pyplot as plt
from pymongo import MongoClient
from operator import itemgetter
from collections import Counter
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
from stopwords import stopwords
from IPython.display import clear_output
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def remove_punctuation(s):
    return s.translate(None, string.punctuation)

def load_stop_words():
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(stopwords)
    return set(stop_words)

def extract_hashtags(dset):
    htags = []
    for d in dset:
        ht = d['payload']['entities']['hashtags']
        if len(ht):
            for i in ht:
                htags.append(i['text'])
    return htags

def tovoc(cp, sent):
    vocvec = []
    for word in sent:
        try:
            i = cp.w2v_dictionary[word]
            vocvec.append(i)
        except:
            pass
    return vocvec

def sigmoid(t):
    return 1/(1+exp(-t))

def list_intersection(l1, l2):
    return set(l1)&set(l2)

def sent2vec(cp, sent):
    M = [cp.w2v_embeddings[v] for v in tovoc(cp, sent)]
    try:
        r = np.sum(M, axis = 0)
        return r
    except:
        print 'OOV'
        return 0
        
def cossim(s1, s2):
    v1 = sent2vec(s1)
    v2 = sent2vec(s2)
    if not v1.shape or not v2.shape:
        return [[0]]
    return cosine_similarity(v1, v2)

def progress(i, n, skip = 100, mode = 1):
    if i%skip == 0 and mode == 1:
        sys.stdout.write("\r%s%%" % "{:5.2f}".format(100*i/float(n)))
        sys.stdout.flush()
        if i >= (n/skip - 1)*skip:
            sys.stdout.write("\r100.00%")
            print("\r")
    if i%skip == 0 and mode == 2:
        sys.stdout.write("\r%s" % str(i))
        sys.stdout.flush()
        

def db_count(con):
    curs = con.cursor()
    table = 'tweets'
    rowsQuery = "SELECT Count() FROM %s" % table
    curs.execute(rowsQuery)
    numberOfRows = curs.fetchone()[0]
    return numberOfRows

def locate(obj):
    if obj['coordinates']:
        return 1, obj['coordinates']['coordinates']
    else:
        box = obj['place']['bounding_box']['coordinates']
        m1, m2 = list((np.sum(box, axis = 1)/4)[0])
        s1, s2 = list((np.array(box[0])[2] - np.array(box[0])[0])/2)
        #return 0, [random.gauss(m1, s1), random.gauss(m2,s2)]
        return 0, [m1, m2]

def resolveEmoji(myText):
    emostr = []
    emo_db = emoji
    b = myText.encode('unicode_escape').split('\\')
    c = [point.replace('000','+').upper() for point in b if len(point) > 8 and point[0] == 'U']
    emj = [(emo_db[emo[:7]]) for emo in c if emo[:7] in emo_db]
    return emj

def extract_links(myText, tokens=False):
    links = []
    rex = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    for rf in re.findall(rex, myText):
        for t in rf:
            if len(t)>2:
                links.append(t)
    if tokens:
        c = '+LINK'
    else:
        c = ' '
    return ((re.sub(rex, c, myText), links))

def w2v_transform(cp, ls):
    M = []
    for i in ls:
        try:
            t = sent2vec(cp, i)
            if t.shape[0] == 512:
                M.append(t)
        except:
            pass
    M = np.array(M)
    return M

def get_last(collection, hours=0, count =0):
    tshift = 5
    cur = ms.db[collection].find().sort("_id", -1)
    if hours:
        time_now = time.mktime(time.localtime())
        for i, doc in enumerate(cur):
            if i%1000 == 0:
                if i==-1:
                    time_now = time.mktime(time.strptime(doc['payload']['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                created_at = time.mktime(time.strptime(doc['payload']['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                dif = tshift*3600 + time_now - created_at
                #print dif - hours*3600
                if dif > hours*3600:
                    break
        print 'retrieved', i, collection, 'tweets', 'posted in', hours, 'hours'
        return list(ms.db[collection].find().sort("_id", -1).limit(i))
    if count:
        print 'retrieved', count, 'tweets in', collection
        return list(ms.db[collection].find().sort("_id", -1).limit(count))
    
def filter_noisy(ms, dev, w=3, m=3, h=4):
    cprc = {}
    cdev = []
    stophash = ['job', 'hiring', 'Job', 'Hiring']
    for i, d in  enumerate(dev):
        prc = ms.process(d)
        if len(prc['words'])>w and len(prc['mentions'])<m and len(prc['hashtags'])<h and prc['lang'] == u'en':
            if not len(list_intersection(prc['hashtags'], stophash)):
                cprc[len(cdev)] = prc
                cdev.append(d)
    print 'filtered', len(dev)-len(cdev), 'documents,', len(cdev), 'left'
    return cdev, cprc

def filter_rare(X, th=3):
    mapping = {}
    Xclean = np.zeros((1, X.shape[1]))
    for i in range(X.shape[0]):
        if X[i].sum() > th:
            Xclean = np.vstack([Xclean, X[i].toarray()])
            mapping[Xclean.shape[0] - 2] = i
        progress(i, X.shape[0])

    Xclean = Xclean[1:,]
    return Xclean, mapping

def filter_coords(cdev, cdict):
    cr = []
    cd = {}
    mapping = {}
    for i, c in enumerate(cdev):
        if c['payload']['coordinates']:
            cd[len(cr)] = cdict[i]
            mapping[len(cr)] = i
            cr.append(c)
    print 'filtered', len(cdev)-len(cr), 'documents,', len(cr), 'left'
    return cr, cd, mapping

def organize_clusters(fcl, top_n = 500, th = 4):
    dc = {}
    top_clusters = []
    for i in range(max(fcl)+1):
        dc[i] = []
    for i, f in enumerate(fcl):
        dc[f].append(i)

    dd = [(d[0], len(dc[d[1]])) for d in enumerate(dc)]
    d1 = sorted(dd, key=itemgetter(1), reverse = True)
    for d in d1[0:top_n]:
        if d[1] > th:
            top_clusters.append(dc[d[0]])
        else:
            break
    return top_clusters

def process_entities(cdev, cprc):
    ent_corpus = []
    hdev = []
    hprc = {}
    mapping = {}
    er = []

    for i in cprc:
        entities = cprc[i]['hashtags'] + cprc[i]['checkins']
        if entities:
            ent_corpus += entities
            if list_intersection(entities, cprc[i]['words']):
                hprc[len(hdev)] = cprc[i]
                mapping[len(hdev)] = i
                hdev.append(' '.join(entities))
            else:
                er.append(i)
    return hdev, hprc, mapping, ent_corpus, er

def boost_entities(features):
    boost_entity = {}
    pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term in features])

    for line in pos_tokens:
        term =''
        for entity in range(len(line)):
            term += line[entity][0].lower() + " "
            if "^" in str(line):
                boost_entity[term.strip()] = 2.5
            else:
                boost_entity[term.strip()] = 1.0
    return boost_entity

def build_voc(corp, thr):
    cnt = collections.Counter(corp).most_common()
    voc = []
    for i, c in enumerate(cnt):
        if c[1]<thr:
            break
    for c in cnt[0:i]:
        voc.append(c[0])
    return voc

def connect_to_mongo(port, host = 'localhost'):
        client = MongoClient(host, port)
        if str(client).split('=')[-1][:-1] == 'True':
            print 'connected'
            return client
        else:
            print 'not connected'
            return 0
        
def retrieve_from_timewindow(ms, col, startEpoch, endEpoch):
    print 'querying..'
    m = ms.db[col].find({"payload.created_at":{'$gte':startEpoch, '$lte':endEpoch}})
    dev = []
    print 'retrieving..'
    for i, obj in enumerate(m):
        progress(i, 1, mode = 2)
        dev.append(obj)
    print ''
    print 'retrieved', len(dev), 'documents from window'
    return dev

def DB_SCAN(GM, th):

    db = DBSCAN(eps=th, min_samples=3).fit(GM)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    gd = {}
    
    for i in range(0, max(labels)+1):
        gd[i] = []

    _ = [gd[label].append(i) for i, label in enumerate(labels) if label!=-1]
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    return gd

def run_geoclust_model(cdev, cprc, th = 0.03):
    print '____________________________________________________'
    print 'running geo model'
    gdev, gprc, gmapping = filter_coords(cdev, cprc)
    GM = []
    for i in gprc:
        GM.append(gprc[i]['crds'])
    GM = np.array(GM)
    X = StandardScaler().fit_transform(GM)

    gc = DB_SCAN(X, th)
    gcf = []
    for g in gc:
        gcf.append([gmapping[t] for t in gc[g]])
        
    print 'detected', len(gcf), 'geo clusters' 
    return gcf

def run_entity_model(cdev, cprc):
    print '____________________________________________________'
    print 'running entity model'
    hdev, hprc, hmapping, entcorp, er = process_entities(cdev, cprc)
    print 'removed', len(cdev)- len(hdev), 'documents', len(hdev), 'left'
    voc = build_voc(entcorp, 2)
    
    ent_vectorizer = CountVectorizer(vocabulary = voc)
    E = ent_vectorizer.fit_transform(hdev)
    
    Eclean, emapping = filter_rare(E, 0)

    E_dense = np.matrix(Eclean).astype('float')
    E_scaled = preprocessing.scale(E_dense)
    E_normalized = preprocessing.normalize(E_scaled, norm='l2')
    
    EMatrix = pairwise_distances(E_normalized, metric='cosine')
    EL = fastcluster.linkage(EMatrix, method='average')
    flat_eclust = hierarchy.fcluster(EL, 0.5, 'distance')
    ec = organize_clusters(flat_eclust, th = 3)
    
    ecf = []
    for cl in ec:
        ecf.append([hmapping[emapping[t]] for t in cl])
    print 'detected', len(ecf), 'entity clusters'      
    return ecf, voc

def run_ngram_model(cdev, cprc):
    print '____________________________________________________'
    print 'running n-gram model'
    wcorp = []
    for i in cprc:
        wcorp.append(' '.join(cprc[i]['words']))
        
    vectorizer = CountVectorizer(analyzer='word', binary=True, min_df=max(int(len(wcorp)*0.0005), 5), ngram_range=(2,3))
    X = vectorizer.fit_transform(wcorp)
    Xclean, mapping = filter_rare(X)
    
    Xdense = np.matrix(Xclean).astype('float')
    X_scaled = preprocessing.scale(Xdense)
    X_normalized = preprocessing.normalize(X_scaled, norm='l2')
    
    textMatrix = pairwise_distances(X_normalized, metric='cosine')
    L = fastcluster.linkage(textMatrix, method='average')
    flat_textclust = hierarchy.fcluster(L, 0.5, 'distance')
    ttc = organize_clusters(flat_textclust)
    
    ncf = []
    for cl in ttc:
        ncf.append([mapping[t] for t in cl])
    print 'detected', len(ncf), 'n-gram clusters'     
    return ncf


class CLST_PROC(object):
    def __init__(self, model, pop_ent):
        self.perceptron = model
        self.voc = pop_ent
	
    def load_lang_models(self):
        self.tfidf_vectorizer = pickle.load( open( 'assets/tfidf_vectorizer', "rb" ) )
        self.w2v_dictionary = pickle.load( open( 'assets/w2v_dictionary', "rb" ) )
        self.w2v_embeddings = pickle.load( open( 'assets/w2v_embeddings', "rb" ) )
        
    def process_cluster(self, dev, prc):
        size = len(dev)
        rawcorp = []
        procorp = []
        users = []
        hashcorp = []
        times = []
        urls = []
        mentions = []
        retweets = []
        followers = []
        friends = []
        status_cnt = []
        crds = []
        bagowords = []
        prowcorp = []
        checkins = []

        for i in range(size):
            pl = dev[i]['payload']
            checkins += prc[i]['checkins']
            bagowords += prc[i]['words']
            prowcorp.append(prc[i]['words'])
            procorp.append(' '.join(prc[i]['words']))
            times.append(prc[i]['timestamp'])
            [hashcorp.append(hs) for hs in prc[i]['hashtags']]
            [urls.append(url) for url in prc[i]['urls']]
            [mentions.append(ment) for ment in prc[i]['mentions']]

            rawcorp.append(pl['text'])
            users.append(pl['user']['id'])
            retweets.append(pl['retweet_count'])
            followers.append(pl['user']['followers_count'])
            friends.append(pl['user']['friends_count'])
            status_cnt.append(pl['user']['statuses_count'])
            if pl['geo']:
                crds.append(pl['geo']['coordinates'])

        #pos_tokens = CMUTweetTagger.runtagger_parse(procorp)
        #prop_nouns = float(str(pos_tokens).count('^'))/size
        prop_nouns = 0

        mtfidf_sim = np.mean(pdist(self.tfidf_vectorizer.transform(procorp).toarray(), lambda u, v: cosine_similarity(u,v)))
        mw2v_sim = np.mean(pdist(w2v_transform(self, prowcorp), lambda u,v: cosine_similarity(u,v)))

        pop_entities = float(len(list_intersection(hashcorp, self.voc)))
        unq_unigrams = float(len(set(bagowords)))/size
        htags = float(len(hashcorp))/size
        unq_htags = len(set(hashcorp))
        unq_checkins = len(set(checkins))
        checkins = float(len(checkins))/size

        cs = np.array(crds)
        try:
            bboxsquare = (max(cs[:,0])-min(cs[:,0]))*(max(cs[:,1])-min(cs[:,1]))
        except:
            bboxsquare = 0

        unq_users = float(len(set(users)))/size
        meanfollowers = np.mean(followers)
        meanfriends = np.mean(friends)
        retweets = sum(retweets)
        mentions = float(len(mentions))/size

        timeframe = float(max(times)-min(times))/60
        inst_urls = float(str(urls).count('instagram'))/size
        urls = float(len(urls))/size

        features = {'size':size, 'prop_nouns':prop_nouns, 'pop_entities':pop_entities, 'unq_unigrams':unq_unigrams,
                    'hashtags':htags, 'unq_users':unq_users, 'unq_hashtags':unq_htags, 'bbox':bboxsquare, 
                    'timeframe':timeframe, 'mentions':mentions, 'mfollowers':meanfollowers, 'mfriends':meanfriends, 
                    'retweets':retweets, 'urls':urls, 'inst_urls':inst_urls,'mtfidf_sim':mtfidf_sim, 
                    'mw2v_sim':mw2v_sim, 'unq_checkins': unq_checkins, 'checkins': checkins}

        return features

def predict_select(model, MX, cluster):
    CL = []
    if MX.shape[0] == len(cluster):
        for i in range(MX.shape[0]):
            prediction = np.argmax(model.predict(MX[i]))
            if prediction == 1:
                CL.append(cluster[i])
    else:
        print 'unalligned matrices'
    return CL

def vectorize_clusters(cdev, cprc, clusters, cp, preprocess = True):
    MX = []
    maxth = 500

    for i, cluster in enumerate(clusters):
        size = len(clusters)
        progress(i, size)
        if len(cluster) < maxth:
            d, p = materialize(cdev, cprc, cluster)
            prc = cp.process_cluster(d, p).values()
            MX.append(prc)

    MX = np.array(MX)
    if preprocess == True:
        MX = preprocessing.scale(MX)
        MX = preprocessing.normalize(MX, norm='l2')
    return MX

def summarize(event):
    print '-----#-----#-----#-----#-----#-----#-----#-----#-----'
    bagowords = []
    bagohashs = []
    bagocheckins = []
    instlinks = []
    crds = []
    times = []
    for p in event[1]:
        bagowords += p['words']
        times.append(p['timestamp'])
        bagohashs += p['hashtags']
        bagocheckins += p['checkins']
        [instlinks.append(l) for l in p['urls'] if 'instagram' in l]
        if p['crds']:
            crds.append(p['crds'])
    print 'time:', datetime.datetime.fromtimestamp(np.mean(times))
    if len(crds):
        print 'location:', sum(np.array(crds))/len(crds)
    print 'popular hashtags:', collections.Counter(bagowords+bagohashs).most_common(5)
    print 'popular checkins:', collections.Counter(bagocheckins).most_common(3)
    for l in instlinks:
        print l

def materialize(cdev, cprc, cluster):
    tdev = []
    tprc = []
    for point in cluster:
        tdev.append(cdev[point])
        tprc.append(cprc[point])
    return tdev, tprc        
        

class WD_PROC_7000(object):
    def __init__(self, db, mode, port = None):
        
        self.client = connect_to_mongo(port)
        self.db = self.client[db]
        self.s_words = load_stop_words()
        self.mode = mode
        self.bag = []
        
    def db_count(self):
        return [{col: self.db[col].count()} for col in self.db.collection_names()]
    
    def sum_db_count(self):
        return sum([t.values()[0] for t in self.db_count()])
        
    def __iter__(self):

        g = self.db.collection_names()
        for col in g[0:-1]:
            for obj in self.db[col].find({}):
                #yield self.process(obj)
                yield(obj)
                
    def tokenise(self, myText, tokens=False, usestop = True):
        words, htags, ments = [], [], []
        for word in re.findall(r'(?u)[@|#]?\w+', myText):
            if word in ['+LINK', '+HASH', '+MENT']:
                pass
            else:
                word = word.lower()
            if usestop:
                if word not in self.s_words:    
                    words.append(word)
            else:
                words.append(word)

        for i, word in enumerate(words):
            if word.startswith('#'):
                words[i] = word.split('#')[1]
                htags.append(word)
                if tokens:
                    words.insert(i, '+HASH')

            if word.startswith('@'):
                ments.append(word)
                words[i] = ''
                if tokens:
                    words[i] = '+MENT' 

        return [word for word in words if len(word)]

        
    def process(self, obj):
        if isinstance(obj, dict):
            try:
                id = obj['_id']
                payload = obj['payload']
                raw_text = payload['text']

                q, crds = locate(payload)
                tstmp = time.strptime(payload['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                
                ent = payload['entities']
                urls, media_urls, hashtags, mentions = [],[],[],[]
                if ent:
                    for et in ent.keys():
                        
                        if len(ent[et]) and et == 'urls':
                            urls = [i['expanded_url'] for i in ent['urls']]
                        if len(ent[et]) and et == 'hashtags':
                            hashtags = [i['text'] for i in ent['hashtags']]
                        if len(ent[et]) and et == 'user_mentions':
                            mentions = [i['screen_name'] for i in ent['user_mentions']]
                        if len(ent[et]) and et == 'media':
                            media_urls = [i['expanded_url'] for i in ent['media']]
 

                text, links = extract_links(raw_text)
                words = self.tokenise(text)
            
                checkins = []
                bag = text.split(' ')
                if '@' in bag:
                    ind = bag.index('@')
                    for t in range(ind, len(bag)):
                        if len(bag[t]):
                            if bag[t][0].isupper():
                                plo = remove_punctuation(bag[t].encode('ascii','ignore').lower())
                                plo = unicode(plo)
                                checkins.append(plo)

                #emoji = resolveEmoji(text)
                lang = payload['lang']
                tstamp = time.mktime(tstmp)
                        
                return {'id':int(id), 'words':words, 'hashtags':hashtags, 'checkins':checkins, 'mentions':mentions, 'ctype':q, 'crds':crds, 'timestamp':tstamp, 'lang':lang, 'urls':urls+media_urls}

            except Exception as e:
                print 'could not unpack object', obj['_id'], e
                return None
            
    def add_to_bag(self, obj):
        if isinstance(obj, dict):
            try:
                payload = obj['payload']
                raw_text = payload['text']
                
                text, links = extract_links(raw_text)
                words, _, _ = self.tokenise(text)

                self.bag += words
            except:
                pass


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, weights = [], activation='logistic'):

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = weights
        if not weights:
            for i in range(1, len(layers) - 1):
                self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

    def load_model(self):
        self.layers = pickle.load( open( 'assets/layers', "rb" ) )
        self.weights = pickle.load( open( 'assets/weights', "rb" ) )
        
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for layer in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[layer].T)*self.activation_deriv(a[layer]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
                
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
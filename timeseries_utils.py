from text_processor import MDB, text_process
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta
from stopwords import stopwords
from utils import *

import re
import math
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_tdelta(d, h):
    return (datetime.now()-timedelta(days=d, hours=h))

def process_batch(cur, geo=False, fsw=False, stem=False):
    st = datetime.now()
    i = 0
    r = []
    l = cur.count()
    for doc in cur:
        if i == 0:
            stt = doc['created']
        t = text_process(doc, geo=geo, filter_sw=fsw, stem=stem)
        r.append({'words': t[0].split(), 'created_at': doc['created'], 'geo': t[1]})
        i += 1
        progress(i, l, skip=100)
    end = doc['created']
    print '\nretrieval and processing took', datetime.now() - st
    return r, stt, end
        
def tstamp_to_datetime(ts):
    return pd.to_datetime(ts)

def find_window(t, start, window):
    w = math.floor((float((t-start).total_seconds()))/window.total_seconds())
    if w < 0: w = 0.0
    return int(w)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def hasBS(inputString):
    if "_" in inputString:
        return 1
    else:
        return 0
    
def plot(ts, t = 3):
    windows = np.array(range(ts.shape[0]))
    plt.plot(windows, ts, 'b-', label='label here')
    plt.plot(windows, t * movingaverage(ts, 3), 'r-', label='label here')
    plt.show()
    
def build_timeseries(dev, st, en, mins, filtration=0, logging=True):
    if logging: print 'processing', len(dev), 'documents'
    
    if filtration == 0:
        mindf = 1
        maxdf = 1.0
        f_th = 0
        l_th = 0
        s_words = []
    
    elif filtration == 1:
        mindf = 1
        maxdf = 1.0
        f_th = 0
        l_th = 0
        s_words = swords
        
    elif filtration == 2:
        mindf = 3
        maxdf = 1.0
        f_th = 3
        l_th = 1
        s_words = swords
        
    else:
        return 0

    t_start = st
    t_end = en
    t_span = (t_end - t_start)
    resolution = int(math.floor(t_span.total_seconds()/(60*mins))) + 1
    t_window = t_span/resolution

    if logging: print 'window size =', mins, 'minutes'

    w_corp = [' '.join(m['words']) for m in dev]
    CV = CountVectorizer(min_df=mindf, max_df=maxdf, token_pattern=tok_regex, stop_words=s_words)
    if not len(w_corp):
        return None
    
    try:
        wtr = CV.fit_transform(w_corp)
        voc = CV.get_feature_names()
    except:
        return None

    if logging: print 'vocab size =', len(voc) 

    wn = {}

    for w in range(resolution + 3):
        wn[w] = {}
        wn[w]['counts'] = np.zeros((1, len(voc)))
        wn[w]['timestamp'] = tstamp_to_datetime(t_start + t_window * w)

    l = len(dev)
    for i, d in enumerate(dev):
        w = find_window(d['created_at'], t_start, t_window)
        wn[w]['counts'] += wtr[i].toarray()

    M = []
    T = []
    for n in wn:
        M.append(wn[n]['counts'][0])
        T.append(wn[n]['timestamp'])

    M = np.array(M).T
    if logging: print 'found', M.shape[1], 'windows' 
    
    P = []
    ind = []
    for term in range(M.shape[0]):
        if M[term].max() > f_th and len(voc[term]) > l_th and not hasNumbers(voc[term]) and not hasBS(voc[term]):
            ind.append(term)
            P.append(M[term])

    P = np.array(P)
    
    vdict = {}
    for i in ind:
        vdict[voc[i]] = i
    
    return P, vdict


def assign_cell(point, bbox, resolution = 10):
    
    spanx = bbox[2] - bbox[0]
    spany = bbox[3] - bbox[1]

    dx = spanx/resolution
    dy = spany/resolution
     
    xr = point['geo'][1][0]
    yr = point['geo'][1][1]
    cx = int(math.floor((xr - bbox[0])/dx))
    cy = int(math.floor((yr - bbox[1])/dy))

    if cx < 0 or cy <0 or cx > resolution - 1 or cy > resolution - 1:
        return None
    else:
        return cx, cy
    
def cell_matrix(res):
    MTR = []
    for i in range(res):
        MTR.append([])
        for j in range(res):
            MTR[i].append({'objects':[], 'series':[], 'voc':[]})
    return MTR

def list_intersection(a, b):
    return list(set(a) & set(b))

def load_stop_words():
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(stopwords)
    return set(stop_words)

tok_regex = re.compile(r'(?u)[@|#]?\w+')
swords = load_stop_words()

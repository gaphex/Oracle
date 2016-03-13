# -*- coding: utf-8 -*-
from pymongo import MongoClient
from stopwords import stopwords
import concurrent.futures
import numpy as np
import string
import nltk
import time
import re
    
class MDB(object):
    def __init__(self, db, mode, port = None):

        self.client = self.connect_to_mongo(port)
        self.db = self.client[db]
        self.mode = mode
        self.bag = []

    def connect_to_mongo(self, port, host = 'localhost'):

        client = MongoClient(host, port, serverSelectionTimeoutMS = 5)
        try:
                dbn = client.database_names()
                print 'connected to database'
                return client
        except:
                print 'could not establish a connection to database'
                return 0

    def db_count(self):
        return [{col: self.db[col].count()} for col in self.db.collection_names()]

    def sum_db_count(self):
        return sum([t.values()[0] for t in self.db_count()])

    def __iter__(self):

        g = self.db.collection_names()
        for col in g[0:-1]:
            for obj in self.db[col].find({}):
                yield(obj)

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
            
def load_stop_words():
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(stopwords)
    return set(stop_words)
            
def remove_punctuation(s):
    return punkt_regex.sub('', s)
            
def extract_links(myText, tokens=False):
    links = []
    u = re.findall(url_regex, myText)
    links = [link for link in u if len(link) > 3]
    if tokens:
        c = '+LINK'
    else:
        c = ' '
    return ((re.sub(url_regex, c, myText), links))           
            
def locate(obj):
    if obj['coordinates']:
        return 1, obj['coordinates']['coordinates']
    else:
        box = obj['place']['bounding_box']['coordinates']
        m1, m2 = list((np.sum(box, axis = 1)/4)[0])
        s1, s2 = list((np.array(box[0])[2] - np.array(box[0])[0])/2)
        #return 0, [random.gauss(m1, s1), random.gauss(m2,s2)]
        return 0, [m1, m2]
    
def tokenise(myText, tokens=False, stop_words = None):
    words, htags, ments = [], [], []
    for word in re.findall(r'(?u)[@|#]?\w+', myText):
        if word in ['+LINK', '+HASH', '+MENT']:
            pass
        else:
            word = word.lower()
        if stop_words:
            if word not in stop_words:
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

def text_process(obj, verbose = False):
    try:
        raw_text = obj['payload']['text']
        if verbose:
            print raw_text
        text, _ = extract_links(raw_text)
        cleantext = text.encode('ascii','ignore')
        words = []
        ws = re.findall(r'(?u)[@|#]?\w+', cleantext)
        if isinstance(ws, list):
            for word in ws:
                if not hasNumbers(word) and not hasBS(word):
                    words.append(word.lower().strip())
        else:
            return None
        
        htags = {}
        for i, word in enumerate(words):
            if word.startswith('#'):
                words[i] = '' #word.split('#')[1]
                htags[i] = word
                
            if word.startswith('@'):
                words[i] = ''
                    
        string = remove_punctuation(' '.join(words))
        wbag = string.split()
        
        if htags:
            for v in htags.keys():
                wbag.insert(v, htags[v])
            
        string = ' '.join(wbag)
        if string and not string.isspace():
            return string.strip()
        else:
            return None
    except Exception as e:
        print e
        return None
    
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def hasBS(inputString):
    if "_" in inputString:
        return 1
    else:
        return 0


def process(obj):
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
            words = tokenise(text, tokens = False, stop_words = swords)

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

            return {'id':int(id), 'words':words, 'hashtags':hashtags, 'checkins':checkins, 'mentions':mentions, \
                    'ctype':q, 'crds':crds, 'timestamp':tstamp, 'lang':lang, 'urls':urls+media_urls}

        except Exception as e:
            print 'could not process object', obj['_id'], e
            return None
    else:
        return None

def batch_process(workload):
    prc = [process(d) for d in workload]
    return prc 

punkt_regex = re.compile('[%s]' % re.escape(string.punctuation))
url_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
swords = load_stop_words()


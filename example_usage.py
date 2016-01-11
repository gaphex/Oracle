from tweet_proc import *

msc = WD_PROC_7000('tweets', 1, port = 27000)

startEpoch = 'Sat Dec 12 14:00:00 +0000 2015'
endEpoch = 'Sat Dec 12 23:00:00 +0000 2015'

dev = retrieve_from_timewindow(msc, 'Boston', startEpoch, endEpoch)
cdev, cprc = filter_noisy(msc, dev, w = 3, h = 6)

geoclusters = run_geoclust_model(cdev, cprc)
entityclusters, voc = run_entity_model(cdev, cprc)
ngramclusters = run_ngram_model(cdev, cprc)

model = NeuralNetwork([], 'logistic')
model.load_model()

ccp = CLST_PROC(model, voc)
ccp.load_lang_models()

GX = vectorize_clusters(cdev, cprc, geoclusters, ccp)
EX = vectorize_clusters(cdev, cprc, entityclusters, ccp)
NX = vectorize_clusters(cdev, cprc, ngramclusters, ccp)

fGX = predict_select(model, GX, geoclusters)
fEX = predict_select(model, EX, entityclusters)
fNX = predict_select(model, NX, ngramclusters)

for i in fGX:
    t = materialize(cdev, cprc, i)
    summarize(t)
    
for i in fEX:
    t = materialize(cdev, cprc, i)
    summarize(t)
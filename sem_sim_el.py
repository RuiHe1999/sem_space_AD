import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm 

import spacy
nlp = spacy.load("el_core_news_sm")

from nltk.corpus import stopwords
from gensim.models import fasttext

# 2. consonants
# stopwords
stpw = stopwords.words('greek')

# fasttext model 
# ft = fasttext.load_facebook_vectors('E:/A-Horace/PhD/FastText/cc.en.300.bin')
ft = fasttext.load_facebook_vectors('cc.el.300.bin')

# demographics
demo = pd.read_csv('Greek_demo.csv')
pars = demo.PAR.unique().tolist()
par2group = dict(zip(demo.PAR, demo.dx))

# graph features
feats = ['FT_WordNum', 'FT_TTR', 'FT_GSim', 'FT_LSim']

# 4. commands
# read transcripts
par2sim = {}
for par in tqdm(pars):
    text = next(open(f'Transcripts/Greek/{par2group[par]}/{par}.txt', 'r', encoding='utf-8'))
    
    # process text
    doc = nlp(text)
    tokens = [token.text for token in doc 
              if ((token.text.lower() not in stpw) and 
                  (token.pos_ not in ['SPACE', 'PUNCT']))]
    
    embeds = np.array([ft[token] for token in tokens])
    
    # all-pair similarity 
    sem_sim = 1 - distance.cdist(embeds, embeds, metric='cosine')
    global_sim = np.nanmean(sem_sim[np.tril_indices(sem_sim.shape[0], k=-1)])
    consec_sim = np.nanmean(np.diagonal(sem_sim, offset=1))
    
    global_var = np.nanvar(sem_sim[np.tril_indices(sem_sim.shape[0], k=-1)])
    consec_var = np.nanvar(np.diagonal(sem_sim, offset=1))
    
    
    par2sim[par] = [len(tokens), len(set(tokens))/len(tokens), global_sim, consec_sim]
    
    
# put the result into a data frame    
task = pd.DataFrame(par2sim).T
# reorganize the data frame
task.columns = feats
task['PAR'] = task.index
task = demo.merge(task, left_on='PAR', right_on='PAR', how='left')
task.index = range(task.shape[0])

task.to_csv('sem_el.csv', index=False)











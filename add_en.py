import numpy as np
import pandas as pd
from tqdm import tqdm 

import spacy
nlp = spacy.load("en_core_web_sm")

# 2. consonants
# demographics
demo = pd.read_csv('English_demo.csv')
pars = demo.PAR.unique().tolist()
par2group = dict(zip(demo.PAR, demo.dx))

# 4. commands
# read transcripts
par2sim = {}
for par in tqdm(pars):
    text = next(open(f'Transcripts/English/{par2group[par]}/{par}.txt', 'r', encoding='utf-8'))
    
    # process text
    doc = nlp(text)
    deps = [[(token.i, token.head.i, token.dep_) for token in sent if token.dep_ != 'punct'] for sent in doc.sents]
    dep_dist = [[np.abs(dep[0] - dep[1]) for dep in dep_] for dep_ in deps]
    par2sim[par] = (np.nanmean([np.nanmean(dep_dist_) for dep_dist_ in dep_dist]),
                    np.nanmean([np.nanmax(dep_dist_) for dep_dist_ in dep_dist]))

# put the result into a data frame    
task = pd.DataFrame(par2sim).T
# reorganize the data frame
task.columns = ['ADD', 'MDD']
task['PAR'] = task.index
task = demo.merge(task, left_on='PAR', right_on='PAR', how='left')
task.index = range(task.shape[0])

task.to_csv('add_en.csv', index=False)











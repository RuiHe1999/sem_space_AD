# packages
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm 

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


# 2. consonants
# demographics
demo = pd.read_csv('English_demo.csv')
pars = demo.PAR.unique().tolist()
par2group = dict(zip(demo.PAR, demo.dx))

# graph features
feats = ['BERT_TokenNum', 'BERT_TTR', 'BERT_GSim', 'BERT_LSim', 'BERT_GVar', 'BERT_LVar']

# window length
window_len = 1

# 4. commands
# read transcripts
par2sim = {}
for par in tqdm(pars):
    text = next(open(f'Transcripts/English/{par2group[par]}/{par}.txt', 'r', encoding='utf-8'))
    
    encoded_input = tokenizer(text, truncation=True, return_tensors='pt')
    output, pooler_output = model(input_ids=encoded_input['input_ids'],
                              attention_mask=encoded_input['attention_mask'],
                              return_dict=False)
    embeds = output[0].detach().numpy()
    
    # all-pair similarity 
    sem_sim = 1 - distance.cdist(embeds, embeds, metric='cosine')
    global_sim = np.nanmean(sem_sim[np.tril_indices(sem_sim.shape[0], k=-1)])
    consec_sim = np.nanmean(np.diagonal(sem_sim, offset=1))
    
    global_var = np.nanvar(sem_sim[np.tril_indices(sem_sim.shape[0], k=-1)])
    consec_var = np.nanvar(np.diagonal(sem_sim, offset=1))
    
    token_num = encoded_input['input_ids'][0].shape[0] - 2
    token_ttr = len(set(encoded_input['input_ids'][0][1:-1].numpy().tolist())) / token_num
    
    par2sim[par] = [token_num, token_ttr, global_sim, consec_sim, global_var, consec_var]


# put the result into a data frame    
task = pd.DataFrame(par2sim).T
# reorganize the data frame
task.columns = feats
task['PAR'] = task.index
task = demo.merge(task, left_on='PAR', right_on='PAR', how='left')
task.index = range(task.shape[0])

task.to_csv('bert_en.csv', index=False)











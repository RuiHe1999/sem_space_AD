import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. consonants
# demographics
demo = pd.read_csv('Greek_demo.csv')
pars = demo.PAR.unique().tolist()
par2group = dict(zip(demo.PAR, demo.dx))

# gpt model 
from transformers import AutoTokenizer, AutoModelForCausalLM
gpt_model_name = 'ilsp/Meltemi-7B-v1'
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
model = model.to(device)
model.eval()

# 3. functions
def compute_perplexity(sentence):
    # Encode the sentence using the tokenizer
    input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
    loss = model(input_ids, labels=input_ids).loss
    ppl = np.exp2(loss.item())
    return ppl


# 4. commands
# read transcripts
par2sim = {}
for par in tqdm(pars):
    text = next(open(f'Transcripts/Greek/{par2group[par]}/{par}.txt', 'r', encoding='utf-8'))
    
    # process text
    par2sim[par] = compute_perplexity(text)
    
ppl_df = pd.DataFrame(par2sim, index=[0]).T
ppl_df.columns = ['PPL']
ppl_df['PAR'] = ppl_df.index
ppl_df = demo.merge(ppl_df, left_on='PAR', right_on='PAR', how='left')
ppl_df.to_csv('ppl_el.csv', index=False)



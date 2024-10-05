import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import torch.nn.functional as F

import clip
from PIL import Image

# 2. consonants
# demographics
demo = pd.read_csv('English_demo.csv')
pars = demo.PAR.unique().tolist()
par2group = dict(zip(demo.PAR, demo.dx))

# CLIP model 
clip_model, clip_preprocess = clip.load("ViT-L/14")

# 3. functions
def clip_sim(text, picture):
    # get text features
    text_features = clip_model.encode_text(clip.tokenize(text, truncate=True))

    
    # get image features
    image = [clip_preprocess(Image.open(picture).convert("RGB"))]
    image_input = torch.tensor(np.stack(image))
    image_feature = clip_model.encode_image(image_input).float()[0]

    return torch.nanmean(F.cosine_similarity(text_features, image_feature)).item()

# 4. commands
# read transcripts
par2sim = {}
for par in tqdm(pars):
    text = next(open(f'Transcripts/English/{par2group[par]}/{par}.txt', 'r', encoding='utf-8'))
    
    # process text
    sim = clip_sim(text, 'pictures/The-Cookie-Theft-picture.jpg')
    par2sim[par] = sim 
    
clip_df = pd.DataFrame(par2sim, index=[0]).T
clip_df.columns = ['CLIP']
clip_df['PAR'] = clip_df.index
clip_df = demo.merge(clip_df, left_on='PAR', right_on='PAR', how='left')
clip_df.to_csv('CLIP_en.csv', index=False)




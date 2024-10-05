import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import torch.nn.functional as F

import clip
import transformers
from PIL import Image
from multilingual_clip import pt_multilingual_clip

# 2. consonants
# demographics
demo = pd.read_csv('Greek_demo.csv')
pars = demo.PAR.unique().tolist()
par2group = dict(zip(demo.PAR, demo.dx))

# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')

# CLIP model 
clip_model, clip_preprocess = clip.load("ViT-L/14")


# 3. commands
# read transcripts
texts = [next(open(f'Transcripts/Greek/{par2group[par]}/{par}.txt', 'r', encoding='utf-8')) for par in pars]
# get text features    
text_features = model.forward(texts, tokenizer)
# get image features
image = [clip_preprocess(Image.open('pictures/greek.jpg').convert("RGB"))]
image_input = torch.tensor(np.stack(image))
image_feature = clip_model.encode_image(image_input).float()[0]

demo['CLIP'] = F.cosine_similarity(text_features, image_feature).detach().numpy()

demo.to_csv('CLIP_el.csv', index=False)
   











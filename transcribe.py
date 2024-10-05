import os 
import whisper 
import pandas as pd
from tqdm import tqdm 

# group information
en_demo = pd.read_csv('English/English_demo.csv')
el_demo = pd.read_csv('Greek/Greek_demo.csv')

id2group = dict(zip(en_demo.adressfname, en_demo.dx))
id2group.update(dict(zip(el_demo.adressfname, el_demo.dx)))

# retrive files
en_files = [os.path.join(root, file) for root, dirs, files in os.walk('English') for file in files if file.endswith('.mp3')]
el_files = [os.path.join(root, file) for root, dirs, files in os.walk('Greek') for file in files if file.endswith('.mp3')] + \
           [os.path.join(root, file) for root, dirs, files in os.walk('Greek') for file in files if file.endswith('.wav')] 

# model = whisper.load_model("small")
model = whisper.load_model("large-v3")

for file in tqdm(en_files):
    result = model.transcribe(file, language="en")
    text = result['text'].strip()
    index = file.split('/')[-1].split('.')[0]
    group = id2group[index]
    path = f'Transcripts/English/{group}/{index}.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

for file in tqdm(el_files):
    result = model.transcribe(file, language="el")
    text = result['text'].strip()
    index = file.split('/')[-1].split('.')[0]
    group = id2group[index]
    path = f'Transcripts/Greek/{group}/{index}.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)









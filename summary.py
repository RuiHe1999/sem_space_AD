
import numpy as np
import pandas as pd
from sklearn import preprocessing

# group information
en_demo = pd.read_csv('English_demo.csv')
el_demo = pd.read_csv('Greek_demo.csv')

en_demo['language'] = 'EN'
el_demo['language'] = 'EL'

# merge demo 
demo = pd.concat([en_demo, el_demo])

# several nans in edcuation for controls in the English dataset
# fill nan in educ with the averaged edcuation for controls in the English dataset
demo['educ'] = demo['educ'].fillna(int(np.round(np.nanmean(en_demo[en_demo['dx']=='Control']['educ']))))

# one nan in mmse for controls in the Greek dataset
# fill nan in mmse with the averaged mmse for controls in the Greek dataset
demo['mmse'] = demo['mmse'].fillna(int(np.round(np.nanmean(el_demo[el_demo['dx']=='Control']['mmse']))))

# change diagnosis name to abbrevations
demo['dx'] = demo['dx'].apply(lambda x: {'Control':'NC', 'ProbableAD':'pAD'}[x])

# results
sem_sim = pd.concat([pd.read_csv('sem_en.csv'), pd.read_csv('sem_el.csv')])
bert_sim = pd.concat([pd.read_csv('bert_en.csv'), pd.read_csv('bert_el.csv')])
# sem_graph = pd.concat([pd.read_csv('FT_graph_en.csv'), pd.read_csv('FT_graph_el.csv')])
# sem_graph = sem_graph.drop(columns=['FT_GSim', 'FT_LSim',])
clip = pd.concat([pd.read_csv('clip_en.csv'), pd.read_csv('clip_el.csv')])
ppl = pd.concat([pd.read_csv('ppl_en.csv'), pd.read_csv('ppl_el.csv')])
add = pd.concat([pd.read_csv('add_en.csv'), pd.read_csv('add_el.csv')])

# merge data
merged_df = demo[['PAR', 'age', 'gender', 'educ', 'dx', 'mmse', 'language']]
for df in [sem_sim, bert_sim, clip, ppl, add]:
    columns_to_keep = [col for col in df.columns if col not in merged_df.columns or col == 'PAR']
    merged_df = pd.merge(merged_df, df[columns_to_keep], on='PAR', how='left')
        
merged_df.to_csv('results/data.csv', index=False)






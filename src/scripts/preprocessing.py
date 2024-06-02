#%%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

#%%
import sys, os
sys.path.append('./')

from src.data import make_dataset
from src.features import build_features
import numpy as np

data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_path = '/home/gourab/Projects/CoV research/Tempel-modified/source_code/data/raw/'

print('---> Reading in TrigramVecs...')
trigram_to_idx, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)
print(f'Number of possible 3-grams: {len(trigram_to_idx)}')
print(f'Dimension of TrigramVecs: {len(trigram_vecs_data[0])}')

print('\n---> Reading in strains...')
strains_by_year = make_dataset.read_strains_from(data_files, data_path='./data/raw/H1N1/')
print(f'Strains from {len(data_files)} years were read.')
print(f'Shape: {np.array(strains_by_year).shape}')
print(f'Example strain:\n{strains_by_year[0][0]}')

print('\n---> Constructing training data...')
num_of_samples = 100

strains_by_year = build_features.sample_strains(strains_by_year, num_of_samples)

trigrams_by_year = build_features.split_to_trigrams(strains_by_year)
print(f'Each of {len(trigrams_by_year[0])} year strains were split into {len(trigrams_by_year[0][0])} trigrams.')
print(f'Shape: {np.array(trigrams_by_year).shape}')
# print(f'Example of trigram is {trigrams_by_year[0][0][0]}')

# trigram_idxs_by_year = build_features.trigrams_to_indexes(trigrams_by_year, trigram_to_idx)
trigram_idxs_by_year = build_features.map_trigrams_to_idxs(trigrams_by_year, trigram_to_idx)
print('\nIndex conversion performed.')
print(f'Shape: {np.array(trigram_idxs_by_year).shape}')

squeezed_trigrams_by_year = build_features.squeeze_trigrams(trigram_idxs_by_year)
print('\nIndex squeezing performed.')
print(f'Shape: {np.array(squeezed_trigrams_by_year).shape}')

# trigram_vecs = build_features.indexes_to_trigram_vecs(squeezed_trigrams_by_year, trigram_vecs_data)
trigram_vecs = build_features.map_idxs_to_vecs(squeezed_trigrams_by_year, trigram_vecs_data)
print('\nTrigramVec conversion performed.')
print(f'Shape: {np.array(trigram_vecs).shape}')
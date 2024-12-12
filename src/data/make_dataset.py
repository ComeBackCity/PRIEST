import pandas as pd
import random
import math

# My modifications
from colorama import Fore, Back
# from pathlib import Path
# import os

# subtype_flag = 0

def subtype_selection(subtype):
    global subtype_flag, data_path
    print(subtype)
    base = '/home/gourab/Desktop/Projects/Covid/Tempel-modified/'
    # print(datSa_path)
    if subtype == 'H1N1':
        subtype_flag = 0
        #data_path = 'C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/2_Conference/CIKM2019/code/influenza-master/data/raw/H1N1/'
        data_path = base + 'source_code/data/raw/H1N1_cluster/'
    elif subtype == 'H3N2':
        subtype_flag = 1
        #data_path = 'C:/Users/yinr0002>/Google Drive/Tier_2_MOE2014/2_Conference/CIKM2019/code/influenza-master/data/raw/H3N2/'
        data_path = base + 'source_code/data/raw/H3N2_cluster/'
    elif subtype == 'H5N1':
        subtype_flag = 2
        #data_path = 'C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/2_Conference/CIKM2019/code/influenza-master/data/raw/H5N1/'
        data_path = base + 'source_code/data/raw/H5N1_cluster/'

    # print('-------------------') 
    # print(subtype_flag, data_path)
    # print('-------------------')
    return subtype_flag, data_path


def read_trigram_vecs(data_path, subtype='H1N1'):
  """
  Reads the csv file containing 100 dimensional prot vecs, the 
  data_path argument indicating where it is located.
  Returns a dictionary that maps a 3gram of amino acids to its
  index and a numpy array containing the trigram vecs.
  """
  _, data_path = subtype_selection(subtype)
  # print(Fore.GREEN + data_path)
  prot_vec_file = 'protVec_100d_3grams.csv'
    
  df = pd.read_csv(data_path + prot_vec_file, delimiter = '\t')
  trigrams = list(df['words'])

  # 2 level mapping
  # trigram -> index -> vector

  trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
  trigram_vecs = df.loc[:, df.columns != 'words'].values
  
  return trigram_to_idx, trigram_vecs


def read_strains_from(data_files, data_path):
  """
  Reads the raw strains from the data_files located by the data_path.
  Returns a pandas series for each data file, contained in a ordered list.
  """
  #_, data_path = subtype_selection(subtype)
  raw_strains = []
  print(data_files)
  print(data_path)
  for file_name in data_files:
    # print(data_path)
    # print(file_name)
    file_path = data_path + file_name
    # print(file_path)
    # print(file_path)
    
    # if not os.path.exists(data_path):
    #   os.makedirs(data_path)
    # else:
    #   print('Already exists')

    # print('Hi------------>')
    df = pd.read_csv(file_path)
    # print(df)
    strains = df['seq']
    raw_strains.append(strains)
    # print(raw_strains[0])
    # print('\n\n\n')
    # print(raw_strains)
  # print('------------>')  
  # print(raw_strains[0])
  # print('------------>')  

  return raw_strains


def train_test_split_strains(strains_by_year, test_split, cluster='random'):
  """
  Shuffles the strains in each year and splits them into two disjoint sets,
  of size indicated by the test_split.
  Expects and returns pandas dataframe or series.
  """
  # print(cluster)
  train_strains, test_strains = [], []
  if cluster == 'random':
      for strains in strains_by_year:
          num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
          shuffled_strains = strains.sample(frac=1).reset_index(drop=True)
          train = shuffled_strains.iloc[:num_of_training_examples].reset_index(drop=True)
          test = shuffled_strains.iloc[num_of_training_examples:].reset_index(drop=True)
          train_strains.append(train)
          test_strains.append(test)
  else:
      #change the starting index for the time-series training samples for multiple experiments
      for strains in strains_by_year:
          num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
          train = strains.iloc[:800].reset_index(drop=True)
          test = strains.iloc[800:1000].reset_index(drop=True)
          train_strains.append(train)
          test_strains.append(test)
  return train_strains, test_strains




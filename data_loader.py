import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
# from PIL import Image
# from tqdm import tqdm
import time
import copy
import random
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from fastNLP.embeddings import ElmoEmbedding, StaticEmbedding
from fastNLP import Vocabulary
import math


class TableDataset(Dataset):
	def __init__(self, data, label, selected_features, permutation=None):
		'''
		data should be numpy matrixes
		'''
		if selected_features is None:
			self.data = data[:, selected_features]
			self.label = label
		elif permutation is None:
			self.data = data[:, selected_features]
			self.label = label
		else:
			self.data = data[permutation][:, selected_features]
			self.label = label[permutation]
		
		self.pos_rate = sum(self.label) / len(self.label)
		print('pos_rate of current task: ', self.pos_rate)
		self.num_class = len(np.unique(self.label))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.label[idx]

def AirSat_data_loader(train_path='RecData/AirSat/train.csv', test_path='RecData/AirSat/test.csv', num_tasks=4, n_bins=5, train_amount=100000):
    train = pd.read_csv(train_path, usecols=range(2, 25)).dropna()
    test = pd.read_csv(test_path, usecols=range(2, 25)).dropna()

    label_dict = {'neutral or dissatisfied': 0, 'satisfied': 1}

    # put numerical data into bins and process the format of data
    ests = {}
    for col in train.columns:
        if len(train[col].value_counts()) > 10 and pd.api.types.is_numeric_dtype(train[col]):
            col_data = train[col].to_numpy().reshape(-1, 1)
            est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
            est.fit(col_data)
            ests[col] = est
            transform_feature = train[col].to_numpy().reshape(-1, 1)
            train[col] = est.transform(transform_feature)
            
            transform_feature = test[col].to_numpy().reshape(-1, 1)
            test[col] = est.transform(transform_feature)
        if pd.api.types.is_numeric_dtype(train[col]):
            # train[col] = train[col].astype(int).apply(lambda x: str(x))
            # test[col] = test[col].astype(int).apply(lambda x: str(x))
            train[col] = train[col].astype(int).apply(lambda x: str(x))
            test[col] = test[col].astype(int).apply(lambda x: str(x))
        else:
            train[col] = train[col].apply(lambda x: x.lower())
            test[col] = test[col].apply(lambda x: x.lower())
        
    train['satisfaction'] = train['satisfaction'].apply(lambda x: label_dict[x])
    test['satisfaction'] = test['satisfaction'].apply(lambda x: label_dict[x])

    # pick out the columns of training data
    input_columns = train.columns[:-1]

    # build the vocabulary for training data
    vocab_word = Vocabulary()

    cell_dict = {}
    for col in input_columns:
        unique_words = train[col].value_counts().keys()
        for word in unique_words:
            vocab_word.add_word_lst(word.split())
            if word not in cell_dict.keys():
                cell_dict['-'.join(word.split())] = word.split()

    for task in range(num_tasks):
        vocab_word.add_word('task'+str(task))
    vocab_word.add_word('task_sh')
    # vocab_word.add_word('<T>')

    # build the basic embedding for all words
    # embed_word = StaticEmbedding(vocab_word, model_dir_or_name='en-glove-6b-50d')
    embed_word = StaticEmbedding(vocab_word, model_dir_or_name=None, embedding_dim=16)

    # concat the cell's words into one token and generate embeddings for cells.
    # cell_embedding_dict = {'<SOS>': embed_word(torch.LongTensor([[vocab_word.to_index('<SOS>')]]))}

    cell_embedding_dict = {}
    for cell in cell_dict.keys():
        embeded = embed_word(torch.LongTensor([[vocab_word.to_index(word) for word in cell_dict[cell]]]))    
        cell_embedding_dict[cell] = torch.mean(embeded, 1)

    # generate the embeddings and the w2i dictionary.
    for task in range(num_tasks):
        cell_embedding_dict['task'+str(task)] = embed_word(torch.LongTensor([vocab_word.to_index('task'+str(task))]))
    cell_embedding_dict['task_sh'] = embed_word(torch.LongTensor([vocab_word.to_index('task_sh')]))

    print(cell_embedding_dict.keys())

    source_embeddings = torch.cat(list(cell_embedding_dict.values()), 0)
    word2idx = {word: i for i, word in enumerate(cell_embedding_dict.keys())}
    print(source_embeddings.shape)

    # convert the training data into indexed words.
    for col in input_columns:
        train[col] = train[col].apply(lambda x: word2idx['-'.join(x.split())])
        test[col] = test[col].apply(lambda x: word2idx['-'.join(x.split())])

    np.random.seed(1)
    feat_numbers = np.random.randint(10, 14, size=num_tasks)

    input_columns_index = list(range(len(input_columns)))
    random.seed(10)
    feat_groups = [random.sample(input_columns_index, feat_numbers[i]) for i in range(num_tasks)]

    total_tasks = num_tasks
    print('number of total tasks: ', total_tasks)
    print(feat_groups)
    numerical_cols = [3, 6, 20, 21]
    numerical_count = [0]*num_tasks
    for i, group in enumerate(feat_groups):
        for c in group:
            if c in numerical_cols:
                numerical_count[i] += 1
    print(numerical_count)

    train_data = train.to_numpy()[:train_amount, :-1]
    train_label = train.to_numpy()[:train_amount, -1]
    test_data = test.to_numpy()[:train_amount, :-1]
    test_label = test.to_numpy()[:train_amount, -1]

    tr_feat_groups = copy.deepcopy(feat_groups)

    task_data_amount = int(1 / total_tasks * len(train_data))

    np.random.seed(1)

    permutation_list = [list(np.random.permutation(len(train_data))[:task_data_amount]) for i in range(total_tasks)]

    train_datasets = []
    test_datasets = []
    num_classes = []

    for task in range(total_tasks):
        task_data_amount = int(1 / 3 * len(train_data))
        permutation = list(np.random.permutation(len(train_data))[:task_data_amount])
        print('prepare data for task: ', task)
        train_datasets.append(TableDataset(train_data.astype(int), train_label, tr_feat_groups[task], permutation))
        num_classes.append(train_datasets[task].num_class)
        test_datasets.append(TableDataset(test_data.astype(int), test_label, tr_feat_groups[task]))

    return train_datasets, test_datasets, num_classes, word2idx, source_embeddings

def jannis_data_loader(train_path='data/jannis/jannis_train.data', label_path='data/jannis/jannis_train.solution', num_tasks=4, n_bins=10, train_amount=100000):
    np.random.seed(1)

    feat_code = range(54)
    data = pd.read_csv(train_path, delimiter=' ', names=feat_code, usecols=feat_code)
    label = pd.read_csv(label_path, header=None, names=['label'])
    label['label'] = label['label'].apply(lambda x: x.split().index('1'))
    
    train = pd.concat([data, label], axis=1).copy()
    train = train[train['label'] != 0]

    # put numerical data into bins and process the format of data
    ests = {}
    for col in data.columns:
        if len(train[col].value_counts()) > 10 and pd.api.types.is_numeric_dtype(train[col]):
            scaler = MinMaxScaler()
            col_data = train[col].to_numpy().reshape(-1, 1)
            col_data = scaler.fit_transform(col_data)
            est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
            est.fit(col_data)
            ests[col] = est
            transform_feature = train[col].to_numpy().reshape(-1, 1)
            train[col] = est.transform(transform_feature)
            
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col] = train[col].astype(int).apply(lambda x: str(x))
        else:
            train[col] = train[col].apply(lambda x: x.lower())
        
    # pick out the columns of training data
    input_columns = train.columns[:-1]

    # build the vocabulary for training data
    vocab_word = Vocabulary()

    cell_dict = {}
    for col in input_columns:
        unique_words = train[col].value_counts().keys()
        for word in unique_words:
            vocab_word.add_word_lst(word.split())
            if word not in cell_dict.keys():
                cell_dict['-'.join(word.split())] = word.split()

    for task in range(num_tasks):
        vocab_word.add_word('task'+str(task))
    vocab_word.add_word('task_sh')
    # vocab_word.add_word('<T>')

    # build the basic embedding for all words
    # embed_word = StaticEmbedding(vocab_word, model_dir_or_name='en-glove-6b-50d')
    embed_word = StaticEmbedding(vocab_word, model_dir_or_name=None, embedding_dim=16)

    cell_embedding_dict = {}
    for cell in cell_dict.keys():
        embeded = embed_word(torch.LongTensor([[vocab_word.to_index(word) for word in cell_dict[cell]]]))    
        cell_embedding_dict[cell] = torch.mean(embeded, 1)

    # generate the embeddings and the w2i dictionary.
    for task in range(num_tasks):
        cell_embedding_dict['task'+str(task)] = embed_word(torch.LongTensor([vocab_word.to_index('task'+str(task))]))
    cell_embedding_dict['task_sh'] = embed_word(torch.LongTensor([vocab_word.to_index('task_sh')]))

    print(cell_embedding_dict.keys())

    source_embeddings = torch.cat(list(cell_embedding_dict.values()), 0)
    word2idx = {word: i for i, word in enumerate(cell_embedding_dict.keys())}
    print(source_embeddings.shape)

    # convert the training data into indexed words.
    for col in input_columns:
        train[col] = train[col].apply(lambda x: word2idx['-'.join(x.split())])

    task_data = []
    for i in range(num_tasks):
        # rand_class = np.random.randint(1, 4)
        task_data.append(train[train['label'] != i+1].copy())
        classes = pd.unique(task_data[i]['label'])
        class_dict = {cls: i for i, cls in enumerate(classes)}
        task_data[i]['label'] = task_data[i]['label'].apply(lambda x: class_dict[x])

    # build the tasks and arrange the order of tasks
    feat_numbers = np.random.randint(34, 44, size=num_tasks)

    input_columns_index = list(range(len(input_columns)))
    random.seed(10)
    feat_groups = [random.sample(input_columns_index, feat_numbers[i]) for i in range(num_tasks)]

    total_tasks = num_tasks
    print('number of total tasks: ', total_tasks)

    # train = train.to_numpy()
    # re_data = train.to_numpy()[:train_amount, :-1]
    # re_label = train.to_numpy()[:train_amount, -1]
    # train_data, test_data, train_label, test_label = train_test_split(re_data, re_label, test_size=0.2, random_state=42)

    tr_feat_groups = copy.deepcopy(feat_groups)

    # task_data_amount = int(1 / total_tasks * len(train_data))
    # permutation_list = [list(np.random.permutation(len(train_data))[:task_data_amount]) for i in range(total_tasks)]

    train_datasets = []
    test_datasets = []
    num_classes = []

    for task in range(total_tasks):
        task_data_numpy = task_data[task].to_numpy()
        train_data, test_data, train_label, test_label = train_test_split(task_data_numpy[:, :-1], task_data_numpy[:, -1], test_size=0.2, random_state=42)

        print('prepare data for task: ', task)
        train_datasets.append(TableDataset(train_data.astype(int), train_label, tr_feat_groups[task]))
        num_classes.append(train_datasets[task].num_class)
        # num_classes.append(4)
        test_datasets.append(TableDataset(test_data.astype(int), test_label, tr_feat_groups[task]))
    
    print(num_classes)

    return train_datasets, test_datasets, num_classes, word2idx, source_embeddings
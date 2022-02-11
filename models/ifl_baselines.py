import pandas as pd

import numpy as np
import pickle
import re
import random
import copy
import math
import time

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils.classifier import Specific_Model
from utils.extractor import Transformer
from utils.support_funcs import AccuarcyCompute

import datetime

from prettytable import PrettyTable


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# train_datasets, test_datasets, word2idx, alpha=0.2, beta=0.2, gamma=5, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10

# blank shared and specific extractors. Joint
def BaselineEnsembleJoint(train_datasets, test_datasets, num_classes, word2idx, save_path, alpha=0.2, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10):
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloaders = []
    test_dataloaders = []
    for task in range(total_tasks):
        train_dataloaders.append(DataLoader(train_datasets[task], batch_size=batch_size, shuffle=True, generator=g))
        test_dataloaders.append(DataLoader(test_datasets[task], batch_size=batch_size, shuffle=True, generator=g))
    
    embeddings = copy.deepcopy(source_embeddings.detach())

    shared_extractor = Transformer(num_tokens=embeddings.shape[0], dim_model=embeddings.shape[1], num_heads=2, 
                            num_encoder_layers=2, num_decoder_layers=2, embeddings=embeddings, dropout_p=0)

    specific_extractors = []
    for task in range(total_tasks):
        specific_extractors.append(Transformer(num_tokens=embeddings.shape[0], dim_model=embeddings.shape[1], num_heads=2, 
                            num_encoder_layers=2, num_decoder_layers=2, embeddings=embeddings, dropout_p=0))

    classifiers = []
    shared_classifiers = []

    accuracy_matrix = np.zeros((total_tasks, total_tasks))
    total_time = 0

    for task in range(total_tasks):
        print('training on task: ', task)
        classifier_params = {'num_specific_features': 8, 'num_specific_classes': num_classes[task]}

        classifiers.append(Specific_Model(classifier_params))
        shared_classifiers.append(Specific_Model(classifier_params))

        shared_extractor = shared_extractor.to(device)
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].to(device)
            classifiers[t] = classifiers[t].to(device)
            shared_classifiers[t] = shared_classifiers[t].to(device)

        shared_optimizer = torch.optim.Adam(shared_extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        specific_optimizer = torch.optim.Adam(specific_extractors[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        shared_classifier_optimizer = torch.optim.Adam(shared_classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        classifier_optimizer = torch.optim.Adam(classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)

        old_specific_optimizers = []
        old_classifier_optimizers = []
        old_shared_classifier_optimizers = []

        for old_task in range(task):
            old_specific_optimizers.append(torch.optim.Adam(specific_extractors[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))
            old_classifier_optimizers.append(torch.optim.Adam(classifiers[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))
            old_shared_classifier_optimizers.append(torch.optim.Adam(shared_classifiers[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))

        # train the new classification task
        current_early_stop = 0
        best_loss = np.inf    
        
        for epoch in range(epochs):
            running_loss = 0

            ts = time.time()
            
            for i, (data, label) in enumerate(train_dataloaders[task]):
                data = data.to(device)
                label = label.to(device)

                # forward propagation
                shared_optimizer.zero_grad()
                specific_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                shared_classifier_optimizer.zero_grad()

                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                shared_output = shared_classifiers[task](mid_shared_output)
                shared_output = torch.softmax(shared_output, dim=1)


                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                mid_specific_output = specific_extractors[task](torch.column_stack((specific_task_info, data)), specific_task_info)
                specific_output = classifiers[task](mid_specific_output)
                specific_output = torch.softmax(specific_output, dim=1)
                
                output = (alpha * shared_output + (1 - alpha) * specific_output)
                output = torch.log(output)
                speci_loss = nn.NLLLoss()(output, label)

                speci_loss.backward()
                shared_optimizer.step()
                specific_optimizer.step()
                classifier_optimizer.step()
                shared_classifier_optimizer.step()

                running_loss += speci_loss.item()

            torch.cuda.empty_cache()

                # finetune the old tasks
            if task > 0:
                for old_task in range(task):
                    for i, (data, label) in enumerate(train_dataloaders[old_task]):
                        data = data.to(device)
                        label = label.to(device)
                        
                        shared_optimizer.zero_grad()
                        old_shared_classifier_optimizers[old_task].zero_grad()
                        old_specific_optimizers[old_task].zero_grad()
                        old_classifier_optimizers[old_task].zero_grad()

                        shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                        mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                        shared_output = shared_classifiers[old_task](mid_shared_output)
                        shared_output = torch.softmax(shared_output, dim=1)

                        specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                        mid_specific_output = specific_extractors[old_task](torch.column_stack((specific_task_info, data)), specific_task_info)
                        specific_output = classifiers[old_task](mid_specific_output)
                        specific_output = torch.softmax(specific_output, dim=1)
                        
                        new_output = (alpha * shared_output + (1 - alpha) * specific_output)
                        new_output = torch.log(new_output)                   

                        speci_loss = nn.NLLLoss()(new_output, label)
                        running_loss += speci_loss.item()

                        speci_loss.backward()
                        shared_optimizer.step()
                        old_shared_classifier_optimizers[old_task].step()
                        old_specific_optimizers[old_task].step()
                        old_classifier_optimizers[old_task].step()
                    
                    torch.cuda.empty_cache()

            te = time.time()
            total_time += te - ts

            if running_loss < best_loss:
                best_loss = running_loss
                current_early_stop = 0
            else:
                current_early_stop += 1
            
            torch.cuda.empty_cache()

            if current_early_stop > early_stop_threshold:
                break
            elif epoch % 10 == 0:
                print('epoch is: ', epoch, 'loss is: ', running_loss)
                        
        # print('currently total tasks: ', task + 1)

        shared_extractor = shared_extractor.cpu()
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].cpu()
            classifiers[t] = classifiers[t].cpu()
            shared_classifiers[t] = shared_classifiers[t].cpu()
        
        shared_extractor.eval()
        for t in range(task + 1):
            specific_extractors[t].eval()
            classifiers[t].eval()
            shared_classifiers[t].eval()

        # test the accuracy of all current models
        for t in range(task + 1):
            accuracy_list = []
            for i, (data, label) in enumerate(test_dataloaders[t]):
                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']
                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']

                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                mid_specific_output = specific_extractors[t](torch.column_stack((specific_task_info, data)), specific_task_info)

                shared_output = shared_classifiers[t](mid_shared_output)
                specific_output = classifiers[t](mid_specific_output)

                output = (alpha * torch.softmax(shared_output, dim=1) + (1 - alpha) * torch.softmax(specific_output, dim=1))
                accuracy_list.append(AccuarcyCompute(output, label))
            accuracy_matrix[t, task] = sum(accuracy_list) / len(test_dataloaders[t].dataset)

        shared_extractor.train()
        for t in range(task + 1):
            specific_extractors[t].train()
            classifiers[t].train()
            shared_classifiers[t].train()

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'learning_rate', 'weight decay', 'batch size', 'alpha'])
    table.add_row([total_time, '%.4f' %np.mean(accuracy_matrix[:, -1]), learning_rate, '%.4f' %weight_decay, batch_size, '%.4f' %alpha])
    print(table)
    print('Table for training result')
    print(accuracy_matrix)
    today = datetime.date.today()
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(accuracy_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close()    
    return np.mean(accuracy_matrix[:, -1])

# blank shared and specific extractors. Finetune
def BaselineEnsembleFinetune(train_datasets, test_datasets, num_classes, word2idx, save_path, alpha=0.2, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10):
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloaders = []
    test_dataloaders = []
    for task in range(total_tasks):
        train_dataloaders.append(DataLoader(train_datasets[task], batch_size=batch_size, shuffle=True, generator=g))
        test_dataloaders.append(DataLoader(test_datasets[task], batch_size=batch_size, shuffle=True, generator=g))

    embeddings = copy.deepcopy(source_embeddings.detach())

    shared_extractor = Transformer(num_tokens=embeddings.shape[0], dim_model=embeddings.shape[1], num_heads=2, 
                            num_encoder_layers=2, num_decoder_layers=2, embeddings=embeddings, dropout_p=0)

    specific_extractors = []
    for task in range(total_tasks):
        specific_extractors.append(Transformer(num_tokens=embeddings.shape[0], dim_model=embeddings.shape[1], num_heads=2, 
                            num_encoder_layers=2, num_decoder_layers=2, embeddings=embeddings, dropout_p=0))

    classifiers = []
    shared_classifiers = []

    accuracy_matrix = np.zeros((total_tasks, total_tasks))
    total_time = 0

    for task in range(total_tasks):
        print('training on task: ', task)
        classifier_params = {'num_specific_features': 8, 'num_specific_classes': num_classes[task]}

        classifiers.append(Specific_Model(classifier_params))
        shared_classifiers.append(Specific_Model(classifier_params))

        shared_extractor = shared_extractor.to(device)
        # shared_classifier = shared_classifier.to(device)
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].to(device)
            classifiers[t] = classifiers[t].to(device)
            shared_classifiers[t] = shared_classifiers[t].to(device)

        shared_optimizer =torch.optim.Adam(shared_extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        specific_optimizer = torch.optim.Adam(specific_extractors[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        shared_classifier_optimizer = torch.optim.Adam(shared_classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        classifier_optimizer = torch.optim.Adam(classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)

        # train the new classification task
        current_early_stop = 0
        best_loss = np.inf    
        
        for epoch in range(epochs):
            running_loss = 0

            ts = time.time()
            
            for i, (data, label) in enumerate(train_dataloaders[task]):
                data = data.to(device)
                label = label.to(device)

                # forward propagation
                shared_optimizer.zero_grad()
                specific_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                shared_classifier_optimizer.zero_grad()

                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                shared_output = shared_classifiers[task](mid_shared_output)
                shared_output = torch.softmax(shared_output, dim=1)

                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                mid_specific_output = specific_extractors[task](torch.column_stack((specific_task_info, data)), specific_task_info)
                specific_output = classifiers[task](mid_specific_output)
                specific_output = torch.softmax(specific_output, dim=1)
                
                output = (alpha * shared_output + (1 - alpha) * specific_output)
                output = torch.log(output)
                speci_loss = nn.NLLLoss()(output, label)

                speci_loss.backward()
                shared_optimizer.step()
                specific_optimizer.step()
                classifier_optimizer.step()
                shared_classifier_optimizer.step()

                running_loss += speci_loss.item()
            
            te = time.time()
            total_time += te - ts
            
            if running_loss < best_loss:
                best_loss = running_loss
                current_early_stop = 0
            else:
                current_early_stop += 1
            metric = running_loss
            
            torch.cuda.empty_cache()

            if current_early_stop > early_stop_threshold:
                break
                        
        # print('currently total tasks: ', task + 1)

        shared_extractor = shared_extractor.cpu()
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].cpu()
            classifiers[t] = classifiers[t].cpu()
            shared_classifiers[t] = shared_classifiers[t].cpu()

        shared_extractor.eval()
        for t in range(task + 1):
            specific_extractors[t].eval()
            classifiers[t].eval()
            shared_classifiers[t].eval()
        
        # test the accuracy of all current models
        for t in range(task + 1):
            accuracy_list = []
            for i, (data, label) in enumerate(test_dataloaders[t]):
                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']
                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']

                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                mid_specific_output = specific_extractors[t](torch.column_stack((specific_task_info, data)), specific_task_info)

                shared_output = shared_classifiers[t](mid_shared_output)
                specific_output = classifiers[t](mid_specific_output)

                output = (alpha * torch.softmax(shared_output, dim=1) + (1 - alpha) * torch.softmax(specific_output, dim=1))
                accuracy_list.append(AccuarcyCompute(output, label))
            accuracy_matrix[t, task] = sum(accuracy_list) / len(test_dataloaders[t].dataset)

        shared_extractor.train()
        for t in range(task + 1):
            specific_extractors[t].train()
            classifiers[t].train()
            shared_classifiers[t].train()

    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'learning_rate', 'weight decay', 'batch size', 'alpha'])
    table.add_row([total_time, '%.4f' %np.mean(accuracy_matrix[:, -1]), learning_rate, '%.4f' %weight_decay, batch_size, '%.4f' %alpha])
    print(table)
    print('Table for training result')
    print(accuracy_matrix)
    today = datetime.date.today()
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(accuracy_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close()    
    return np.mean(accuracy_matrix[:, -1])
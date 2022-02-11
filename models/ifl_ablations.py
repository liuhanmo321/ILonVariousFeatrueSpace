import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import time
import copy
import random
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math

from utils.classifier import Specific_Model
from utils.extractor import Transformer
from utils.support_funcs import AccuarcyCompute

import datetime

from prettytable import PrettyTable


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Refined Discriminability. Ensemble model, modified loss function
def SharedOnly(train_datasets, test_datasets, num_classes, word2idx, save_path, use_dst=True, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10):
    def MultiClassCrossEntropy(logits, labels, T):
        # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
        labels = Variable(labels.data, requires_grad=False)
        outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
        labels = torch.softmax(labels / T, dim=1)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
        return Variable(outputs.data, requires_grad=True)

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

    # shared_classifier = Specific_Model(classifier_params)
    shared_classifiers = []

    accuracy_matrix = np.zeros((total_tasks, total_tasks))
    total_time = 0

    for task in range(total_tasks):
        print('training on task: ', task + 1)
        # print('pos rate: ', pos_rates[task])
        classifier_params = {'num_specific_features': 8, 'num_specific_classes': num_classes[task]}
        shared_classifiers.append(Specific_Model(classifier_params))

        shared_extractor = shared_extractor.to(device)
        # shared_classifier = shared_classifier.to(device)
        for t in range(task + 1):
            shared_classifiers[t] = shared_classifiers[t].to(device)

        old_shared_extractor = copy.deepcopy(shared_extractor).to(device)
        # for t in range(task + 1):
        #     old_shared_classifier = copy.deepcopy(shared_classifier).to(device)

        shared_optimizer = torch.optim.Adam(shared_extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        shared_classifier_optimizer = torch.optim.Adam(shared_classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)

        old_shared_classifier_optimizers = []

        for old_task in range(task):
            old_shared_classifier_optimizers.append(torch.optim.Adam(shared_classifiers[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))

        # train the new classification task
        current_early_stop = 0
        best_loss = np.inf
        
        for epoch in range(epochs):            
            total_loss = 0
            running_loss = 0
            
            ts = time.time()
            for i, (data, label) in enumerate(train_dataloaders[task]):
                data = data.to(device)
                label = label.to(device)

                # forward propagation
                shared_optimizer.zero_grad()
                shared_classifier_optimizer.zero_grad()

                for old_task in range(task):
                    old_shared_classifier_optimizers[old_task].zero_grad()

                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                old_mid_shared_output = old_shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)

                shared_output = shared_classifiers[task](mid_shared_output)                
                # shared_p = torch.softmax(shared_output, dim=1)
                # output = torch.log(shared_p)
                # norm_loss = nn.NLLLoss()(output, label)
                norm_loss = nn.CrossEntropyLoss()(shared_output, label)
                running_loss += norm_loss.item()
                total_loss = norm_loss             

                if task > 0 and use_dst:
                    for old_task in range(task):                        
                        past_shared_output = shared_classifiers[old_task](mid_shared_output)
                        old_past_shared_output = shared_classifiers[old_task](old_mid_shared_output)
                                            
                        speci_loss = MultiClassCrossEntropy(past_shared_output, old_past_shared_output, T=T)
                        total_loss += speci_loss / task
                        running_loss += speci_loss.item() / task
                
                total_loss.backward()
                shared_classifier_optimizer.step()
                shared_optimizer.step()
                for old_task in range(task):
                    old_shared_classifier_optimizers[old_task].step()
            
            te = time.time()
            total_time += te - ts

            if metrics == 'loss':
                if running_loss < best_loss:
                    best_loss = running_loss
                    current_early_stop = 0
                else:
                    current_early_stop += 1
            
            torch.cuda.empty_cache()

            if current_early_stop > early_stop_threshold:
                break

        shared_extractor = shared_extractor.cpu()
        for t in range(task + 1):
            shared_classifiers[t] = shared_classifiers[t].cpu()

        shared_extractor.eval()
        for t in range(task + 1):
            shared_classifiers[t].eval()
        # test the accuracy of all current models
        for t in range(task + 1):
            accuracy_list = []
            for i, (data, label) in enumerate(test_dataloaders[t]):
                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']
                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)

                shared_output = shared_classifiers[t](mid_shared_output)
                output = torch.softmax(shared_output, dim=1)
                accuracy_list.append(AccuarcyCompute(output, label))
            accuracy_matrix[t, task] = sum(accuracy_list) / len(test_dataloaders[t].dataset)
        
        shared_extractor.train()
        # shared_classifier.train()
        for t in range(task + 1):
            shared_classifiers[t].train()


    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'learning_rate', 'weight decay', 'batch size'])
    table.add_row([total_time, '%.4f' %np.mean(accuracy_matrix[:, -1]), learning_rate, '%.4f' %weight_decay, batch_size])
    print(table)
    print('Table for training result')
    print(accuracy_matrix)

    today = datetime.date.today()
    print('===========================================================================')
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(accuracy_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close()    
    return np.mean(accuracy_matrix[:, -1])


def SpecificOnly(train_datasets, test_datasets, num_classes, word2idx, save_path, use_dis=True, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10):

    g = torch.Generator()
    g.manual_seed(0)
    train_dataloaders = []
    test_dataloaders = []
    for task in range(total_tasks):
        train_dataloaders.append(DataLoader(train_datasets[task], batch_size=batch_size, shuffle=True, generator=g))
        test_dataloaders.append(DataLoader(test_datasets[task], batch_size=batch_size, shuffle=True, generator=g))

    embeddings = copy.deepcopy(source_embeddings.detach())

    specific_extractors = []
    for task in range(total_tasks):
        specific_extractors.append(Transformer(num_tokens=embeddings.shape[0], dim_model=embeddings.shape[1], num_heads=2, 
                            num_encoder_layers=2, num_decoder_layers=2, embeddings=embeddings, dropout_p=0))

    # shared_classifier = Specific_Model(classifier_params)
    classifiers = []

    accuracy_matrix = np.zeros((total_tasks, total_tasks))
    total_time = 0

    for task in range(total_tasks):
        print('training on task: ', task + 1)
        # print('pos rate: ', pos_rates[task])

        classifier_params = {'num_specific_features': 8, 'num_specific_classes': num_classes[task]}
        classifiers.append(Specific_Model(classifier_params))
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].to(device)
            classifiers[t] = classifiers[t].to(device)

        specific_optimizer = torch.optim.Adam(specific_extractors[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        classifier_optimizer = torch.optim.Adam(classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)

        # train the new classification task
        current_early_stop = 0
        best_loss = np.inf
        
        for epoch in range(epochs):            
            total_loss = 0
            running_loss = 0
            
            ts = time.time()
            for i, (data, label) in enumerate(train_dataloaders[task]):
                data = data.to(device)
                label = label.to(device)
                # forward propagation
                specific_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                prepared_data = torch.column_stack((specific_task_info, data))
                
                mid_specific_output = specific_extractors[task](prepared_data, specific_task_info)
                old_mid_specific_outputs = [specific_extractors[t](prepared_data, specific_task_info) for t in range(task)]
                specific_output = classifiers[task](mid_specific_output)
                old_specific_outputs = [classifiers[task](old_mid_specific_outputs[t]) for t in range(task)]

                # specific_p = [torch.softmax(specific_output, dim=1)]
                specific_p = [torch.softmax(output, dim=1) for output in old_specific_outputs]
                specific_p.append(torch.softmax(specific_output, dim=1))
                label_p = [-nn.NLLLoss(reduction='none')(p, label) for p in specific_p]
            
                if task > 0 and use_dis:
                    with torch.no_grad():
                        temp_dis_score = 0
                        avg_label_p = [torch.mean(label_p[t]) for t in range(task)]
                        for t in range(task):
                            temp_dis_score += label_p[task] - avg_label_p[t]
                        # for t in range(task):
                        #     temp_dis_score += label_p[task] - label_p[t]
                        temp_dis_score = temp_dis_score / task
                        # temp_dis_score = label_p[task]
                        temp_dis_score = torch.exp(-temp_dis_score * 5)
                        temp_dis_score = F.normalize(temp_dis_score, dim=0)
                        dis_score = torch.reshape(temp_dis_score, (data.shape[0], 1))

                if task > 0 and use_dis:
                    # dis_output = dis_score * torch.log(alpha * temp_shared_p + (1 - alpha) * specific_p[task])
                    dis_output = dis_score * torch.log(specific_p[task])
                    output = torch.log(specific_p[task])
                    norm_loss = nn.NLLLoss()(output, label)
                    dis_loss = nn.NLLLoss()(dis_output, label)
                    running_loss += norm_loss.item() + 0.2 * dis_loss.item()
                    total_loss = norm_loss + 0.2 * dis_loss
                else:
                    output = torch.log(specific_p[task])
                    norm_loss = nn.NLLLoss()(output, label)
                    running_loss += norm_loss.item()
                    total_loss = norm_loss             
            
                total_loss.backward()
                classifier_optimizer.step()
                specific_optimizer.step()

            te = time.time()
            total_time += te - ts

            if metrics == 'loss':
                if running_loss < best_loss:
                    best_loss = running_loss
                    current_early_stop = 0
                else:
                    current_early_stop += 1
            
            torch.cuda.empty_cache()

            if current_early_stop > early_stop_threshold:
                break

        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].cpu()
            classifiers[t] = classifiers[t].cpu()

        for t in range(task + 1):
            specific_extractors[t].eval()
            classifiers[t].eval()
        # test the accuracy of all current models
        for t in range(task + 1):
            accuracy_list = []
            for i, (data, label) in enumerate(test_dataloaders[t]):
                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']

                mid_specific_output = specific_extractors[t](torch.column_stack((specific_task_info, data)), specific_task_info)

                specific_output = classifiers[t](mid_specific_output)
                output = torch.softmax(specific_output, dim=1)
                accuracy_list.append(AccuarcyCompute(output, label))
            accuracy_matrix[t, task] = sum(accuracy_list) / len(test_dataloaders[t].dataset)
        
        for t in range(task + 1):
            specific_extractors[t].train()
            classifiers[t].train()


    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'learning_rate', 'weight decay', 'batch size'])
    table.add_row([total_time, '%.4f' %np.mean(accuracy_matrix[:, -1]), learning_rate, '%.4f' %weight_decay, batch_size])
    print(table)
    print('Table for training result')
    print(accuracy_matrix)

    today = datetime.date.today()
    print('===========================================================================')
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(accuracy_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close()    
    return np.mean(accuracy_matrix[:, -1])


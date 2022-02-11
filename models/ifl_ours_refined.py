import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary

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
def DesignedEnsembleDisLabeled(train_datasets, test_datasets, num_classes, word2idx, save_path, alpha=0.2, beta=0.2, gamma=5, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10):

    def MultiClassCrossEntropy(logits, labels, T):
        # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
        labels = Variable(labels.data, requires_grad=False)
        outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
        labels = torch.softmax(labels / T, dim=1)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
        return Variable(outputs.data, requires_grad=True)
    # def MultiClassCrossEntropy(logits, labels, T):
    #     # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    #     labels = Variable(labels.data, requires_grad=False)
    #     outputs = torch.sum(logits * labels, dim=1, keepdim=False)
    #     outputs = -torch.mean(outputs, dim=0, keepdim=False)
    #     return Variable(outputs.data, requires_grad=True)
    # def Ensemble_P(shared_output, specific_output, T):
    #     p = alpha * torch.softmax(shared_output, dim=1) + (1 - alpha) * torch.softmax(specific_output, dim=1)
    #     q = torch.pow(p, 1/T)
    #     q = torch.div(q, torch.sum(q, dim=1, keepdim=True))
    #     return q

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

    # shared_classifier = Specific_Model(classifier_params)
    shared_classifiers = []
    classifiers = []

    accuracy_matrix = np.zeros((total_tasks, total_tasks))
    specific_acc_matrix = np.zeros((total_tasks, total_tasks))
    shared_acc_matrix = np.zeros((total_tasks, total_tasks))
    total_time = 0

    for task in range(total_tasks):
        print('training on task: ', task + 1)
        # print('pos rate: ', pos_rates[task])
        classifier_params = {'num_specific_features': 8, 'num_specific_classes': num_classes[task]}

        classifiers.append(Specific_Model(classifier_params))
        shared_classifiers.append(Specific_Model(classifier_params))

        shared_extractor = shared_extractor.to(device)
        # shared_classifier = shared_classifier.to(device)
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].to(device)
            classifiers[t] = classifiers[t].to(device)
            shared_classifiers[t] = shared_classifiers[t].to(device)

        old_shared_extractor = copy.deepcopy(shared_extractor).to(device)
        # for t in range(task + 1):
        #     old_shared_classifier = copy.deepcopy(shared_classifier).to(device)

        shared_optimizer = torch.optim.Adam(shared_extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # if task > 0:
        #     shared_optimizer = torch.optim.Adam(shared_extractor.parameters(), lr=learning_rate/2, weight_decay=weight_decay)
        specific_optimizer = torch.optim.Adam(specific_extractors[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        shared_classifier_optimizer = torch.optim.Adam(shared_classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        classifier_optimizer = torch.optim.Adam(classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)

        # old_specific_optimizers = []
        # old_classifier_optimizers = []
        old_shared_classifier_optimizers = []

        for old_task in range(task):
            # old_specific_optimizers.append(torch.optim.Adam(specific_extractors[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))
            # old_classifier_optimizers.append(torch.optim.Adam(classifiers[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))
            old_shared_classifier_optimizers.append(torch.optim.Adam(shared_classifiers[old_task].parameters(), lr=learning_rate/2, weight_decay=weight_decay))

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
                specific_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                shared_classifier_optimizer.zero_grad()

                for old_task in range(task):
                    # old_specific_optimizers[old_task].zero_grad()
                    # old_classifier_optimizers[old_task].zero_grad()
                    old_shared_classifier_optimizers[old_task].zero_grad()

                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                old_mid_shared_output = old_shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)

                shared_output = shared_classifiers[task](mid_shared_output)
                # old_shared_outputs = []
                # for old_task in range(task):
                #     old_shared_outputs.append(shared_classifiers[old_task](old_mid_shared_output))
                # old_shared_output = old_shared_classifier(old_mid_shared_output) 

                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                prepared_data = torch.column_stack((specific_task_info, data))
                mid_specific_outputs = []
                
                mid_specific_outputs = [specific_extractors[t](prepared_data, specific_task_info) for t in range(task + 1)]
                specific_outputs = [classifiers[task](output) for output in mid_specific_outputs]

                shared_p = torch.softmax(shared_output, dim=1)
                specific_p = [torch.softmax(output, dim=1) for output in specific_outputs]
                added_p = [(alpha * shared_p + (1 - alpha) * specific_p[t]) for t in range(task + 1)]
                label_p = [-nn.NLLLoss(reduction='none')(p, label) for p in specific_p]

                if task > 0:
                    with torch.no_grad():
                        temp_dis_score = 0
                        # avg_label_p = [torch.mean(label_p[t]) for t in range(task)]
                        # for t in range(task):
                        #     temp_dis_score += label_p[task] - avg_label_p[t]
                        for t in range(task):
                            temp_dis_score += label_p[task] - label_p[t]
                        temp_dis_score = temp_dis_score / task
                        # temp_dis_score = label_p[task]
                        temp_dis_score = torch.exp(-temp_dis_score * gamma)
                        temp_dis_score = F.normalize(temp_dis_score, dim=0)
                        dis_score = torch.reshape(temp_dis_score, (data.shape[0], 1))

                if task > 0:
                    temp_shared_p = Variable(shared_p.data, requires_grad=False).to(device)
                    # dis_output = dis_score * torch.log(alpha * temp_shared_p + (1 - alpha) * specific_p[task])
                    dis_output = dis_score * torch.log(specific_p[task])
                    output = torch.log(added_p[task])
                    norm_loss = nn.NLLLoss()(output, label)
                    dis_loss = nn.NLLLoss()(dis_output, label)
                    running_loss += norm_loss.item() + beta * dis_loss.item()
                    total_loss = norm_loss + beta * dis_loss
                else:
                    output = torch.log(added_p[task])
                    norm_loss = nn.NLLLoss()(output, label)
                    running_loss += norm_loss.item()
                    total_loss = norm_loss             
                # weight = torch.tensor([pos_rates[task], 1 - pos_rates[task]], dtype=torch.float).to(device)

                # if task > 0:
                #     for old_task in range(task):                        
                #         past_specific_output = classifiers[old_task](mid_specific_outputs[old_task])
                #         past_shared_output = shared_classifiers[old_task](mid_shared_output)
                #         old_past_shared_output = shared_classifiers[old_task](old_mid_shared_output)
                #         temp_specific_output = Variable(past_specific_output.data, requires_grad=False).to(device)
                #         norm_q = Ensemble_P(past_shared_output, temp_specific_output, T=T)
                #         new_output = torch.log(norm_q)
                #         old_output = Ensemble_P(old_past_shared_output, temp_specific_output, T=T)
                                            
                #         speci_loss = MultiClassCrossEntropy(new_output, old_output, T=T)
                #         total_loss += speci_loss / task
                #         running_loss += speci_loss.item() / task

                if task > 0:
                    for old_task in range(task):                        
                        past_shared_output = shared_classifiers[old_task](mid_shared_output)
                        old_past_shared_output = shared_classifiers[old_task](old_mid_shared_output)
                                            
                        speci_loss = MultiClassCrossEntropy(past_shared_output, old_past_shared_output, T=T)
                        total_loss += speci_loss
                        running_loss += speci_loss.item()
                
                total_loss.backward()
                classifier_optimizer.step()
                shared_classifier_optimizer.step()
                shared_optimizer.step()
                specific_optimizer.step()
                for old_task in range(task):
                    # old_specific_optimizers[old_task].step()
                    # old_classifier_optimizers[old_task].step()
                    old_shared_classifier_optimizers[old_task].step()
            
            te = time.time()
            total_time += te - ts

            if metrics == 'loss':
                if running_loss < best_loss:
                    best_loss = running_loss
                    current_early_stop = 0
                else:
                    current_early_stop += 1
                metric = running_loss
            
            torch.cuda.empty_cache()

            if current_early_stop > early_stop_threshold:
                # print('early stop triggered')
                # print('Epoch: ', epoch, '\tLoss: ', running_loss, '\tmetric: ', metric)
                break
            if epoch % 15 == 0:
                print('Epoch: ', epoch, '\tLoss: ', running_loss)
                        
        # print('currently total tasks: ', task + 1)

        shared_extractor = shared_extractor.cpu()
        # shared_classifier = shared_classifier.cpu()
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].cpu()
            classifiers[t] = classifiers[t].cpu()
            shared_classifiers[t] = shared_classifiers[t].cpu()

        
        shared_extractor.eval()
        # shared_classifier.eval()
        for t in range(task + 1):
            specific_extractors[t].eval()
            classifiers[t].eval()
            shared_classifiers[t].eval()
        # test the accuracy of all current models
        for t in range(task + 1):
            accuracy_list = []
            specific_acc_list = []
            shared_acc_list = []
            for i, (data, label) in enumerate(test_dataloaders[t]):
                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']
                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']

                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                mid_specific_output = specific_extractors[t](torch.column_stack((specific_task_info, data)), specific_task_info)

                shared_output = shared_classifiers[t](mid_shared_output)
                specific_output = classifiers[t](mid_specific_output)
                shared_p = torch.softmax(shared_output, dim=1)
                specific_p = torch.softmax(specific_output, dim=1)
                output = (alpha * shared_p + (1 - alpha) * specific_p)
                accuracy_list.append(AccuarcyCompute(output, label))
                specific_acc_list.append(AccuarcyCompute(specific_p, label))
                shared_acc_list.append(AccuarcyCompute(shared_p, label))                
            accuracy_matrix[t, task] = sum(accuracy_list) / len(test_dataloaders[t].dataset)
            specific_acc_matrix[t, task] = sum(specific_acc_list) / len(test_dataloaders[t].dataset)
            shared_acc_matrix[t, task] = sum(shared_acc_list) / len(test_dataloaders[t].dataset)

        shared_extractor.train()
        # shared_classifier.train()
        for t in range(task + 1):
            specific_extractors[t].train()
            classifiers[t].train()
            shared_classifiers[t].train()


    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'learning_rate', 'weight decay', 'batch size', 'alpha', 'beta', 'gamma'])
    table.add_row([total_time, '%.4f' %np.mean(accuracy_matrix[:, -1]), learning_rate, '%.4f' %weight_decay, batch_size, '%.4f' %alpha, '%.4f' %beta, gamma])
    print(table)
    print('Table for training result')
    print(accuracy_matrix)

    print('===========================================================================')
    with open(save_path, 'a+') as f:
        f.write(table.get_string())
        f.write('\n')
        f.write('the accuracy matrix is: \nrows for different tasks and columns for accuracy after increment' + '\n')
        f.write(str(accuracy_matrix))
        f.write('\n')
        f.write('pred from specific\n')
        f.write(str(specific_acc_matrix))
        f.write('\n')
        f.write('pred from shared\n')
        f.write(str(shared_acc_matrix))
        f.write('\n')
        f.write('====================================================================\n\n')
        f.close()    
    return np.mean(accuracy_matrix[:, -1])


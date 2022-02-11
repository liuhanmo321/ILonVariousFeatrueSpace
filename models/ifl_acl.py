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
from utils.discriminator import Discriminator
from utils.support_funcs import AccuarcyCompute

import datetime

from prettytable import PrettyTable


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Refined Discriminability. Ensemble model, modified loss function
def BaselineACL(train_datasets, test_datasets, num_classes, word2idx, save_path, epochs=150, batch_size=64, learning_rate=0.005, weight_decay=0.1, source_embeddings=None, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10):

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

    accuracy_matrix = np.zeros((total_tasks, total_tasks))
    total_time = 0

    for task in range(total_tasks):
        print('training on task: ', task + 1)
        # print('pos rate: ', pos_rates[task])
        classifier_params = {'num_specific_features': 16, 'num_specific_classes': num_classes[task]}

        classifiers.append(Specific_Model(classifier_params))

        discriminator_params = {'num_shared_features': 8, 'ntasks': task+1}
        discriminator = Discriminator(discriminator_params).to(device)

        shared_extractor = shared_extractor.to(device)
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].to(device)
            classifiers[t] = classifiers[t].to(device)

        shared_optimizer = torch.optim.Adam(shared_extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        specific_optimizer = torch.optim.Adam(specific_extractors[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        classifier_optimizer = torch.optim.Adam(classifiers[task].parameters(), lr=learning_rate, weight_decay=weight_decay)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

        old_specific_optimizers = []
        old_classifier_optimizers = []

        for old_task in range(task):
            old_specific_optimizers.append(torch.optim.Adam(specific_extractors[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))
            old_classifier_optimizers.append(torch.optim.Adam(classifiers[old_task].parameters(), lr=learning_rate, weight_decay=weight_decay))

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

                t_real_D = (task + 1) * torch.ones_like(label).to(device)
                t_fake_D = torch.zeros_like(label).to(device)
                for _ in range(5):

                    # forward propagation
                    shared_optimizer.zero_grad()
                    specific_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()

                    for old_task in range(task):
                        old_specific_optimizers[old_task].zero_grad()
                        old_classifier_optimizers[old_task].zero_grad()

                    shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                    mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                    
                    specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                    mid_specific_output = specific_extractors[task](torch.column_stack((specific_task_info, data)), specific_task_info)

                    mid_output = torch.cat([mid_shared_output, mid_specific_output], dim=1)

                    output = classifiers[task](mid_output)

                    dis_output = discriminator.forward(mid_shared_output, t_real_D, task)
                    adv_loss= nn.CrossEntropyLoss()(dis_output, t_real_D)

                    diff_loss= DiffLoss()(mid_shared_output, mid_specific_output)

                    norm_loss = nn.CrossEntropyLoss()(output, label)
                    running_loss += norm_loss.item() + 0.05 * adv_loss.item() + 0.1 * diff_loss.item()
                    total_loss = norm_loss +  0.05 * adv_loss + 0.1 * diff_loss            
                    # weight = torch.tensor([pos_rates[task], 1 - pos_rates[task]], dtype=torch.float).to(device)
                
                    total_loss.backward()
                    classifier_optimizer.step()
                    shared_optimizer.step()
                    specific_optimizer.step()

                    for old_task in range(task):
                        old_specific_optimizers[old_task].step()
                        old_classifier_optimizers[old_task].step()
                
                for _ in range(1):
                    discriminator_optimizer.zero_grad()

                    shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64).to(device) + word2idx['task_sh']
                    mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)

                    dis_real_output = discriminator.forward(mid_shared_output.detach(), t_real_D, task)
                    adv_real_loss= nn.CrossEntropyLoss()(dis_real_output, t_real_D)
                    adv_real_loss.backward(retain_graph=True)

                    z_fake = torch.as_tensor(np.random.normal(0.0, 1.0, (data.size(0), discriminator_params['num_shared_features'])),dtype=torch.float32, device=device)
                    dis_fake_output = discriminator.forward(z_fake, t_real_D, task)
                    dis_fake_loss = nn.CrossEntropyLoss()(dis_fake_output, t_fake_D)
                    dis_fake_loss.backward(retain_graph=True)

                    discriminator_optimizer.step()
            
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
                break

        shared_extractor = shared_extractor.cpu()
        for t in range(task + 1):
            specific_extractors[t] = specific_extractors[t].cpu()
            classifiers[t] = classifiers[t].cpu()

        
        shared_extractor.eval()
        # shared_classifier.eval()
        for t in range(task + 1):
            specific_extractors[t].eval()
            classifiers[t].eval()
        # test the accuracy of all current models
        for t in range(task + 1):
            accuracy_list = []
            for i, (data, label) in enumerate(test_dataloaders[t]):
                shared_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']
                specific_task_info = torch.zeros((data.shape[0], 1), dtype=torch.int64) + word2idx['task_sh']

                mid_shared_output = shared_extractor(torch.column_stack((shared_task_info, data)), shared_task_info)
                mid_specific_output = specific_extractors[t](torch.column_stack((specific_task_info, data)), specific_task_info)

                mid_output = torch.cat([mid_shared_output, mid_specific_output], dim=1)

                output = classifiers[t](mid_output)
                output = torch.softmax()
                accuracy_list.append(AccuarcyCompute(output, label))
            accuracy_matrix[t, task] = sum(accuracy_list) / len(test_dataloaders[t].dataset)
        
        shared_extractor.train()
        # shared_classifier.train()
        for t in range(task + 1):
            specific_extractors[t].train()
            classifiers[t].train()


    print('Table for HyperParameters')
    table = PrettyTable(['time', 'avg_acc', 'learning_rate', 'weight decay', 'batch size'])
    table.add_row([total_time, '%.4f' %np.mean(accuracy_matrix[:, -1]), learning_rate, '%.4f' %weight_decay, batch_size])
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
        f.write('====================================================================\n\n')
        f.close()    
    return np.mean(accuracy_matrix[:, -1])

class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
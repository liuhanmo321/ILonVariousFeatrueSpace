import torch
import argparse

torch.backends.cudnn.benchmark = True

import numpy as np
# from PIL import Image
# from tqdm import tqdm
import time
import datetime
import pandas as pd

from models.ifl_ours_refined import DesignedEnsembleDisLabeled
from models.ifl_baselines import BaselineEnsembleFinetune, BaselineEnsembleJoint
from models.ifl_ablations import SharedOnly, SpecificOnly
from models.ifl_acl import BaselineACL
from data_loader import AirSat_data_loader, jannis_data_loader

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

parser = argparse.ArgumentParser(description="Parser")
parser.add_argument('--model', type=str, default='designed', help='choose the model to train')
parser.add_argument('--dataset', type=str, default='aps', help='choose the dataset to train')
parser.add_argument('-dst', type=bool, default=True, help='apply distillation loss')
parser.add_argument('-dis', type=bool, default=True, help='apply discriminative loss')
parser.add_argument('-gpu', type=str, default="6", help='select the GPU')
parser.add_argument('--saved_emb', type=bool, default=True, help='use saved embeddings')
parser.add_argument('--path_note', type=str, default='', help='comments to saving path')
args = parser.parse_args()

today = datetime.date.today()

if args.dataset == 'aps':
    train_datasets, test_datasets, num_classes, word2idx, source_embeddings = AirSat_data_loader('data/AirSat/train.csv', 'data/AirSat/test.csv', num_tasks=4, train_amount=100000)
    if args.saved_emb:
        source_embeddings = torch.load('savings/aps_embeddings.pt')
else:
    train_datasets, test_datasets, num_classes, word2idx, source_embeddings = jannis_data_loader(num_tasks=3, train_amount=30000, n_bins=10)

save_path = 'results/'+str(today)+'_'+args.dataset+'_'+args.model+'_'+args.path_note+'.txt'
if not args.dst:
    print("option for distillation loss ", args.dst)
    save_path = 'results/'+str(today)+'_'+args.dataset+'_'+args.model+'_nodst'+'_'+args.path_note+'.txt'
if not args.dis:
    print("option for discriminative loss ", args.dis)
    save_path = 'results/'+str(today)+'_'+args.dataset+'_'+args.model+'_nodis'+'_'+args.path_note+'.txt'
# print(save_path)

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

def run_model(params):
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    final_avg_acc = DesignedEnsembleDisLabeled(train_datasets, test_datasets, num_classes, word2idx, save_path=save_path, alpha=alpha, beta=beta, gamma=gamma, epochs=300, batch_size=batch_size, learning_rate=lr, weight_decay=weight_decay, source_embeddings=source_embeddings, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10)

    return final_avg_acc

def run_joint(params):
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    alpha = params["alpha"]

    final_avg_acc = BaselineEnsembleJoint(train_datasets, test_datasets, num_classes, word2idx, save_path=save_path, alpha=alpha, epochs=300, batch_size=batch_size, learning_rate=lr, weight_decay=weight_decay, source_embeddings=source_embeddings, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10)
    
    return final_avg_acc

def run_finetune(params):
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    alpha = params["alpha"]

    final_avg_acc = BaselineEnsembleFinetune(train_datasets, test_datasets, num_classes, word2idx, save_path=save_path, alpha=alpha, epochs=300, batch_size=batch_size, learning_rate=lr, weight_decay=weight_decay, source_embeddings=source_embeddings, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10)

    return final_avg_acc

def run_shared(params):
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    final_avg_acc = SharedOnly(train_datasets, test_datasets, num_classes, word2idx, save_path=save_path, use_dst=args.dst, epochs=300, batch_size=batch_size, learning_rate=lr, weight_decay=weight_decay, source_embeddings=source_embeddings, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10)

    return final_avg_acc

def run_specific(params):
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    final_avg_acc = SpecificOnly(train_datasets, test_datasets, num_classes, word2idx, save_path=save_path, use_dis=True, epochs=300, batch_size=batch_size, learning_rate=lr, weight_decay=weight_decay, source_embeddings=source_embeddings, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10)

    return final_avg_acc

def run_acl(params):
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]

    final_avg_acc = BaselineACL(train_datasets, test_datasets, num_classes, word2idx, save_path=save_path, epochs=300, batch_size=batch_size, learning_rate=lr, weight_decay=weight_decay, source_embeddings=source_embeddings, total_tasks=3, device=device, T=2, metrics='loss', early_stop_threshold=10)

    return final_avg_acc

if __name__ == '__main__':

    space = {
            # "lr": hp.choice("lr", [0.0005, 0.0001, 0.00005, 0.00001]),
            "lr": hp.choice("lr", [0.00005, 0.00001]),
            # "weight_decay": hp.uniform("weight_decay", 0, 0.1),
            "weight_decay": hp.choice("weight_decay", [0]),
            "batch_size": hp.choice("batch_size", [256, 512]),
            "alpha": hp.uniform("alpha", 0.1, 0.5),
            # "alpha": hp.choice("alpha", [0.1, 0.2, 0.3, 0.4, 0.5]),
            # "alpha": hp.choice("alpha", [0.13]),
            # "beta": hp.uniform("beta", 0.05, 0.5),
            "beta": hp.choice("beta", [0.33]),
            "gamma": hp.choice("gamma", [5, 10, 15])
    }

    def f(params):
        if args.model == 'designed':
            acc = run_model(params)
            return {'loss': -acc, 'status': STATUS_OK}
        if args.model == 'joint':
            acc = run_joint(params)
            return {'loss': -acc, 'status': STATUS_OK}
        if args.model == 'finetune':
            acc = run_finetune(params)
            return {'loss': -acc, 'status': STATUS_OK}
        if args.model == 'shared':
            acc = run_shared(params)
            return {'loss': -acc, 'status': STATUS_OK}
        if args.model == 'specific':
            acc = run_specific(params)
            return {'loss': -acc, 'status': STATUS_OK}
        if args.model == 'acl':
            acc = run_specific(params)
            return {'loss': -acc, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=20, trials=trials)
    
    print('best performance:', best)     

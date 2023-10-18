#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import nn
from torch.nn import init, MarginRankingLoss
from transformers import BertModel, RobertaModel
from transformers import BertTokenizer, RobertaTokenizer
from torch.optim import Adam
from distutils.version import LooseVersion
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.autograd import Variable
from transformers import AutoConfig, AutoModel, AutoTokenizer
import nltk
import re
import Levenshtein
import spacy
import en_core_web_sm
import torch.optim as optim
from torch.distributions import Categorical
from numpy import linalg as LA
from transformers import AutoModelForMaskedLM
from nltk.corpus import wordnet
import torch.nn.functional as F
import random
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import words as wal
from sklearn.utils import resample


# In[56]:


class MyDataset(Dataset):
    def __init__(self,file_name):
        df1 = pd.read_csv(file_name)
        df1 = df1[230000:]
        df1 = df1.fillna("")
        res = df1['X']
#         ab = df1['X']
#         res = [sub.replace("<mask>", "[MASK]") for sub in ab]
        self.X_list = res.to_numpy()
        self.y_list = df1['y'].to_numpy()
    def __len__(self):
        return len(self.X_list)
    def __getitem__(self,idx):
        mapi = []
        mapi.append(self.X_list[idx])
        mapi.append(self.y_list[idx])
        return mapi


# In[59]:


class Step1_model(nn.Module):
    def __init__(self, hidden_size=512):
        super(Step1_model, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    def forward(self, mapi):
        y = mapi[1]
        print(y)
        nl = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        lb = ' '.join(nl).lower()
        x = tokenizer.tokenize(lb)
        nlab = len(x)
        print(nlab)
        rand_no = random.random()
        tok_map = {2: 0.4363429005892416,
                   1: 0.6672580202327398,
                   4: 0.7476060740459144,
                   3: 0.9618703668504087,
                   6: 0.9701028532809564,
                   7: 0.9729244545819342,
                   8: 0.9739508754144756,
                   5: 0.9994508859743607,
                   9: 0.9997507867114407,
                   10: 0.9999112969650892,
                   11: 0.9999788802297832,
                   0: 0.9999831041838266,
                   12: 0.9999873281378701,
                   22: 0.9999957760459568,
                   14: 1.0000000000000002}
        for key in tok_map.keys():
            if rand_no < tok_map[key]:
                pred = key
                break
        predicted = torch.tensor([pred], dtype = float) 
        if pred == nlab:
            l2 = 0
        else:
            l2 = 1
        actual = torch.tensor([nlab], dtype = float) 
        l1 = Variable(torch.tensor([(actual-predicted)**2],dtype=float),requires_grad = True)
        return {'loss':l1, 'actual_pred':pred, 'acc': l2}


# In[60]:


epoch_number = 0
EPOCHS = 5
run_int = 0
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = Step1_model()
myDs=MyDataset('dat_test.csv')
train_loader=DataLoader(myDs,batch_size=2,shuffle=True)
best_loss = torch.full((1,), fill_value=100000)


flag = 0
def train_one_epoch(transformer_model, dataset):
    global flag
    tot_loss1 = 0.0
    tot_loss2 = 0.0
    cnt = 0
    for batch in dataset:
        p = 0
        inputs = batch
        for i in range(len(inputs[0])):
            cnt += 1
            l = []
            l.append(inputs[0][i])
            l.append(inputs[1][i])
            opi = transformer_model(l)
            loss1 = opi['loss']
            loss2 = opi['acc']
            tot_loss1 += loss1
            tot_loss2 += loss2

    tot_loss1/=cnt
    tot_loss2/=cnt
    print('MSE  loss: ')
    print(tot_loss1)
    print('accuracy: ')
    print(tot_loss2)
    return {'MSE loss': tot_loss1, 'accuracy': tot_loss2}

model.eval()
avg_loss = train_one_epoch(model,train_loader)







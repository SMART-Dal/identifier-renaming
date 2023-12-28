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
        self.model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=6)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.config = AutoConfig.from_pretrained("microsoft/graphcodebert-base")

    def forward(self, mapi):
        X_init = mapi[0]
        X_init = X_init.replace("[MASK]", " ".join([tokenizer.mask_token] * 1))
        y = mapi[1]
        y = y.rstrip("\n")
        print(y)
        nl = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        lb = ' '.join(nl).lower()
        x = tokenizer.tokenize(lb)
        nlab = len(x)
        print(nlab)
        tokens = self.tokenizer.encode_plus(X_init, add_special_tokens=False,return_tensors='pt')
        input_id_chunki = tokens['input_ids'][0].split(510)
        input_id_chunks = []
        mask_chunks  = []
        mask_chunki = tokens['attention_mask'][0].split(510)
        for tensor in input_id_chunki:
            input_id_chunks.append(tensor)
        for tensor in mask_chunki:
            mask_chunks.append(tensor)
        xi = torch.full((1,), fill_value=101)
        yi = torch.full((1,), fill_value=1)
        zi = torch.full((1,), fill_value=102)
        for r in range(len(input_id_chunks)):
            input_id_chunks[r] = torch.cat([xi, input_id_chunks[r]],dim = -1)
            input_id_chunks[r] = torch.cat([input_id_chunks[r],zi],dim=-1)
            mask_chunks[r] = torch.cat([yi, mask_chunks[r]],dim=-1)
            mask_chunks[r] = torch.cat([mask_chunks[r],yi],dim=-1)
        di = torch.full((1,), fill_value=0)
        for i in range(len(input_id_chunks)):
            pad_len = 512 - input_id_chunks[i].shape[0]
            if pad_len > 0:
                for p in range(pad_len):
                    input_id_chunks[i] = torch.cat([input_id_chunks[i],di],dim=-1)
                    mask_chunks[i] = torch.cat([mask_chunks[i],di],dim=-1)
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)
        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int()
        }
        with torch.no_grad():
            outputs = self.model(**input_dict)
        last_hidden_state = outputs.logits.squeeze()
        lhs_agg = []
        if len(last_hidden_state) == 1:
            lhs_agg.append(last_hidden_state)
        else:
            for p in range(len(last_hidden_state)):
                lhs_agg.append(last_hidden_state[p])
        lhs = lhs_agg[0]
        for i in range(len(lhs_agg)):
            if i == 0:
                continue
            lhs+=lhs_agg[i]
        lhs/=len(lhs_agg)
        print(lhs)
        predicted_prob = torch.softmax(lhs, dim=0)
        if nlab > 6:
            nlab = 6
        pll = -1*torch.log(predicted_prob[nlab-1])
        
        pred = torch.argmax(predicted_prob).item()
        pred+=1
        print(pred)
        predicted = torch.tensor([pred], dtype = float) 
        if pred == nlab:
            l2 = 1
        else:
            l2 = 0
        actual = torch.tensor([nlab], dtype = float) 
        l1 = Variable(torch.tensor([(actual-predicted)**2],dtype=float),requires_grad = True)
        return {'loss1':l1, 'loss2':l2}


# In[60]:


epoch_number = 0
EPOCHS = 5
run_int = 0
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = Step1_model()
myDs=MyDataset('dat_test.csv')
train_loader=DataLoader(myDs,batch_size=2,shuffle=True)
best_loss = torch.full((1,), fill_value=100000)


# In[61]:


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
            loss1 = opi['loss1']
            loss2 = opi['loss2']
            tot_loss1 += loss1
            tot_loss2 += loss2

    tot_loss1/=cnt
    tot_loss2/=cnt
    print('MSE: ')
    print(tot_loss1)
    print('Acc: ',tot_loss2)
    return {'tot loss1': tot_loss1,'tot_loss2':tot_loss2}


# In[62]:

model.eval()
avg_loss = train_one_epoch(model,train_loader)




# In[ ]:





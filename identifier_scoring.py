#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init, MarginRankingLoss
from transformers import BertModel, RobertaModel
from transformers import BertTokenizer, RobertaTokenizer
from torch.optim import Adam
from distutils.version import LooseVersion
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.autograd import Variable
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch.optim as optim
from torch.distributions import Categorical
import random
from transformers import AutoModelForMaskedLM, BertForMaskedLM, AdamW
from transformers import BertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import XLMRobertaTokenizer
import os
import csv
from sklearn.model_selection import train_test_split
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import math
from nltk.corpus import words
from sklearn.model_selection import train_test_split
import random
import re
import random


# In[2]:


class MyDataset(Dataset):
    def __init__(self,file_name):
        df1 = pd.read_csv(file_name)
        df1 = df1[200:300]
        df1 = df1.fillna("")
        res = df1['X'].to_numpy()
        self.X_list = res
        self.y_list = df1['y'].to_numpy()
    def __len__(self):
        return len(self.X_list)
    def __getitem__(self,idx):
        mapi = []
        mapi.append(self.X_list[idx])
        mapi.append(self.y_list[idx])
        return mapi


# In[3]:


class Step1_model(nn.Module):
    def __init__(self, hidden_size=512):
        super(Step1_model, self).__init__()
        self.hidden_size = hidden_size
        self.model = AutoModelForMaskedLM.from_pretrained('microsoft/graphcodebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.config = AutoConfig.from_pretrained("microsoft/graphcodebert-base")
        self.linear_layer = nn.Linear(self.model.config.vocab_size, self.model.config.vocab_size)
    def foo (self,data):
        result = []
        if type(data) == tuple:
            return data[1]
        if type(data) == list:
            for inner in data:
                result.append(foo(inner))
        res = []
        for a in result[0]:
            res.append(a[:2])
        return res
    def loss_func1(self, word, y):
        if word =='NA':
            return torch.full((1,), fill_value=100)
        try:
            pred_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', word)
            target_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
            pred_tag = self.foo(nltk.pos_tag(pred_list))
            target_tag = self.foo(nltk.pos_tag(target_list))
            str1 = ' '.join(pred_tag)  # Convert lists to strings
            str2 = ' '.join(target_tag)
            distance = Levenshtein.distance(str1, str2)
            dist = torch.Tensor([distance])
        except:
            dist = torch.Tensor([2*len(target_list)])
        return dist
    def loss_func2(self, word, y):
        if word =='NA':
            return  torch.full((1,), fill_value=100)
        nlp = en_core_web_sm.load()
        pred_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', word)
        target_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        try:
            str1 = ' '.join(pred_list)  # Convert lists to strings
            str2 = ' '.join(target_list)
            tokens1 = nlp(str1)
            tokens2 = nlp(str2)
            embedding1 = sum(token.vector for token in tokens1) / len(tokens1)
            embedding2 = sum(token.vector for token in tokens2) / len(tokens2)
            w1= LA.norm(embedding1)
            w2= LA.norm(embedding2)
            distance = 1 - (embedding1.dot(embedding2) / (w1 * w2))
            dist = torch.Tensor([distance])
        except:
            dist = torch.Tensor([1])
        return dist
    def forward(self, mapi):
        global variable_names
        global base_model
        global tot_pll
        global base_tot_pll
        X_init1 = mapi[0]
        X_init = mapi[0]
        y = mapi[1]
        print(y)
        y_tok = self.tokenizer.encode(y)[1:-1]
        nl = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        lb = ' '.join(nl).lower()
        x = self.tokenizer.tokenize(lb)
        num_sub_tokens_label = len(x)
        X_init = X_init.replace("[MASK]", " ".join([self.tokenizer.mask_token] * num_sub_tokens_label))
        sent_pll = 0.0
        base_sent_pll = 0.0
        for m in range(num_sub_tokens_label):
            print(m)
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
            maski = []
            u = 0
            ad = 0
            for l in range(len(input_dict['input_ids'])):
                masked_pos = []
                for i in range(len(input_dict['input_ids'][l])):
                    if input_dict['input_ids'][l][i] == 50264: #103
                        u+=1
                        if i != 0 and input_dict['input_ids'][l][i-1] == 50264:
                            continue
                        masked_pos.append(i)
                        ad+=1
                maski.append(masked_pos)
            print('number of mask tok',u)
            print('number of seq', ad)
            with torch.no_grad():
                output = self.model(**input_dict)
                base_output = base_model(**input_dict)
            last_hidden_state = output[0].squeeze()
            base_last_hidden_state = base_output[0].squeeze()
            l_o_l_sa = []
            base_l_o_l_sa = []
            if len(maski) == 1:
                masked_pos = maski[0]
                for k in masked_pos:
                    l_o_l_sa.append(last_hidden_state[k])
                    base_l_o_l_sa.append(base_last_hidden_state[k])
            else:
                for p in range(len(maski)):
                    masked_pos = maski[p]
                    for k in masked_pos:
                        l_o_l_sa.append(last_hidden_state[p][k])
                        base_l_o_l_sa.append(base_last_hidden_state[p][k])
            sum_state = l_o_l_sa[0]
            base_sum_state = base_l_o_l_sa[0]
            for i in range(len(l_o_l_sa)):
                if i == 0:
                    continue
                sum_state += l_o_l_sa[i]
                base_sum_state += base_l_o_l_sa[i]
            yip = len(l_o_l_sa)
            sum_state /= yip
            base_sum_state /= yip
            probs = F.softmax(sum_state, dim=0)
            base_probs = F.softmax(base_sum_state, dim=0)
            a_lab = y_tok[m]
            prob = probs[a_lab]
            base_prob = base_probs[a_lab]
            log_prob = -1*math.log(prob)
            base_log_prob = -1*math.log(base_prob)
            sent_pll+=log_prob
            base_sent_pll+=base_log_prob
            xl = X_init.split()
            xxl = []
            for p in range(len(xl)):
                if xl[p] == self.tokenizer.mask_token:
                    if p != 0 and xl[p-1] == self.tokenizer.mask_token:
                        xxl.append(xl[p])
                        continue
                    xxl.append(self.tokenizer.convert_ids_to_tokens(y_tok[m]))
                    continue
                xxl.append(xl[p])
            X_init = " ".join(xxl)
        sent_pll/=num_sub_tokens_label
        base_sent_pll/=num_sub_tokens_label
        print("Sent PLL:")
        print(sent_pll)
        print("Base Sent PLL:")
        print(base_sent_pll)
        print("Net % difference:")
        diff = (sent_pll-base_sent_pll)*100/base_sent_pll
        print(diff)
        tot_pll += sent_pll
        base_tot_pll+=base_sent_pll
        print()
        print()
        y = random.choice(variable_names)
        print(y)
        X_init = X_init1
        y_tok = self.tokenizer.encode(y)[1:-1]
        nl = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        lb = ' '.join(nl).lower()
        x = self.tokenizer.tokenize(lb)
        num_sub_tokens_label = len(x)
        X_init = X_init.replace("[MASK]", " ".join([self.tokenizer.mask_token] * num_sub_tokens_label))
        sent_pll = 0.0
        base_sent_pll = 0.0
        for m in range(num_sub_tokens_label):
            print(m)
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
            maski = []
            u = 0
            ad = 0
            for l in range(len(input_dict['input_ids'])):
                masked_pos = []
                for i in range(len(input_dict['input_ids'][l])):
                    if input_dict['input_ids'][l][i] == 50264: #103
                        u+=1
                        if i != 0 and input_dict['input_ids'][l][i-1] == 50264:
                            continue
                        masked_pos.append(i)
                        ad+=1
                maski.append(masked_pos)
            print('number of mask tok',u)
            print('number of seq', ad)
            with torch.no_grad():
                output = self.model(**input_dict)
                base_output = base_model(**input_dict)
            last_hidden_state = output[0].squeeze()
            base_last_hidden_state = base_output[0].squeeze()
            l_o_l_sa = []
            base_l_o_l_sa = []
            if len(maski) == 1:
                masked_pos = maski[0]
                for k in masked_pos:
                    l_o_l_sa.append(last_hidden_state[k])
                    base_l_o_l_sa.append(base_last_hidden_state[k])
            else:
                for p in range(len(maski)):
                    masked_pos = maski[p]
                    for k in masked_pos:
                        l_o_l_sa.append(last_hidden_state[p][k])
                        base_l_o_l_sa.append(base_last_hidden_state[p][k])
            sum_state = l_o_l_sa[0]
            base_sum_state = base_l_o_l_sa[0]
            for i in range(len(l_o_l_sa)):
                if i == 0:
                    continue
                sum_state += l_o_l_sa[i]
                base_sum_state += base_l_o_l_sa[i]
            yip = len(l_o_l_sa)
            sum_state /= yip
            base_sum_state /= yip
            probs = F.softmax(sum_state, dim=0)
            base_probs = F.softmax(base_sum_state, dim=0)
            a_lab = y_tok[m]
            prob = probs[a_lab]
            base_prob = base_probs[a_lab]
            log_prob = -1*math.log(prob)
            base_log_prob = -1*math.log(base_prob)
            sent_pll+=log_prob
            base_sent_pll+=base_log_prob
            xl = X_init.split()
            xxl = []
            for p in range(len(xl)):
                if xl[p] == self.tokenizer.mask_token:
                    if p != 0 and xl[p-1] == self.tokenizer.mask_token:
                        xxl.append(xl[p])
                        continue
                    xxl.append(self.tokenizer.convert_ids_to_tokens(y_tok[m]))
                    continue
                xxl.append(xl[p])
            X_init = " ".join(xxl)
        sent_pll/=num_sub_tokens_label
        base_sent_pll/=num_sub_tokens_label
        print("Sent PLL:")
        print(sent_pll)
        print("Base Sent PLL:")
        print(base_sent_pll)
        print("Net % difference:")
        diff = (sent_pll-base_sent_pll)*100/base_sent_pll
        print(diff)
        print()
        print("******")
        print()
    


# In[4]:


tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = Step1_model()
model.load_state_dict(torch.load('var_runs/model_98_3'))
base_model = AutoModelForMaskedLM.from_pretrained('microsoft/graphcodebert-base')
model.eval()
base_model.eval()


# In[5]:


myDs=MyDataset('dat.csv')
loader=DataLoader(myDs,batch_size=2,shuffle=True)
loop = tqdm(loader, leave=True)


# In[6]:


tot_pll = 0.0
base_tot_pll = 0.0
variable_names = [
    'x', 'y', 'myVariable', 'dataPoint', 'randomNumber', 'userAge', 'resultValue', 'inputValue', 'tempValue', 'indexCounter', 
    'itemPrice', 'userName', 'testScore', 'acceleration', 'productCount', 'errorMargin', 'piValue', 'sensorReading', 
    'currentTemperature', 'velocityVector', 'variable1', 'variable2', 'valueA', 'valueB', 'counter', 'flag', 'total', 
    'average', 'valueX', 'valueY', 'valueZ', 'price', 'quantity', 'name', 'age', 'score', 'weight', 'height', 'distance', 
    'time', 'radius', 'width', 'length', 'temperature', 'pressure', 'humidity', 'voltage', 'current', 'resistance'
]

for batch in loop:
    inputs = batch
    try:
        for i in range(len(inputs[0])):
            l = []
            l.append(inputs[0][i])
            l.append(inputs[1][i])
            model(l)
    except:
        continue

tot_pll/=len(myDs)
print('Total PLL per sentence: ')
print(tot_pll)
base_tot_pll/=len(myDs)
print('Total Base PLL per sentence: ')
print(base_tot_pll)
print("Net % difference average:")
tot_diff = (tot_pll-base_tot_pll)*100/base_tot_pll
print(tot_diff)
  


# In[ ]:





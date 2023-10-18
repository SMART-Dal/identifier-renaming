#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import init, MarginRankingLoss
from torch.optim import Adam
from distutils.version import LooseVersion
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
from transformers import AutoConfig, AutoModel, AutoTokenizer
import nltk
import re
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F
import random


# In[2]:


maskis = []
n_y = []
class MyDataset(Dataset):
    def __init__(self,file_name):
        global maskis
        global n_y
        df = pd.read_csv(file_name)
        df = df.fillna("")
        self.inp_dicts = []
        for r in range(df.shape[0]):
            X_init = df['X'][r]
            y = df['y'][r]
            n_y.append(y)
            nl = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
            lb = ' '.join(nl).lower()
            x = tokenizer.tokenize(lb)
            num_sub_tokens_label = len(x)
            X_init = X_init.replace("[MASK]", " ".join([tokenizer.mask_token] * num_sub_tokens_label))
            tokens = tokenizer.encode_plus(X_init, add_special_tokens=False,return_tensors='pt')
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
            vb = torch.ones_like(input_id_chunks[0])
            fg = torch.zeros_like(input_id_chunks[0])
            maski = []
            for l in range(len(input_id_chunks)):
                masked_pos = []
                for i in range(len(input_id_chunks[l])):
                    if input_id_chunks[l][i] == tokenizer.mask_token_id: #103
                        if i != 0 and input_id_chunks[l][i-1] == tokenizer.mask_token_id:
                            continue
                        masked_pos.append(i)
                maski.append(masked_pos)
            maskis.append(maski)
            while (len(input_id_chunks)<250):
                input_id_chunks.append(vb)
                mask_chunks.append(fg)
            input_ids = torch.stack(input_id_chunks)
            attention_mask = torch.stack(mask_chunks)
            input_dict = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.int()
            }
            self.inp_dicts.append(input_dict)
            del input_dict
            del input_ids
            del attention_mask
            del maski
            del mask_chunks
            del input_id_chunks
            del di
            del fg
            del vb
            del mask_chunki
            del input_id_chunki
            del X_init
            del y
            del tokens
            del x
            del lb
            del nl
        del df
    def __len__(self):
        return len(self.inp_dicts)
    def __getitem__(self,idx):
        return self.inp_dicts[idx]


# In[3]:


tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
base_model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
model.load_state_dict(torch.load('var_runs/model_26_2'))
model.eval()
base_model.eval()
myDs=MyDataset('d_t.csv') 
train_loader=DataLoader(myDs,batch_size=1,shuffle=False)


# In[4]:


variable_names = [
    # One-word Variable Names
    'count', 'value', 'result', 'flag', 'max', 'min', 'data', 'input', 'output', 'name', 'index', 'status', 'error', 'message', 'price', 'quantity', 'total', 'length', 'size', 'score',

    # Two-word Variable Names
    'studentName', 'accountBalance', 'isFound', 'maxScore', 'userAge', 'carModel', 'bookTitle', 'arrayLength', 'employeeID', 'itemPrice', 'customerAddress', 'productCategory', 'orderNumber', 'transactionType', 'bankAccount', 'shippingMethod', 'deliveryDate', 'purchaseAmount', 'inventoryItem', 'salesRevenue',

    # Three-word Variable Names
    'numberOfStudents', 'averageTemperature', 'userIsLoggedIn', 'totalSalesAmount', 'employeeSalaryRate', 'maxAllowedAttempts', 'selectedOption', 'shippingAddress', 'manufacturingDate', 'connectionPool', 'customerAccountBalance', 'employeeSalaryReport', 'productInventoryCount', 'transactionProcessingStatus', 'userAuthenticationToken', 'orderShippingAddress', 'databaseConnectionPoolSize', 'vehicleEngineTemperature', 'sensorDataProcessingRate', 'employeePayrollSystem',

    # Four-word Variable Names
    'customerAccountBalanceValue', 'employeeSalaryReportData', 'productInventoryItemCount', 'transactionProcessingStatusFlag', 'userAuthenticationTokenKey', 'orderShippingAddressDetails', 'databaseConnectionPoolMaxSize', 'vehicleEngineTemperatureReading', 'sensorDataProcessingRateLimit', 'employeePayrollSystemData', 'customerOrderShippingAddress', 'productCatalogItemNumber', 'transactionProcessingSuccessFlag', 'userAuthenticationAccessToken', 'databaseConnectionPoolConfig', 'vehicleEngineTemperatureSensor', 'sensorDataProcessingRateLimitation', 'employeePayrollSystemConfiguration', 'customerAccountBalanceHistoryData', 'transactionProcessingStatusTracking'
]
var_list = []
for j in range(6):
    d =[]
    var_list.append(d)
for var in variable_names:
    try:
        var_list[len(tokenizer.tokenize(var))-1].append(var)
    except:
        continue


# In[5]:


tot_pll = 0.0
base_tot_pll = 0.0
loop = tqdm(train_loader, leave=True)
cntr = 0
for batch in loop:
    maxi = torch.tensor(0.0, requires_grad=True)
    for i in range(len(batch['input_ids'])):            
        cntr+=1
        maski = maskis[cntr-1]
        li = len(maski)
        input_ids = batch['input_ids'][i][:li]
        att_mask = batch['attention_mask'][i][:li]
        y = n_y[cntr-1]
        ty = tokenizer.encode(y)[1:-1]
        num_sub_tokens_label = len(ty)
        if num_sub_tokens_label > 6:
            continue
        print("Ground truth:", y)
        m_y = random.choice(var_list[num_sub_tokens_label-1])
        m_ty = tokenizer.encode(m_y)[1:-1]
        print("Mock truth:", m_y)
#            input_ids, att_mask = input_ids.to(device),att_mask.to(device)
        outputs = model(input_ids, attention_mask = att_mask)
        base_outputs = base_model(input_ids, attention_mask = att_mask)
        last_hidden_state = outputs[0].squeeze()
        base_last_hidden_state = base_outputs[0].squeeze()
        l_o_l_sa = []
        base_l_o_l_sa = []
        sum_state = []
        base_sum_state = []
        for t in range(num_sub_tokens_label):
            c = []
            d = []
            l_o_l_sa.append(c)
            base_l_o_l_sa.append(d)
        if len(maski) == 1:
            masked_pos = maski[0]
            for k in masked_pos:
                for t in range(num_sub_tokens_label):
                    l_o_l_sa[t].append(last_hidden_state[k+t])
                    base_l_o_l_sa[t].append(base_last_hidden_state[k+t])
        else:
            for p in range(len(maski)):
                masked_pos = maski[p]
                for k in masked_pos:
                    for t in range(num_sub_tokens_label):
                        if (k+t) >= len(last_hidden_state[p]):
                            l_o_l_sa[t].append(last_hidden_state[p+1][k+t-len(last_hidden_state[p])])
                            base_l_o_l_sa[t].append(base_last_hidden_state[p+1][k+t-len(base_last_hidden_state[p])])
                            continue
                        l_o_l_sa[t].append(last_hidden_state[p][k+t])
                        base_l_o_l_sa[t].append(base_last_hidden_state[p][k+t])
        for t in range(num_sub_tokens_label):
            sum_state.append(l_o_l_sa[t][0])
            base_sum_state.append(base_l_o_l_sa[t][0])
        for i in range(len(l_o_l_sa[0])):
            if i == 0:
                continue
            for t in range(num_sub_tokens_label):
                sum_state[t] = sum_state[t] + l_o_l_sa[t][i]
                base_sum_state[t] = base_sum_state[t] + base_l_o_l_sa[t][i]
        yip = len(l_o_l_sa[0])
        val = 0.0
        m_val = 0.0
        m_base_val = 0.0
        base_val = 0.0
        for t in range(num_sub_tokens_label):
            sum_state[t] /= yip
            base_sum_state[t] /= yip
            probs = F.softmax(sum_state[t], dim=0)
            base_probs = F.softmax(base_sum_state[t], dim=0)
            val = val - torch.log(probs[ty[t]])
            m_val = m_val - torch.log(probs[m_ty[t]])
            base_val = base_val - torch.log(base_probs[ty[t]])
            m_base_val = m_base_val - torch.log(base_probs[m_ty[t]])
        val = val / num_sub_tokens_label
        base_val = base_val / num_sub_tokens_label
        m_val = m_val / num_sub_tokens_label
        m_base_val = m_base_val / num_sub_tokens_label
        print("Sent PLL:")
        print(val)
        print("Base Sent PLL:")
        print(base_val)
        print("Net % difference:")
        diff = (val-base_val)*100/base_val
        print(diff)
        tot_pll += val
        base_tot_pll+=base_val
        print()
        print()
        print("Mock Sent PLL:")
        print(m_val)
        print("Mock Base Sent PLL:")
        print(m_base_val)
        print("Mock Net % difference:")
        m_diff = (m_val-m_base_val)*100/m_base_val
        print(m_diff)
        for c in sum_state:
            del c
        for d in base_sum_state:
            del d
        del sum_state
        del base_sum_state
        for c in l_o_l_sa:
            del c
        for c in base_l_o_l_sa:
            del c
        del l_o_l_sa
        del base_l_o_l_sa
        del maski
        del input_ids
        del att_mask
        del last_hidden_state
        del base_last_hidden_state
print("Tot PLL: ", tot_pll)
print("Base Tot PLL: ", base_tot_pll)


# In[ ]:





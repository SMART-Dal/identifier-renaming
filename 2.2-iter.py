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

def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
        if name.startswith("model.roberta.encoder.layer.0"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.1"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.2"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.3"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.4"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.5"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.6"):
            param.requires_grad = False
        if name.startswith("model.roberta.encoder.layer.7"):
            param.requires_grad = False
   #     if name.startswith("model.roberta.encoder.layer.8"):
  #          param.requires_grad = False
#        if name.startswith("model.roberta.encoder.layer.9"):
 #           param.requires_grad = False
    return model

maskis = []
n_y = []
class MyDataset(Dataset):
    def __init__(self,file_name):
        global maskis
        global n_y
        df = pd.read_csv(file_name)
        df = df.sample(frac=1)
        df = df.fillna("")
        self.inp_dicts = []
        tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
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


def my_func():
    global maskis
    global n_y
    epoch_number = 0
    EPOCHS = 5
    run_int = 26
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
#    model = model.half()
#    model = freeze(model)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    myDs=MyDataset('dat1.csv') 
    train_loader=DataLoader(myDs,batch_size=1,shuffle=False)
    best_loss = torch.full((1,), fill_value=100000)
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=True)
        tot_loss = 0.0
        cntr = 0
        for batch in loop:
            try:
                optimizer.zero_grad()
                maxi = torch.tensor(0.0, requires_grad=True)
                for i in range(len(batch['input_ids'])):            
                    cntr+=1
                    maski = maskis[cntr-1]
                    li = len(maski)
                    input_ids = batch['input_ids'][i][:li]
                    att_mask = batch['attention_mask'][i][:li]
                    y = n_y[cntr-1]
                    print("Ground truth:", y)
                    ty = tokenizer.encode(y)[1:-1]
                    num_sub_tokens_label = len(ty)
                    qw = ""
                    val = torch.tensor(0.0, requires_grad=True)
                    for m in range(num_sub_tokens_label):
                        outputs = model(input_ids, attention_mask = att_mask)
                        last_hidden_state = outputs[0].squeeze()
                        l_o_l_sa = []
                        sum_state = []
                        if len(maski) == 1:
                            masked_pos = maski[0]
                            for k in masked_pos:
                                l_o_l_sa.append(last_hidden_state[k+m])
                        else:
                            for p in range(len(maski)):
                                masked_pos = maski[p]
                                for k in masked_pos:
                                    if (k+m) >= len(last_hidden_state[p]):
                                        l_o_l_sa.append(last_hidden_state[p+1][k+m-len(last_hidden_state[p])])
                                        continue
                                    l_o_l_sa.append(last_hidden_state[p][k+m])
                        sum_state = l_o_l_sa[0]
                        for i in range(len(l_o_l_sa)):
                            if i == 0:
                                continue
                            sum_state = sum_state + l_o_l_sa[i]
                        yip = len(l_o_l_sa)
                        sum_state = sum_state / yip
                        probs = F.softmax(sum_state, dim=0)
                        val = val - torch.log(probs[ty[m]])
                        idx = torch.topk(sum_state, k=5, dim=0)[1]
                        wor= [tokenizer.decode(i.item()).strip() for i in idx]  
                        for kl in wor:
                            if all(char.isalpha() for char in kl):
                                qw+=kl.capitalize()
                                break 
                        des = input_ids.clone()
                        if len(maski) == 1:
                            masked_pos = maski[0]
                            for k in masked_pos:
                                des[k+m] = idx[0]
                        else:
                            for p in range(len(maski)):
                                masked_pos = maski[p]
                                for k in masked_pos:
                                    if (k+m) >= len(des[p]):
                                        des[p+1][k+m-len(des[p])] = idx[0]
                                        continue
                                    des[p][k+m] = idx[0]
                        del input_ids
                        input_ids = des
                        for c in sum_state:
                            del c
                        del sum_state
                        for c in l_o_l_sa:
                            del c
                        del l_o_l_sa
                        del last_hidden_state
                    val = val / num_sub_tokens_label
                    maxi = maxi + val
                    print("Prediction: ", qw)
                    print("*****")
                    del maski
                    del input_ids
                    del att_mask
                    del qw
                tot_loss +=maxi
                maxi = maxi / len(batch['input_ids'])
                maxi.backward()
                optimizer.step()
                if cntr%200 == 0:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cntr': cntr  # Add any additional information you want>
                    }
                    model_path = 'var_runs_iter/model_{}_{}_{}.pth'.format(run_>
                    torch.save(checkpoint,model_path)
#                print(list(model.parameters())[0].grad)
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=maxi.item())
            except:
                continue
        tot_loss/=len(myDs)
        print(tot_loss)
        if tot_loss < best_loss:
            best_loss = tot_loss
        model_path = 'var_runs_iter/model_{}_{}'.format(run_int, epoch)
        torch.save(model.state_dict(), model_path)


# In[ ]:


if __name__ == "__main__":
    my_func()


# In[ ]:





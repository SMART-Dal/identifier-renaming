#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
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

# import csv


# from trl import PPOTrainer, PPOConfig, AutoModelForMaskedLM, create_reference_model
# from trl.core import respond_to_batch
nltk.download("punkt")
nltk.download("words")
nltk.download('wordnet')


# In[2]:


# java_keywords = [
#     "abstract",
#     "assert",
#     "boolean",
#     "break",
#     "byte",
#     "case",
#     "catch",
#     "char",
#     "class",
#     "const",
#     "continue",
#     "default",
#     "do",
#     "double",
#     "else",
#     "enum",
#     "extends",
#     "exception",
#     "error",
#     "method",
#     "builder",
#     "null",
#     "one",
#     "two",
#     "three",
#     "array",
#     "callback",
#     "zero",
#     "parameter",
#     "parameters",
#     "final",
#     "finally",
#     "float",
#     "for",
#     "goto",
#     "if",
#     "implements",
#     "import",
#     "instanceof",
#     "int",
#     "interface",
#     "long",
#     "native",
#     "new",
#     "package",
#     "private",
#     "protected",
#     "public",
#     "return",
#     "short",
#     "static",
#     "strictfp",
#     "string",
#     "super",
#     "switch",
#     "synchronized",
#     "this",
#     "throw",
#     "throws",
#     "transient",
#     "try",
#     "void",
#     "volatile",
#     "while"
# ]
# REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
# BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS =nltk.corpus.stopwords.words('english')


# In[3]:


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
#         if name.startswith("model.roberta.encoder.layer.0"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.1"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.2"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.3"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.4"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.5"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.6"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.7"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.8"):
#             param.requires_grad = False
#         if name.startswith("model.roberta.encoder.layer.9"):
#             param.requires_grad = False
    return model


# In[4]:


class MyDataset(Dataset):
    def __init__(self,file_name):
        df1 = pd.read_csv(file_name)
        df1 = df1.fillna("")
        res = df1['X']
#         ab = df1['X']
#         res = [sub.replace("<mask>", "[MASK]") for sub in ab]
        self.X_list = res
        self.y_list = df1['y']
    def __len__(self):
        return len(self.X_list)
    def __getitem__(self,idx):
        mapi = []
        mapi.append(self.X_list[idx])
        mapi.append(self.y_list[idx])
        return mapi


# In[5]:


# old_inp = []
# old_mhs = []
class Step1_model(nn.Module):
    def __init__(self, hidden_size=512):
#         global old_inp
#         global old_mhs
#         self.oi = old_inp
#         self.old_mhs = old_mhs
        super(Step1_model, self).__init__()
        self.hidden_size = hidden_size
#         self.model = AutoModel.from_pretrained("roberta-base")
#         self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#         self.config = AutoConfig.from_pretrained("roberta-base")
        self.model = AutoModelForMaskedLM.from_pretrained('microsoft/graphcodebert-base')
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.config = AutoConfig.from_pretrained("microsoft/graphcodebert-base")
        self.linear_layer = nn.Linear(self.model.config.vocab_size, self.model.config.vocab_size)

#         self.model = AutoModelForMaskedLM.from_pretrained('bert-base-cased')
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#         self.config = AutoConfig.from_pretrained("bert-base-cased")
        for param in self.model.base_model.parameters():
            param.requires_grad = True
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
            # Calculate the average word embedding for each string
            embedding1 = sum(token.vector for token in tokens1) / len(tokens1)
            embedding2 = sum(token.vector for token in tokens2) / len(tokens2)
            # Calculate the cosine similarity between the embeddings
            w1= LA.norm(embedding1)
            w2= LA.norm(embedding2)
            distance = 1 - (embedding1.dot(embedding2) / (w1 * w2))
            dist = torch.Tensor([distance])
        except:
            dist = torch.Tensor([1])
        return dist
#     def compute_loss(self, logits, target_word):
#         # Apply softmax to obtain probabilities
#         probabilities = F.softmax(logits, dim=-1)
#         log_probs = torch.log(probabilities)
#         target_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', target_word)
#         joined_sentence = " ".join(target_list)
#         target_tokens = self.tokenizer.tokenize(joined_sentence)
#         loss = 0
#         for j in range(len(target_tokens)):
#             target_index = self.tokenizer.convert_tokens_to_ids(target_tokens[j])
#             # Retrieve the probability of the target word
#             target_prob = probabilities[:, target_index]  # Assuming target_index is known
#             # Compute the negative log-likelihood loss
#             l = -torch.log(target_prob)
#             loss+=l
#         return {'loss':loss,'log_probs':log_probs}
    def forward(self, mapi):
        english_dict = set(wal.words())
        X_init = mapi[0]
        y = mapi[1]
        print(y)
        nl = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        lb = ' '.join(nl).lower()
        x = self.tokenizer.tokenize(lb)
        num_sub_tokens_label = len(x)
        X_init = X_init.replace("[MASK]", " ".join([tokenizer.mask_token] * num_sub_tokens_label))
        preds = []
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
                # get required padding length
                pad_len = 512 - input_id_chunks[i].shape[0]
                # check if tensor length satisfies required chunk size
                if pad_len > 0:
                    # if padding length is more than 0, we must add padding
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
#             if u<8:
#                 print("maski: ", maski)
            print('number of mask tok',u)
            print('number of seq', ad)
            with torch.no_grad():
                output = self.model(**input_dict)
            last_hidden_state = output[0].squeeze()
            l_o_l_sa = []
            lhs_agg = []
            if len(maski) == 1:
                masked_pos = maski[0]
                lhs_agg.append(last_hidden_state)
                for k in masked_pos:
                    l_o_l_sa.append(last_hidden_state[k])
            else:
                for p in range(len(maski)):
                    lhs_agg.append(last_hidden_state[p])
                    masked_pos = maski[p]
                    for k in masked_pos:
                        l_o_l_sa.append(last_hidden_state[p][k])
            sum_state = l_o_l_sa[0]
            lhs = lhs_agg[0]
            for i in range(len(lhs_agg)):
                if i == 0:
                    continue
                lhs+=lhs_agg[i]
            lhs/=len(lhs_agg)
            for i in range(len(l_o_l_sa)):
                if i == 0:
                    continue
                sum_state += l_o_l_sa[i]
            yip = len(l_o_l_sa)
            sum_state /= yip
    #         try:
            idx = torch.topk(sum_state, k=1, dim=0)[1]
            qw = [self.tokenizer.decode(i.item()).strip() for i in idx][0]
            preds.append(qw)
            xl = X_init.split()
            xxl = []
            for p in range(len(xl)):
                if xl[p] == tokenizer.mask_token:
                    if p != 0 and xl[p-1] == tokenizer.mask_token:
                        xxl.append(xl[p])
                        continue
                    xxl.append(qw)
                    continue
                xxl.append(xl[p])
            X_init = " ".join(xxl)
        we = preds[0]
        for t in range(len(preds)):
            if t == 0:
                continue
            we+=preds[t].capitalize()

        word_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', we)
        z = 0
        while z < len(word_list) - 1:
            word1 = word_list[z].lower()
            word2 = word_list[z + 1].lower()
            merged_word = word1 + word2

            if word1 not in english_dict and word2 not in english_dict:
                # Merge the 2 words and insert the resulting word at index (z)
                word_list[z] = merged_word
                word_list.pop(z + 1)

            elif word1 in english_dict and word2 not in english_dict:
                # Combine the words to see if the resulting word is in the dictionary
                if merged_word in english_dict:
                    # Merge the words and insert the merged word at index (z)
                    word_list[z] = merged_word
                    word_list.pop(z + 1)
                else:
                    if not (z+2)<len(word_list):
                        z+=1
                        continue
                    a = merged_word+word_list[z + 2].lower()
                    if a in english_dict:
                        word_list[z] = a
                        word_list.pop(z + 1)
                        word_list.pop(z + 2)
                    else:
                        z+=1
                        continue
            elif word1 not in english_dict and word2 in english_dict:
                # Combine the words to see if the resulting word is in the dictionary
                if merged_word in english_dict:
                    # Merge the words and insert the merged word at index (z)
                    word_list[z] = merged_word
                    word_list.pop(z + 1)
                else:
                    if not (z+2)<len(word_list):
                        z+=1
                        continue
                    a = merged_word+word_list[z + 2].lower()
                    if a in english_dict:
                        word_list[z] = a
                        word_list.pop(z + 1)
                        word_list.pop(z + 2)
                    else:
                        z+=1
                        continue
            else:
                z += 1
                continue
            z+=1
        fin_str = ""
        for o in range(len(word_list)):
            if o == 0:
                fin_str+=word_list[o].lower()
                continue
            fin_str+=word_list[o].lower().capitalize()
        word = fin_str
#         except:
#             word = "NA"         
        pred_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', word)
        target_list = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))|[a-z]+|\d+', y)
        target_string = ' '.join(pred_list)
        predicted_string = ' '.join(target_list)
        precision, recall, f1_score, _ = precision_recall_fscore_support([target_string], [predicted_string], average='micro')
        rank = 0
        for d, wordh in enumerate(pred_list):
            if d >= len(target_list):
                break
            if wordh == target_list[d]:
                rank = d + 1
                break
        mrr = 1.0 / rank if rank > 0 else 0.0
        target_set = set(target_list)
        predicted_set = set(pred_list)

        # Calculate Precision at K for K=3 (top 3 predictions)
        K = 3
        top_k_predictions = pred_list[:K]

        # Count the number of correct predictions in the top K
        correct_predictions = sum(1 for word in top_k_predictions if word in target_set)

        # Calculate Precision at K
        pak = correct_predictions / K
        print ("Guess : ",word)
#         maxi = Variable(torch.tensor(0.5, dtype=torch.float), requires_grad = True)
        maxi = Variable(0.2*self.loss_func2(word,y) + 0.8*self.loss_func1(word,y), requires_grad = True)
#         maxi.requires_grad()
        
        
#         logits = self.linear_layer(lhs)
#         cl = self.compute_loss(logits,y)
#         if len(old_inp) == 0:
#             old_log_probs = torch.rand_like(cl['log_probs'])
#             logits1 = torch.rand_like(logits)
#             hits = torch.rand_like(last_hidden_state[0])
#         else:
#             with torch.no_grad():
#                 op = self.model(**self.oi[-1])
#             self.oi.clear()
#             self.oi.append(input_dict)
#             last_hidden_state1 = op[0].squeeze()
#             lhs_agg1 = []
#             if len(maski) == 1:
#                 lhs_agg1.append(last_hidden_state1)
#             else:
#                 for p in range(len(maski)):
#                     lhs_agg1.append(last_hidden_state1[p])
#             lhs1 = lhs_agg1[0]
#             for i in range(len(lhs_agg1)):
#                 if i == 0:
#                     continue
#                 lhs1+=lhs_agg1[i]
#             lhs1/=len(lhs_agg1)
#             logits1 = self.linear_layer(lhs1)
#             old_probabilities = F.softmax(logits1, dim=-1)
#             old_log_probs = torch.log(old_probabilities)
#             hits = self.old_mhs[-1]
#             if len(self.old_mhs == 5):
#                 self.old_mhs.clear()
#             self.old_mhs.append(sum_state)
            
        return {'pak':pak,'mrr':mrr,'f1':f1_score,'returned_word':word, 'actual_pred':word, 'loss':maxi}


# In[6]:


flag = 0
def train_one_epoch(transformer_model, epoch_index, tb_writer, dataset,scheduler):
    global flag
    global myDs
    f1 = 0
    mrr = 0
    pak = 0
    for batch in dataset:
        p = 0
        inputs = batch
        optimizer.zero_grad()
        try:
            for i in range(len(inputs[0])):
                l = []
                l.append(inputs[0][i])
                l.append(inputs[1][i])
                opi = transformer_model(l)
                pred = opi['actual_pred']
                if pred == 'NA':
                    print("a")
                    continue
                loss1 = opi['loss']
                f1 += opi['f1']
                mrr+=opi['mrr']
                pak+=opi['pak']
                if p == 0:
                    loss = Variable(loss1, requires_grad = True)
                    p+=1
                else:
                    loss = torch.cat([loss, loss1],dim = -1)
                    p+=1
        except:
            continue
        
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        if p % 1 == 0:
            print('  batch loss: {}'.format(loss))
    print(" F1: "+str(f1))
    print(" MRR: "+str(mrr))
    print(" P@3: "+str(pak))
    l = len(myDs)
    f1 /= l
    mrr /= l
    pak /= l
    print("Avg F1: "+str(f1))
    print("Avg MRR: "+str(mrr))
    print("Avg P@3: "+str(pak))
    return loss.sum()


# In[7]:


epoch_number = 0
EPOCHS = 5
run_int = 26
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = Step1_model()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
myDs=MyDataset('dat.csv')
num_training_steps = len(myDs)/24  # Adjust this based on the number of training steps
num_warmup_steps = 0.15* num_training_steps # 10-20% of num_training_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# vDs=MyDataset('valid_sing_cand.csv')
train_loader=DataLoader(myDs,batch_size=24,shuffle=True)
# validation_loader=DataLoader(vDs,batch_size=1,shuffle=False)
best_loss = torch.full((1,), fill_value=100000)
key = "<summary>"
kys = []
kcs = []
c = tokenizer.encode(key[1:-1])[1]
kcs.insert(0,c)
kys.append(key)
tokenizer.add_tokens(kys, special_tokens=True)
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    # POS tag tokens
    for j in range(1,len(kcs)+1):
        d = -1*j
        model.roberta.embeddings.word_embeddings.weight[d, :] += model.roberta.embeddings.word_embeddings.weight[kcs[j-1], :].clone()
# In[ ]:


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(model, epoch_number, writer, train_loader,scheduler)

    # We don't need gradients on to do reporting
    model.train(False)

#     running_vloss = 0.0
#     for i, vdata in enumerate(validation_loader):
#         try:
#             vinputs, vlabels = vdata
#             voutputs = model(vinputs)
#             vloss = loss_fn(voutputs, vlabels)
#             running_vloss += vloss
#         except:
#             flag+=1
#             continue

#     avg_vloss = running_vloss / (i + 1)
#     print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('LOSS train {}'.format(avg_loss))
    # Log the running loss averaged per batch
    # for both training and validation
#     writer.add_scalars('Training vs. Validation Loss',
#                     { 'Training' : avg_loss, 'Validation' : avg_vloss },
#                     epoch_number + 1)
#     writer.add_scalars('Training Loss',
#                     { 'Training' : avg_loss},
#                     epoch_number + 1)
#     writer.flush()

    # Track best performance, and save the model's state
    if avg_loss < best_loss:
        best_loss = avg_loss
    model_path = 'var_runs/model_{}_{}'.format(run_int, epoch_number)
    torch.save(model.state_dict(), model_path)
#     if avg_vloss < best_vloss:
#         best_vloss = avg_vloss
#         model_path = 'model_{}_{}'.format(timestamp, epoch_number)
#         torch.save(model.state_dict(), model_path)


    epoch_number += 1


# In[ ]:





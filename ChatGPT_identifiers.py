#!/usr/bin/env python
# coding: utf-8

# In[8]:


from openai import OpenAI
import pandas as pd
import time


# In[2]:


client = OpenAI(api_key = "") #insert API key


# In[ ]:


def call_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I have very long Java classes with all instances of one variable in the code replaced with [MASK]. I want you to predict what the ideal variable name should be that replaces [MASK]. I will provide the code from the next prompt. Output the variable name and nothing else"},
            {"role": "assistant", "content": "Sure, I'd be happy to help! Please provide the relevant code, and I'll do my best to suggest an appropriate variable name to replace [MASK]"},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.2,
        max_tokens=200
    )
    return response['choices'][0]['message']['content']


# In[6]:


df = pd.read_csv('pe.csv')
X = df['X']


# In[7]:


response_list = []
count = 0
for data in X:
    print(count, file=open('print_output.txt', 'a'))
    data = data.strip()
    response = call_gpt(data)
    print(response, file=open('print_output.txt', 'a'))
    response_list.append(response)
    time.sleep(10)
    count+=1


# In[ ]:


file_path = "generations.txt"
with open(file_path, 'w') as file:
    for sentence in response_list:
        file.write(sentence + '\n')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from javalang import tree, parser
import pandas as pd
import numpy
import javalang
from collections import defaultdict
import re
import shutil
import os


# In[2]:


# extracting identifier sorted in desceding order of their frequency
def parse_identifiers(code):
    tree = javalang.parse.parse(code)
    identifier_count = defaultdict(int)
    
    for _, node in tree.filter(javalang.tree.LocalVariableDeclaration):
        for declarator in node.declarators:
            identifier = declarator.name
            identifier_count[identifier] += 1
    
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        for parameter in node.parameters:
            identifier = parameter.name
            identifier_count[identifier] += 1
    
    sorted_identifiers = sorted(identifier_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_identifiers


# In[3]:


# extract the top "n" identifiers from the list of identifier tuples
def find_top_identifiers(identifiers):
    commonly_used_identifiers = ['i', 'j', 'k', 'result', 'output', 'temp', 'tmp', 'value', 'data', 'input', 'args',
                                 'index', 'flag', 'is', 'count', 'num', 'max', 'min', 'config', 'settings',
                                 'param', 'var']
    
    filtered_identifiers = [(identifier, count) for identifier, count in identifiers
                            if len(identifier) > 4 and identifier not in commonly_used_identifiers
                            and not re.search(r'\d$', identifier)]
#     filtered_identifiers = [(identifier, count) for identifier, count in identifiers
#                             if len(identifier) > 4 and identifier not in commonly_used_identifiers
#                             and not re.search(r'\d$', identifier) and count >= min_occurrences]
    print(filtered_identifiers)
    sorted_identifiers = sorted(filtered_identifiers, key=lambda x: x[1], reverse=True)
    top_identifiers = sorted_identifiers
    
    return top_identifiers


# In[4]:


# takes code snippets and replaces them with masks
def replace_identifier_with_mask(java_code, output_file_path, identifier, identifier_output_path):
    
    modified_content = java_code.replace(identifier, " [MASK] ")
    
    with open(output_file_path, 'w+') as output_file:
        output_file.write(modified_content)
    
    with open(identifier_output_path, 'w+') as identifier_output_file:
        identifier_output_file.write(f"{identifier}\n")


# In[6]:


# input folder contains .java files cloned from Github repositories
input_path = 'Dataset/inp-txt/'  # Replace with the actual folder path
folder_path = input_path
r = 0
for file_name in os.listdir(folder_path):
    print(r)
    try:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            java_code = file.read()
        identifiers = parse_identifiers(java_code)
    except:
        identifiers = []
    top_identifiers = find_top_identifiers(identifiers)
    # Print top identifiers for each file
    print(f"File: {file_name}")
    if r < 5:
        for identifier, count in top_identifiers:
            print(identifier, "-", count)

    # Modify files and save them with the identifier names
    # Modified file path stores new java snippets with masked identifiers as .txt files
    # r (for the original java file) and i (to distinguish identifiers from same java file)
    # are used for indexing
    if len(top_identifiers) != 0:
        for i, (identifier, count) in enumerate(top_identifiers, start=1):
            # stores the masked code snippet
            modified_file_path = f"Dataset/op-txt/{r}_{i}.txt"
            # stores the masked identifier in a different file
            id_path = f"Dataset/op-txt/id_{r}_{i}.txt"
            # Replace identifier in the new file
            replace_identifier_with_mask(java_code, modified_file_path, identifier, id_path)
        r+=1


# In[7]:


#iterates over the masked code snippets and their respective identifers to create a .csv file
code = []
iden = []
folder_path = 'Dataset/op-txt/' 
file_names = sorted(os.listdir(folder_path))
for file_name in file_names:
    if file_name.endswith(".txt"): 
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            print(file_name)
            a = file.read()
            if file_name.startswith("id"): 
                iden.append(a)
            else:
                code.append(a)
df = pd.DataFrame({'code': code, 'identifier': iden})


# In[8]:


# new_df = pd.read_csv('dat.csv')
# concatenated_df = pd.concat([df, new_df])
df.to_csv('dat.csv',index=False)


# In[ ]:





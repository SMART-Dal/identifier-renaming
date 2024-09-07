#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import google.generativeai as genai
import pandas as pd
import time
import argparse

def call_gemini(prompt, model):
    response = model.generate_content(
        """You are a helpful assistant.
        I have very long Java classes with all instances of one variable in the code replaced with [MASK]. 
        I want you to predict what the ideal variable name should be that replaces [MASK]. 
        I will provide the code from the next prompt. Output the variable name and nothing else.
        Code:
        """ + prompt
    )
    return response.text

def run_gemini(test_csv_file, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.0-pro') # Changed to Gemini 1.0 Pro
    
    df = pd.read_csv(test_csv_file)
    X = df['X']
    response_list = []
    count = 0 #edit count value
    X = X[count:]
    
    for data in X:
        print(count, file=open('print_output.txt', 'a'))
        try:
            data = data.strip()
            response = call_gemini(data, model)
            print(response, file=open('print_output.txt', 'a'))
            response_list.append(response)
        except:
            print("except hit")
            print("NA", file=open('print_output.txt', 'a'))
            response_list.append("NA")
        time.sleep(10)
        count+=1

    file_path = "generations_gemini.txt"
    with open(file_path, 'w') as file:
        for sentence in response_list:
            file.write(sentence + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv_file", type=str, help="Path to the test csv file")
    parser.add_argument("--api_key", type=str, help="Google API key")
    args = parser.parse_args()
    run_gemini(args.test_csv_file, args.api_key)


# In[ ]:





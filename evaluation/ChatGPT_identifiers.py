#!/usr/bin/env python
# coding: utf-8



from openai import OpenAI
import pandas as pd
import time, os
import argparse



def call_gpt(prompt, client):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I have very long Java classes with all instances of one variable in the code replaced with [MASK]. I want you to predict what the ideal variable name should be that replaces [MASK]. I will provide the code from the next prompt. Output the variable name and nothing else"},
            {"role": "assistant", "content": "Please provide the relevant code, and I'll do my best to suggest an appropriate variable name to replace [MASK]"},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

def run_gpt(test_csv_file, api_key):
    client = OpenAI(api_key = api_key)
    df = pd.read_csv(test_csv_file)
    X = df['X']
    response_list = []
    count = 0 #edit count value
    X = X[count:]
    for data in X:
        print(count, file=open('print_output.txt', 'a'))
        try:
            data = data.strip()
            response = call_gpt(data, client)
            print(response, file=open('print_output.txt', 'a'))
            response_list.append(response)
        except:
            print("except hit")
            print("NA", file=open('print_output.txt', 'a'))
            response_list.append("NA")
        time.sleep(10)
        count+=1

    file_path = "generations.txt"
    with open(file_path, 'w') as file:
        for sentence in response_list:
            file.write(sentence + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv_file", type=str, help="Path to the test csv file")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    args = parser.parse_args()
    run_gpt(args.test_csv_file, args.api_key)
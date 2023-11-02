# Identifier-Renaming
Generating higher quality identifier names by using context and following conventions <br>
0.0 is used to generate the csv dataset from the .java files after 4.0 <br>
1.0 is code for the classifier to predict the number of mask tokens to insert in for the variable name <br>
2.2-all is used for finetuning the GraphCodeBert model on variable names <br>
4.0 is used to create the dataset. Repositories are cloned and this code file iterates over all the files and preprocesses them for creation of dataset.<br>
procTest.ipynb processes the text generated while evaluating the trained model and generates relevant graphs<br>
class_eval.py is used to evaluate the performance of the classifer <br>
identifier_scoring.py uses two non-fine-tuned models GraphCodeBERT and CodeBERT for the metric<br>
stat_sampling.py evaluates the use of random sampling technique to predict number of mask tokens<br>
model_eval.py is the code to evaluate the trained model<br>
model_test.csv is the subset of data used to evaluate the model<br>
test.csv is the dataset used for the evaluation of the readability metric and the fine-tuned model <br>

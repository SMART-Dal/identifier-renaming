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
train_1.csv, train_2.csv, train_3.csv are the subdivisions of the training data <br>
The full dataset can be accessed<a href="https://drive.google.com/file/d/1pz8Td857p3CiglfXMVNhGpJB75IqQYwr/view?usp=sharing"> here</a>  <br>
Steps for replication:<br>
1. Install requirements.txt <br>
2. To build the dataset, edit the repository details in 4.0.py and execute the code for the desired repositories.<br>
3. Iterate over the saved files to generate the .csv file with 0.0.py<br>
4. To train the model on the desired dataset, edit the dataset and name for the saved model in 2.2-all.py and execute the code<br>
5. The remaining code files have a multitude of uses detailed above<br>
